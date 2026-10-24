"""Gradio UI: Đăng ký (5 góc) | Điểm danh real-time | Quản lý & Lịch sử."""

import os

# Fully offline: no Gradio / Ultralytics telemetry or update checks.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("YOLO_OFFLINE", "True")

import time  # noqa: E402
from datetime import date, datetime

import cv2
import gradio as gr
import numpy as np

from core.attendance import AttendanceService, FaceState
from core.embedder import Embedder
from core.enrollment import POSE_HINTS, POSES, EnrollmentSession
from core.hybrid_detector import HybridDetector
from core.pipeline import FacePipeline
from core.repository import Repository
from core.vector_store import VectorStore
from utils.config import get_settings
from utils.face_utils import draw_face
from utils.logger import get_logger

log = get_logger("app")
settings = get_settings()

STATE_COLORS = {  # BGR
    FaceState.UNKNOWN: (60, 60, 230),
    FaceState.VERIFYING: (0, 200, 255),
    FaceState.CHECKED_IN: (80, 220, 80),
    FaceState.COOLDOWN: (200, 160, 60),
}
PLACEHOLDER = np.full((112, 112, 3), 40, np.uint8)
STREAM_EVERY = 0.15
# Raw (un-mirrored) frames so left/right poses are the subject's own; 720p for far faces.
WEBCAM = gr.WebcamOptions(
    mirror=False, constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}}
)


def _missing_models() -> list[str]:
    names = (settings.yolo_model, settings.landmarker_model, settings.embedder_model)
    return [n for n in names if not settings.model_path(n).exists()]


class App:
    def __init__(self):
        self.pipeline = FacePipeline(HybridDetector(settings), Embedder(settings))
        self.repo = Repository(settings.db_path)
        self.store = VectorStore(settings.qdrant_url, settings.collection, settings.embedding_dim)
        self.attendance = AttendanceService(settings, self.repo)

    # ---------- Enrollment ----------
    @staticmethod
    def _session(state: dict | None) -> dict:
        if state is None:
            state = {"session": EnrollmentSession(settings), "frame": None}
        return state

    @staticmethod
    def _gallery(session: EnrollmentSession) -> list[tuple[np.ndarray, str]]:
        items = []
        for pose in POSES:
            cap = session.captures.get(pose)
            if cap is None:
                items.append((PLACEHOLDER, f"✗ {pose}"))
            else:
                img = cv2.cvtColor(cap.aligned, cv2.COLOR_BGR2RGB)
                items.append((img, f"✓ {pose} (yaw {cap.yaw:+.0f}°, pitch {cap.pitch:+.0f}°)"))
        return items

    def _status(self, session: EnrollmentSession, message: str) -> str:
        done = len(POSES) - len(session.missing)
        return f"**{done}/{len(POSES)} góc** — {message}"

    def _offer_frame(self, frame_rgb: np.ndarray, state: dict, manual: bool):
        session: EnrollmentSession = state["session"]
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        faces = self.pipeline.analyze(frame)
        if not faces:
            target = session.target
            msg = POSE_HINTS[target] if target else "Đã đủ 5 góc, bấm Lưu"
            return frame_rgb, self._status(session, f"Không thấy khuôn mặt. {msg}")
        # Enrollment is close-range: take the largest face.
        face = max(faces, key=lambda f: f.detection.size)
        result = session.offer(face, time.monotonic(), manual=manual)
        det = face.detection
        ok = result.pose is not None and result.pose == (result.captured or session.target)
        label = f"{result.pose or '?'} yaw {det.yaw or 0:+.0f} pitch {det.pitch or 0:+.0f}"
        draw_face(frame, det.bbox, label, (80, 220, 80) if ok else (0, 165, 255))
        msg = result.message
        if len(faces) > 1:
            msg += " (có nhiều người trong khung hình)"
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self._status(session, msg)

    def enroll_stream(self, frame_rgb, state):
        state = self._session(state)
        if frame_rgb is None:
            return gr.skip(), gr.skip(), gr.skip(), state
        state["frame"] = frame_rgb
        before = len(state["session"].captures)
        preview, status = self._offer_frame(frame_rgb, state, manual=False)
        gallery = (
            self._gallery(state["session"])
            if len(state["session"].captures) != before else gr.skip()
        )
        return preview, status, gallery, state

    def enroll_manual(self, frame_rgb, state):
        """'Chụp' button (frame = last webcam frame) or uploaded image."""
        state = self._session(state)
        frame_rgb = frame_rgb if frame_rgb is not None else state.get("frame")
        if frame_rgb is None:
            return gr.skip(), "Chưa có ảnh", gr.skip(), state
        preview, status = self._offer_frame(frame_rgb, state, manual=True)
        return preview, status, self._gallery(state["session"]), state

    def enroll_retake(self, pose, state):
        state = self._session(state)
        session = state["session"]
        session.reset(pose if pose != "Tất cả" else None)
        return self._status(session, POSE_HINTS[session.target]), self._gallery(session), state

    def enroll_save(self, name: str, employee_id: str, state):
        state = self._session(state)
        session: EnrollmentSession = state["session"]
        name, employee_id = (name or "").strip(), (employee_id or "").strip()
        if not name or not employee_id:
            return "❌ Nhập họ tên và mã nhân viên", gr.skip(), state
        if not session.complete:
            return f"❌ Còn thiếu góc: {', '.join(session.missing)}", gr.skip(), state
        if self.repo.get_person_by_employee_id(employee_id):
            return f"❌ Mã nhân viên {employee_id} đã tồn tại", gr.skip(), state
        dup = self.store.search(session.captures["front"].embedding, settings.search_top_k)
        if dup is not None and dup.score >= settings.match_threshold:
            return (
                f"❌ Khuôn mặt đã được đăng ký cho {dup.name} ({dup.employee_id}),"
                f" score {dup.score:.2f}",
                gr.skip(), state,
            )
        person_id = self.repo.add_person(name, employee_id)
        try:
            self.store.add_faces(
                person_id, name, employee_id,
                {pose: cap.embedding for pose, cap in session.captures.items()},
            )
        except Exception:
            self.repo.delete_person(person_id)
            log.exception("Qdrant insert failed")
            return "❌ Lỗi lưu vector vào Qdrant (Docker đã chạy chưa?)", gr.skip(), state
        session.reset()
        log.info("Enrolled %s (%s) as person %d", name, employee_id, person_id)
        return f"✅ Đã đăng ký {name} ({employee_id})", self._gallery(session), state

    # ---------- Attendance ----------
    def attend_stream(self, frame_rgb):
        if frame_rgb is None:
            return gr.skip(), gr.skip()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        faces = self.pipeline.analyze(frame)
        observations = [
            (f.detection.bbox, self.store.search(f.embedding, settings.search_top_k))
            for f in faces
        ]
        statuses = self.attendance.update(observations, time.time())
        logged = False
        for st in statuses:
            if st.match is None:
                label = "Unknown"
            else:
                label = f"{st.match.name} {st.match.score:.2f}"
                if st.state == FaceState.CHECKED_IN:
                    label += " ✓"
                    logged = True
            draw_face(frame, st.bbox, label, STATE_COLORS[st.state])
        recent = self.repo.history(limit=15) if logged else gr.skip()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), recent

    # ---------- Management ----------
    def delete_person(self, person_id):
        if person_id is None:
            return "Nhập ID", self.repo.list_persons()
        person_id = int(person_id)
        self.store.delete_person(person_id)
        self.repo.delete_person(person_id)
        self.attendance.forget_person(person_id)
        return f"Đã xoá người ID {person_id}", self.repo.list_persons()

    def history(self, day_str: str):
        try:
            day = date.fromisoformat(day_str.strip()) if day_str and day_str.strip() else None
        except ValueError:
            return gr.skip(), None, "Ngày không hợp lệ (YYYY-MM-DD)"
        df = self.repo.history(day)
        path = settings.data_dir / f"attendance_{day or 'all'}.csv"
        df.to_csv(path, index=False)
        return df, str(path), f"{len(df)} bản ghi"

    # ---------- UI ----------
    def build(self) -> gr.Blocks:
        with gr.Blocks(title="Face Attendance") as demo:
            gr.Markdown("# Điểm danh khuôn mặt")
            with gr.Tab("Đăng ký"):
                enroll_state = gr.State(None)
                with gr.Row():
                    name = gr.Textbox(label="Họ tên")
                    emp = gr.Textbox(label="Mã nhân viên")
                status = gr.Markdown(
                    f"**0/{len(POSES)} góc** — {POSE_HINTS['front']}. Tự chụp khi giữ đúng góc."
                )
                with gr.Row():
                    cam = gr.Image(
                        sources=["webcam"], streaming=True, webcam_options=WEBCAM,
                        type="numpy", label="Webcam",
                    )
                    preview = gr.Image(label="Preview", interactive=False)
                gallery = gr.Gallery(
                    value=self._gallery(EnrollmentSession(settings)), columns=5, height=200,
                    label="Các góc đã chụp", allow_preview=False,
                )
                with gr.Row():
                    snap = gr.Button("📸 Chụp")
                    retake_pose = gr.Dropdown(["Tất cả", *POSES], value="Tất cả", label="Chụp lại")
                    retake = gr.Button("↺ Chụp lại")
                    save = gr.Button("💾 Lưu", variant="primary")
                with gr.Accordion("Hoặc tải ảnh lên (từng góc)", open=False):
                    upload = gr.Image(sources=["upload"], type="numpy", label="Ảnh")
                    add_upload = gr.Button("Thêm ảnh")

                cam.stream(
                    self.enroll_stream, [cam, enroll_state],
                    [preview, status, gallery, enroll_state],
                    stream_every=STREAM_EVERY, concurrency_limit=1, show_progress="hidden",
                )
                none_frame = gr.State(None)
                snap.click(
                    self.enroll_manual, [none_frame, enroll_state],
                    [preview, status, gallery, enroll_state],
                )
                add_upload.click(
                    self.enroll_manual, [upload, enroll_state],
                    [preview, status, gallery, enroll_state],
                )
                retake.click(
                    self.enroll_retake, [retake_pose, enroll_state],
                    [status, gallery, enroll_state],
                )
                save.click(
                    self.enroll_save, [name, emp, enroll_state], [status, gallery, enroll_state]
                )

            with gr.Tab("Điểm danh"):
                with gr.Row():
                    acam = gr.Image(
                        sources=["webcam"], streaming=True, webcam_options=WEBCAM,
                        type="numpy", label="Camera",
                    )
                    aview = gr.Image(label="Nhận diện", interactive=False)
                recent = gr.Dataframe(self.repo.history(limit=15), label="Điểm danh gần nhất")
                acam.stream(
                    self.attend_stream, [acam], [aview, recent],
                    stream_every=STREAM_EVERY, concurrency_limit=1, show_progress="hidden",
                )

            with gr.Tab("Quản lý & Lịch sử"):
                persons = gr.Dataframe(self.repo.list_persons, label="Người đã đăng ký")
                with gr.Row():
                    del_id = gr.Number(label="ID cần xoá", precision=0)
                    del_btn = gr.Button("🗑 Xoá", variant="stop")
                    refresh = gr.Button("↻ Làm mới")
                del_msg = gr.Markdown()
                del_btn.click(self.delete_person, [del_id], [del_msg, persons])
                refresh.click(self.repo.list_persons, None, persons)

                with gr.Row():
                    day = gr.Textbox(
                        value=date.today().isoformat(), label="Ngày (YYYY-MM-DD, trống = tất cả)"
                    )
                    load = gr.Button("Xem lịch sử")
                hist_msg = gr.Markdown()
                hist = gr.Dataframe(label="Lịch sử")
                csv = gr.File(label="Export CSV")
                load.click(self.history, [day], [hist, csv, hist_msg])
        return demo


def main():
    missing = _missing_models()
    if missing:
        raise SystemExit(
            f"Thiếu model: {missing}. Chạy: uv run python scripts/download_models.py"
        )
    log.info("Starting at %s", datetime.now().isoformat(timespec="seconds"))
    App().build().launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()

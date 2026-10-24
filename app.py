"""Gradio UI: Đăng ký (7 góc) | Điểm danh real-time | Quản lý & Lịch sử."""

import os

# Fully offline: no Gradio / Ultralytics telemetry or update checks.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("YOLO_OFFLINE", "True")

import threading  # noqa: E402
import time
from datetime import date, datetime
from urllib.parse import urlsplit

import cv2
import gradio as gr
import numpy as np

from core.attendance import AttendanceService, FaceState
from core.embedder import Embedder
from core.enhancer import FaceEnhancer
from core.enrollment import DONE_HINT, POSE_HINTS, POSES, EnrollmentSession
from core.hybrid_detector import HybridDetector
from core.pipeline import FacePipeline
from core.repository import Repository
from core.rtsp import RTSP_SCHEMES, RtspCamera, redact
from core.vector_store import VectorStore
from utils.config import get_settings
from utils.face_utils import draw_badge, draw_face
from utils.logger import get_logger

log = get_logger("app")
settings = get_settings()

STATE_COLORS = {  # BGR
    FaceState.UNKNOWN: (60, 60, 230),
    FaceState.VERIFYING: (0, 200, 255),
    FaceState.CHECKED_IN: (80, 220, 80),
    FaceState.COOLDOWN: (200, 160, 60),
}
BADGE_COLOR = (60, 170, 40)  # BGR green
PLACEHOLDER = np.full((112, 112, 3), 40, np.uint8)
STREAM_EVERY = 0.15
# Gradio processes only the latest frame, so a short interval just means less idle time.
ATTEND_STREAM_EVERY = 0.05


CAM_FRONT, CAM_BACK, CAM_RTSP = "Camera trước", "Camera sau", "Camera IP (RTSP)"


def _facing_webcam(facing: str) -> gr.WebcamOptions:
    """Phone camera by facing mode ("user" = selfie, "environment" = back).

    Raw (un-mirrored) frames so left/right poses are the subject's own; 720p for far faces.
    """
    return gr.WebcamOptions(
        mirror=False,
        constraints={
            "video": {"facingMode": facing, "width": {"ideal": 1280}, "height": {"ideal": 720}}
        },
    )


# The raw webcam can't be hidden (hiding unmounts it and stops the stream), so the
# processed output is stacked on top of it; only the webcam's control bar stays above.
CSS = """
.cam-stage { position: relative; height: min(75vh, 720px); gap: 0 !important; }
.cam-stage .cam-src, .cam-stage .cam-out {
    position: absolute !important; inset: 0; height: 100% !important; margin: 0 !important;
}
.cam-stage .cam-out { z-index: 1; pointer-events: none; }
.cam-stage .cam-src .button-wrap { z-index: 2; }
.cam-stage .cam-src [title="grant webcam access"] { position: relative; z-index: 2; }
.cam-stage .cam-out img { object-fit: contain; }
"""


def _camera_stage(output_label: str) -> tuple[gr.Image, gr.Image, gr.Image]:
    """Front + back webcam under one output image; one webcam is shown at a time."""
    with gr.Column(elem_classes="cam-stage"):
        front = gr.Image(
            sources=["webcam"], streaming=True, webcam_options=_facing_webcam("user"),
            type="numpy", show_label=False, height="100%", elem_classes="cam-src",
        )
        back = gr.Image(
            sources=["webcam"], streaming=True, webcam_options=_facing_webcam("environment"),
            type="numpy", show_label=False, height="100%", elem_classes="cam-src",
            visible=False,
        )
        # JPEG: Gradio's default WebP costs ~55 ms per 720p frame vs ~3 ms.
        output = gr.Image(
            label=output_label, interactive=False, format="jpeg", buttons=[],
            height="100%", elem_classes="cam-out",
        )
    return front, back, output


def _show_camera(side: str):
    return gr.update(visible=side == CAM_FRONT), gr.update(visible=side == CAM_BACK)


def _show_attend_source(source: str):
    return (*_show_camera(source), gr.update(visible=source == CAM_RTSP))


def _missing_models() -> list[str]:
    names = (settings.yolo_model, settings.landmarker_model, settings.embedder_model)
    if settings.enhance_enabled:
        names += (settings.denoise_model, settings.deblur_model)
    return [n for n in names if not settings.model_path(n).exists()]


class App:
    def __init__(self):
        enhancer = FaceEnhancer(settings) if settings.enhance_enabled else None
        self.pipeline = FacePipeline(HybridDetector(settings), Embedder(settings), enhancer)
        self.repo = Repository(settings.db_path)
        self.store = VectorStore(settings.qdrant_url, settings.collection, settings.embedding_dim)
        self.attendance = AttendanceService(settings, self.repo)
        self._badge: tuple[str, float] | None = None  # (text, show until)
        # One server-side IP camera at a time: starting a new one (any client) stops the old.
        self._rtsp: RtspCamera | None = None
        self._rtsp_lock = threading.Lock()

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
            msg = POSE_HINTS[target] if target else DONE_HINT
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
        return self._attend_frame(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    def _attend_frame(self, frame: np.ndarray):
        """Recognize + log + annotate one BGR frame -> (RGB preview, recent table or skip)."""
        faces = self.pipeline.analyze(frame)
        observations = [
            (f.detection.bbox, self.store.search(f.embedding, settings.search_top_k))
            for f in faces
        ]
        now = time.time()
        statuses = self.attendance.update(observations, now)
        logged = False
        for face, st in zip(faces, statuses, strict=True):
            if st.match is None:
                label = "Unknown"
            else:
                label = f"{st.match.name} {st.match.score:.2f}"
                if st.state == FaceState.CHECKED_IN:
                    label += " ✓"
                    logged = True
                    text = (
                        f"✓ {st.match.name} ({st.match.employee_id})"
                        f" {datetime.fromtimestamp(now):%H:%M}"
                    )
                    self._badge = (text, now + settings.badge_seconds)
            if face.enhanced:  # NAFNet restored this crop (cv2 font: ASCII only)
                label += " +" + "/".join(face.enhanced)
            draw_face(frame, st.bbox, label, STATE_COLORS[st.state])
        if self._badge and now < self._badge[1]:
            draw_badge(frame, self._badge[0], BADGE_COLOR, settings.badge_font)
        recent = self.repo.history(limit=15) if logged else gr.skip()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), recent

    def rtsp_stream(self, url: str):
        """Generator: pull frames from an IP camera and stream the annotated preview."""
        url = (url or "").strip()
        if urlsplit(url).scheme.lower() not in RTSP_SCHEMES or not urlsplit(url).hostname:
            yield gr.skip(), gr.skip(), "❌ URL phải có dạng `rtsp://[user:pass@]ip[:554]/đường-dẫn`"
            return
        cam = RtspCamera(url, settings.rtsp_timeout_seconds, settings.rtsp_reconnect_seconds)
        with self._rtsp_lock:
            if self._rtsp is not None:
                self._rtsp.close()
            self._rtsp = cam
        shown = redact(url)
        yield gr.skip(), gr.skip(), f"⏳ Đang kết nối `{shown}`…"
        seq, status = 0, None
        try:
            while not cam.closed:
                frame, seq_new = cam.read(seq, timeout=1.0)
                if frame is None:
                    msg = cam.error or "⏳ Đang chờ khung hình…"
                    if msg != status:
                        status = msg
                        yield gr.skip(), gr.skip(), msg
                    continue
                seq = seq_new
                h, w = frame.shape[:2]
                preview, recent = self._attend_frame(frame.copy())
                scale = settings.rtsp_display_max_width / w
                if scale < 1:
                    preview = cv2.resize(
                        preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                    )
                msg = f"🟢 `{shown}` — {w}×{h}"
                yield preview, recent, msg if msg != status else gr.skip()
                status = msg
        finally:
            # Runs on Stop / source switch (Gradio cancels the generator) and on errors.
            cam.close()
            with self._rtsp_lock:
                if self._rtsp is cam:
                    self._rtsp = None
        # Reached when the camera was closed elsewhere (Stop, or another client took it over).
        yield gr.skip(), gr.skip(), "⏹ Đã dừng"

    def rtsp_stop(self):
        with self._rtsp_lock:
            if self._rtsp is not None:
                self._rtsp.close()
                self._rtsp = None
        return "⏹ Đã dừng"

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

    def clear_history(self, day_str: str):
        try:
            day = date.fromisoformat(day_str.strip()) if day_str and day_str.strip() else None
        except ValueError:
            return gr.skip(), None, "Ngày không hợp lệ (YYYY-MM-DD)", gr.skip()
        deleted = self.repo.clear_history(day)
        self.attendance.reset_cooldowns()
        log.info("Cleared %d attendance records (%s)", deleted, day or "all")
        return (
            self.repo.history(day), None, f"Đã xoá {deleted} bản ghi ({day or 'tất cả'})",
            self.repo.history(limit=15),
        )

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
                cam_side = gr.Radio([CAM_FRONT, CAM_BACK], value=CAM_FRONT, label="Camera")
                cam, cam_back, preview = _camera_stage("Preview")
                gallery = gr.Gallery(
                    value=self._gallery(EnrollmentSession(settings)), columns=len(POSES),
                    height=200, label="Các góc đã chụp", allow_preview=False,
                )
                with gr.Row():
                    snap = gr.Button("📸 Chụp")
                    retake_pose = gr.Dropdown(["Tất cả", *POSES], value="Tất cả", label="Chụp lại")
                    retake = gr.Button("↺ Chụp lại")
                    save = gr.Button("💾 Lưu", variant="primary")
                with gr.Accordion("Hoặc tải ảnh lên (từng góc)", open=False):
                    upload = gr.Image(sources=["upload"], type="numpy", label="Ảnh")
                    add_upload = gr.Button("Thêm ảnh")

                for webcam in (cam, cam_back):
                    webcam.stream(
                        self.enroll_stream, [webcam, enroll_state],
                        [preview, status, gallery, enroll_state],
                        stream_every=STREAM_EVERY, concurrency_limit=1, show_progress="hidden",
                    )
                cam_side.change(_show_camera, cam_side, [cam, cam_back])
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
                acam_side = gr.Radio(
                    [CAM_FRONT, CAM_BACK, CAM_RTSP], value=CAM_FRONT, label="Camera"
                )
                with gr.Group(visible=False) as rtsp_box:
                    with gr.Row(equal_height=True):
                        rtsp_url = gr.Textbox(
                            value=settings.rtsp_url, label="RTSP URL", type="password",
                            placeholder="rtsp://user:pass@192.168.1.10:554/stream1", scale=4,
                        )
                        rtsp_start = gr.Button("▶ Kết nối", variant="primary", scale=1)
                        rtsp_stop = gr.Button("■ Dừng", scale=1)
                    rtsp_status = gr.Markdown(padding=True)
                acam, acam_back, aview = _camera_stage("Nhận diện")
                recent = gr.Dataframe(self.repo.history(limit=15), label="Điểm danh gần nhất")
                for webcam in (acam, acam_back):
                    webcam.stream(
                        self.attend_stream, [webcam], [aview, recent],
                        stream_every=ATTEND_STREAM_EVERY, concurrency_limit=1,
                        show_progress="hidden",
                    )
                rtsp_run = gr.on(
                    [rtsp_start.click, rtsp_url.submit], self.rtsp_stream, [rtsp_url],
                    [aview, recent, rtsp_status], show_progress="hidden",
                )
                rtsp_stop.click(self.rtsp_stop, None, rtsp_status, cancels=[rtsp_run])
                acam_side.change(
                    _show_attend_source, acam_side, [acam, acam_back, rtsp_box]
                ).then(self.rtsp_stop, None, rtsp_status, cancels=[rtsp_run])

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
                    clear = gr.Button("🗑 Xoá lịch sử", variant="stop")
                hist_msg = gr.Markdown()
                hist = gr.Dataframe(label="Lịch sử")
                csv = gr.File(label="Export CSV")
                load.click(self.history, [day], [hist, csv, hist_msg])
                clear.click(self.clear_history, [day], [hist, csv, hist_msg, recent])
        return demo


def main():
    missing = _missing_models()
    if missing:
        raise SystemExit(
            f"Thiếu model: {missing}. Chạy: uv run python scripts/download_models.py"
        )
    log.info("Starting at %s", datetime.now().isoformat(timespec="seconds"))
    ssl = {}
    if settings.ssl_certfile.exists() and settings.ssl_keyfile.exists():
        ssl = {
            "ssl_certfile": str(settings.ssl_certfile),
            "ssl_keyfile": str(settings.ssl_keyfile),
            "ssl_verify": False,  # self-signed
        }
    else:
        log.warning(
            "No TLS cert: phone cameras need HTTPS. Run: uv run python scripts/make_cert.py"
        )
    App().build().launch(
        server_name=settings.server_host, server_port=settings.server_port, css=CSS, **ssl
    )


if __name__ == "__main__":
    main()

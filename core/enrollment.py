"""Guided multi-angle enrollment: front, left/right (turned and full profile), up, down.

UI-independent state machine. "left"/"right" are the subject's own left/right in an
un-mirrored camera frame (turning to your left moves the nose towards image right).
"""

from dataclasses import dataclass, field

import numpy as np

from core.types import FaceResult
from utils.config import Settings
from utils.face_utils import sharpness

POSES = ("front", "left", "full_left", "right", "full_right", "up", "down")
POSE_HINTS = {
    "front": "Nhìn thẳng vào camera",
    "left": "Quay mặt sang TRÁI",
    "full_left": "Quay HẲN mặt sang TRÁI (nhìn ngang)",
    "right": "Quay mặt sang PHẢI",
    "full_right": "Quay HẲN mặt sang PHẢI (nhìn ngang)",
    "up": "Ngẩng mặt LÊN",
    "down": "Cúi mặt XUỐNG",
}
DONE_HINT = f"Đã đủ {len(POSES)} góc, bấm Lưu"


def classify_pose(yaw: float, pitch: float, s: Settings) -> str | None:
    if abs(yaw) > s.full_yaw_max or abs(pitch) > s.max_angle:
        return None
    if abs(yaw) < s.front_max_angle and abs(pitch) < s.front_max_angle:
        return "front"
    if abs(pitch) < s.tilt_pitch_min:
        if abs(yaw) >= s.full_yaw_min:
            return "full_left" if yaw > 0 else "full_right"
        if yaw >= s.side_yaw_min:
            return "left"
        if yaw <= -s.side_yaw_min:
            return "right"
    if abs(yaw) < s.side_yaw_min:
        if pitch >= s.tilt_pitch_min:
            return "up"
        if pitch <= -s.tilt_pitch_min:
            return "down"
    return None


@dataclass
class Capture:
    aligned: np.ndarray
    embedding: np.ndarray
    yaw: float
    pitch: float


@dataclass
class OfferResult:
    pose: str | None  # detected pose of the offered face
    captured: str | None  # pose slot filled by this offer, if any
    message: str


@dataclass
class EnrollmentSession:
    settings: Settings
    captures: dict[str, Capture] = field(default_factory=dict)
    _pending_pose: str | None = None
    _pending_since: float = 0.0

    @property
    def missing(self) -> list[str]:
        return [p for p in POSES if p not in self.captures]

    @property
    def target(self) -> str | None:
        """Front first (it is the identity reference), then the rest in order."""
        return self.missing[0] if self.missing else None

    @property
    def complete(self) -> bool:
        return not self.missing

    def reset(self, pose: str | None = None) -> None:
        if pose is None:
            self.captures.clear()
        else:
            self.captures.pop(pose, None)
        self._pending_pose = None

    def offer(self, face: FaceResult, now: float, manual: bool = False) -> OfferResult:
        """Feed one analyzed face. Captures it once its pose is held for `hold_seconds`
        (or immediately if `manual`) and it passes the quality gates."""
        s, det = self.settings, face.detection
        pose = classify_pose(det.yaw or 0.0, det.pitch or 0.0, s)
        target = self.target
        if target is None:
            return OfferResult(pose, None, DONE_HINT)
        if pose is None or pose not in self.missing or (target == "front" and pose != "front"):
            self._pending_pose = None
            return OfferResult(pose, None, POSE_HINTS[target])

        if det.size < s.min_face_size:
            return OfferResult(pose, None, "Mặt quá nhỏ, lại gần camera hơn")
        if sharpness(face.aligned) < s.min_sharpness:
            return OfferResult(pose, None, "Ảnh bị mờ, giữ yên")
        if self.captures:
            # Best match over all captured poses: a full profile barely resembles the frontal
            # shot, but does resemble the 30-degree one captured before it.
            sim = max(float(np.dot(c.embedding, face.embedding)) for c in self.captures.values())
            if sim < s.same_person_min_sim:
                return OfferResult(pose, None, "Không khớp với ảnh chính diện, kiểm tra lại người")

        if not manual:
            if self._pending_pose != pose:
                self._pending_pose, self._pending_since = pose, now
            if now - self._pending_since < s.hold_seconds:
                return OfferResult(pose, None, f"Giữ nguyên… ({POSE_HINTS[pose]})")

        self.captures[pose] = Capture(
            face.aligned, face.embedding, det.yaw or 0.0, det.pitch or 0.0
        )
        self._pending_pose = None
        nxt = self.target
        return OfferResult(pose, pose, POSE_HINTS[nxt] if nxt else DONE_HINT)

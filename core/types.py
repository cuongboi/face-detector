from dataclasses import dataclass, field

import numpy as np


@dataclass
class FaceDetection:
    bbox: np.ndarray  # (4,) x1, y1, x2, y2 in frame pixels
    score: float
    kps5: np.ndarray | None = None  # (5, 2) eyes, nose, mouth corners in frame pixels
    landmarks: np.ndarray | None = None  # (76, 2) Vision landmarks in frame pixels
    yaw: float | None = None  # degrees, + = nose towards image right
    pitch: float | None = None  # degrees, + = looking up
    source: str = "vision"  # where kps5 came from

    @property
    def size(self) -> float:
        return float(min(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]))


@dataclass
class FaceResult:
    """A detection carried through align + embed."""

    detection: FaceDetection
    aligned: np.ndarray  # 112x112 BGR
    embedding: np.ndarray  # (512,) L2-normalized
    enhanced: list[str] = field(default_factory=list)  # NAFNet steps applied: denoise, deblur
    extras: dict = field(default_factory=dict)

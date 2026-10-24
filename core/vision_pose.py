"""Head pose from Apple Vision for faces detected elsewhere (SCRFD).

One full-frame VNDetectFaceRectanglesRequest (revision 3 reports continuous yaw/pitch, ~10ms).
Vision only finds faces wider than ~5% of the frame, i.e. near ones such as during enrollment,
where an accurate pose matters; far faces keep the coarse keypoint estimate.
"""

import threading

import numpy as np
import objc
import Vision

from core.head_pose import pose_from_vision
from core.types import FaceDetection
from core.vision_bridge import bgr_to_cgimage
from utils.face_utils import iou
from utils.logger import get_logger

log = get_logger(__name__)

MATCH_IOU = 0.3


class VisionPose:
    def __init__(self):
        self._lock = threading.Lock()  # Vision requests are stateful

    def annotate(self, frame_bgr: np.ndarray, faces: list[FaceDetection]) -> None:
        """Set yaw/pitch on the faces Vision also sees (matched by IoU); others are left None."""
        with self._lock, objc.autorelease_pool():
            poses = self._poses(frame_bgr)
        for det in faces:
            best = max(poses, key=lambda p: iou(det.bbox, p[0]), default=None)
            if best is not None and iou(det.bbox, best[0]) >= MATCH_IOU:
                det.yaw, det.pitch = best[1]

    @staticmethod
    def _poses(frame_bgr: np.ndarray) -> list[tuple[np.ndarray, tuple[float, float]]]:
        h, w = frame_bgr.shape[:2]
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            bgr_to_cgimage(frame_bgr), {}
        )
        req = Vision.VNDetectFaceRectanglesRequest.alloc().init()
        req.setRevision_(Vision.VNDetectFaceRectanglesRequestRevision3)
        ok, err = handler.performRequests_error_([req], None)
        if not ok:
            log.warning("Vision face rectangles failed: %s", err)
            return []
        out = []
        for obs in req.results() or []:
            if obs.yaw() is None or obs.pitch() is None:
                continue
            bb = obs.boundingBox()  # normalized, bottom-left origin
            x1, y2 = bb.origin.x * w, (1 - bb.origin.y) * h
            box = np.array([x1, y2 - bb.size.height * h, x1 + bb.size.width * w, y2], np.float32)
            out.append((box, pose_from_vision(float(obs.yaw()), float(obs.pitch()))))
        return out

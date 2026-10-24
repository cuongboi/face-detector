"""Apple Vision face detection + 76-point landmarks + head pose, on the Neural Engine.

One handler call per frame: face rectangles on the full frame and on an NxN grid of
overlapping tiles (Vision misses faces smaller than ~5% of its input, i.e. people at 5-10m),
merged with NMS, then landmarks for the kept faces. Rectangles revision 3 is what gives
continuous yaw/pitch; the landmarks request carries them over from its input observations.
"""

import threading

import numpy as np
import objc
import Vision

from core.aligner import kps5_from_vision
from core.head_pose import pose_from_kps5, pose_from_vision
from core.types import FaceDetection
from core.vision_bridge import bgr_to_cgimage
from utils.config import Settings
from utils.face_utils import iou
from utils.logger import get_logger

log = get_logger(__name__)

KPS_REGIONS = (
    "leftEye", "rightEye", "leftPupil", "rightPupil", "noseCrest", "outerLips",
)


def tile_grid(n: int, overlap: float) -> list[tuple[float, float, float, float]]:
    """NxN normalized (x, y, w, h) tiles sharing `overlap` with their neighbours.

    Symmetric in y, so it is valid for Vision's bottom-left origin as is. Rounded because Vision
    rejects a region of interest that is outside [0, 1] by even 1e-16.
    """
    if n <= 1:
        return []
    size = round(1 / (n - (n - 1) * overlap), 6)
    starts = [round((1 - size) * i / (n - 1), 6) for i in range(n)]
    return [(x, y, size, size) for y in starts for x in starts]


def _overlaps(a: np.ndarray, b: np.ndarray, thresh: float) -> bool:
    """IoU, or containment: a tile edge cutting a face yields a partial box inside the full one."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return iou(a, b) >= thresh or (smaller > 0 and iw * ih / smaller >= 0.6)


def nms(items: list[tuple[FaceDetection, object]], thresh: float) -> list:
    """Greedy NMS over (detection, payload) pairs, highest score first."""
    kept: list = []
    for item in sorted(items, key=lambda it: it[0].score, reverse=True):
        if not any(_overlaps(item[0].bbox, k[0].bbox, thresh) for k in kept):
            kept.append(item)
    return kept


class VisionDetector:
    def __init__(self, settings: Settings):
        self.s = settings
        self.tiles = tile_grid(settings.vision_tile_grid, settings.vision_tile_overlap)
        # Vision requests are stateful objects; keep calls serialized (Gradio worker threads).
        self._lock = threading.Lock()
        self.detect(np.zeros((360, 640, 3), np.uint8))  # first call loads the models (~0.5s)
        log.info("VisionDetector: full frame + %d tiles", len(self.tiles))

    def detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        with self._lock, objc.autorelease_pool():
            return self._detect(frame_bgr)

    @staticmethod
    def _rect_request(roi=None):
        req = Vision.VNDetectFaceRectanglesRequest.alloc().init()
        req.setRevision_(Vision.VNDetectFaceRectanglesRequestRevision3)
        if roi is not None:
            req.setRegionOfInterest_(roi)
        return req

    def _detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        h, w = frame_bgr.shape[:2]
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            bgr_to_cgimage(frame_bgr), {}
        )
        reqs = [self._rect_request()] + [
            self._rect_request(((x, y), (tw, th))) for x, y, tw, th in self.tiles
        ]
        ok, err = handler.performRequests_error_(reqs, None)
        if not ok:
            log.warning("Vision face rectangles failed: %s", err)
            return []

        candidates = []
        for req in reqs:
            for obs in req.results() or []:
                det = FaceDetection(bbox=self._bbox(obs, w, h), score=float(obs.confidence()))
                if det.score >= self.s.vision_conf and det.size >= self.s.min_face_size:
                    candidates.append((det, obs))
        faces = nms(candidates, self.s.vision_nms_iou)
        if not faces:
            return []

        lm_req = Vision.VNDetectFaceLandmarksRequest.alloc().init()
        lm_req.setRevision_(Vision.VNDetectFaceLandmarksRequestRevision3)
        lm_req.setConstellation_(Vision.VNRequestFaceLandmarksConstellation76Points)
        lm_req.setInputFaceObservations_([obs for _, obs in faces])
        ok, err = handler.performRequests_error_([lm_req], None)
        if not ok:
            log.warning("Vision face landmarks failed: %s", err)
            return []

        out = []
        # Results are NOT in input order (they get shuffled between calls), so each detection
        # is rebuilt from its own landmarks observation rather than zipped with `faces`.
        for obs in lm_req.results() or []:
            lm = obs.landmarks()
            if lm is None:
                continue
            regions = {}
            for name in (*KPS_REGIONS, "allPoints"):
                region = getattr(lm, name)()
                if region is not None and region.pointCount():
                    regions[name] = self._points(region, w, h)
            det = FaceDetection(bbox=self._bbox(obs, w, h), score=float(obs.confidence()))
            try:
                det.kps5 = kps5_from_vision(regions)
            except KeyError:  # region missing, e.g. heavy occlusion
                continue
            det.landmarks = regions.get("allPoints")
            if obs.yaw() is not None and obs.pitch() is not None:
                det.yaw, det.pitch = pose_from_vision(float(obs.yaw()), float(obs.pitch()))
            else:
                det.yaw, det.pitch = pose_from_kps5(det.kps5)
            out.append(det)
        return out

    @staticmethod
    def _bbox(obs, w: int, h: int) -> np.ndarray:
        """Pixel xyxy (y down) from a Vision observation's normalized, bottom-left-origin box."""
        bb = obs.boundingBox()
        x1, y2 = bb.origin.x * w, (1 - bb.origin.y) * h
        return np.array([x1, y2 - bb.size.height * h, x1 + bb.size.width * w, y2], np.float32)

    @staticmethod
    def _points(region, w: int, h: int) -> np.ndarray:
        pts = region.pointsInImageOfSize_((w, h))
        return np.array(
            [(pts[i].x, h - pts[i].y) for i in range(region.pointCount())], np.float32
        )

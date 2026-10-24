"""SCRFD-10G face detector (InsightFace buffalo_l det_10g) as Core ML, on the Neural Engine.

One-stage, anchor-free-style CNN tuned for tiny faces (WIDER-Face hard), so people at 5-10m are
found in a single pass. It also gives the 5 keypoints ArcFace was trained on, which keeps
alignment stable on 25-60px faces where landmark models fall apart.

Frames wider/taller than `det_max_native_width` x (same aspect as the input) are also scanned as
overlapping tiles, so the network never sees faces shrunk below ~0.6x; the full-frame pass still
catches faces too big for a tile. Head pose comes from Apple Vision when it sees the face too.
"""

import math
import threading
from concurrent.futures import ThreadPoolExecutor

import coremltools as ct
import cv2
import numpy as np

from core.head_pose import pose_from_kps5
from core.types import FaceDetection
from core.vision_pose import VisionPose
from utils.config import Settings
from utils.face_utils import nms
from utils.logger import get_logger

log = get_logger(__name__)

INPUT_W, INPUT_H = 1280, 736  # fixed (16:9, multiple of 32) so the model stays on the ANE
STRIDES = (8, 16, 32)
ANCHORS = 2  # per location
OUTPUTS = tuple(f"{kind}_{s}" for kind in ("score", "bbox", "kps") for s in STRIDES)


def anchor_centers(stride: int) -> np.ndarray:
    """(H*W*ANCHORS, 2) x, y pixel centers in model input space, row-major like the heads."""
    h, w = INPUT_H // stride, INPUT_W // stride
    ys, xs = np.mgrid[:h, :w]
    centers = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32) * stride
    return np.repeat(centers, ANCHORS, axis=0)


CENTERS = {s: anchor_centers(s) for s in STRIDES}


def decode(out: dict[str, np.ndarray], conf: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw heads -> (scores (N,), boxes (N, 4) xyxy, kps (N, 5, 2)) in model input pixels."""
    scores, boxes, kps = [], [], []
    for s in STRIDES:
        score = out[f"score_{s}"].reshape(-1)
        keep = score >= conf
        if not keep.any():
            continue
        c = CENTERS[s][keep]
        d = out[f"bbox_{s}"].reshape(-1, 4)[keep] * s
        k = out[f"kps_{s}"].reshape(-1, 10)[keep] * s
        scores.append(score[keep])
        boxes.append(np.concatenate([c - d[:, :2], c + d[:, 2:]], axis=1))
        kps.append(k.reshape(-1, 5, 2) + c[:, None, :])
    if not scores:
        return (
            np.zeros(0, np.float32),
            np.zeros((0, 4), np.float32),
            np.zeros((0, 5, 2), np.float32),
        )
    return np.concatenate(scores), np.concatenate(boxes), np.concatenate(kps)


def plan_tiles(w: int, h: int, max_native_w: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """Pixel (x, y, w, h) windows: the full frame, plus a grid of overlapping tiles when the
    frame is larger than `max_native_w` x (matching height) so small faces keep their size."""
    windows = [(0, 0, w, h)]
    max_native_h = round(max_native_w * INPUT_H / INPUT_W)
    nx, ny = math.ceil(w / max_native_w), math.ceil(h / max_native_h)
    if nx * ny == 1:
        return windows
    tw, th = min(w, math.ceil(w / nx) + overlap), min(h, math.ceil(h / ny) + overlap)
    xs = [round((w - tw) * i / (nx - 1)) if nx > 1 else 0 for i in range(nx)]
    ys = [round((h - th) * i / (ny - 1)) if ny > 1 else 0 for i in range(ny)]
    return windows + [(x, y, tw, th) for y in ys for x in xs]


def letterbox(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit into the top-left of the model canvas, keeping aspect -> (NCHW RGB blob, scale)."""
    h, w = img.shape[:2]
    scale = min(INPUT_W / w, INPUT_H / h)
    nw, nh = round(w * scale), round(h * scale)
    canvas = np.zeros((INPUT_H, INPUT_W, 3), np.uint8)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=interp)
    return cv2.dnn.blobFromImage(canvas, swapRB=True), scale


class ScrfdDetector:
    def __init__(self, settings: Settings):
        self.s = settings
        path = settings.model_path(settings.detector_model)
        self.model = ct.models.CompiledMLModel(str(path), compute_units=ct.ComputeUnit.ALL)
        self.pose = VisionPose() if settings.pose_vision else None
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="scrfd")
        self.detect(np.zeros((360, 640, 3), np.uint8))  # warm-up
        log.info("ScrfdDetector: %s (Core ML), Vision pose %s", path.name, bool(self.pose))

    def detect(self, frame_bgr: np.ndarray, pose: bool = True) -> list[FaceDetection]:
        """`pose=False` skips Apple Vision (yaw/pitch then come from the 5 keypoints)."""
        h, w = frame_bgr.shape[:2]
        windows = plan_tiles(w, h, self.s.det_max_native_width, self.s.det_tile_overlap)
        candidates = []
        # Resizing runs in parallel (OpenCV releases the GIL); the ANE runs one predict at a time.
        blobs = self._pool.map(
            lambda r: letterbox(frame_bgr[r[1] : r[1] + r[3], r[0] : r[0] + r[2]]), windows
        )
        with self._lock:
            for (x0, y0, _, _), (blob, scale) in zip(windows, blobs, strict=True):
                scores, boxes, kps = decode(self.model.predict({"input": blob}), self.s.det_conf)
                offset = np.array([x0, y0], np.float32)
                for sc, box, kp in zip(scores, boxes, kps, strict=True):
                    det = FaceDetection(
                        bbox=box / scale + np.tile(offset, 2),
                        score=float(sc),
                        kps5=(kp / scale + offset).astype(np.float32),
                        source="scrfd",
                    )
                    if det.size >= self.s.min_face_size:
                        candidates.append(det)
        faces = nms(candidates, self.s.det_nms_iou)
        for det in faces:
            det.bbox = np.clip(det.bbox, 0, [w, h, w, h]).astype(np.float32)
        if pose and self.pose is not None and faces:
            self.pose.annotate(frame_bgr, faces)
        for det in faces:
            if det.yaw is None:
                det.yaw, det.pitch = pose_from_kps5(det.kps5)
        return faces

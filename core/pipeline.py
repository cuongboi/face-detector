import numpy as np

from core.aligner import align_5pt
from core.embedder import Embedder
from core.enhancer import FaceEnhancer
from core.types import FaceResult
from core.vision_detector import VisionDetector
from utils.face_utils import expand_bbox


class FacePipeline:
    """detect -> (restore) -> align -> embed, shared by enrollment and attendance."""

    def __init__(
        self, detector: VisionDetector, embedder: Embedder, enhancer: FaceEnhancer | None = None
    ):
        self.detector = detector
        self.embedder = embedder
        self.enhancer = enhancer

    def analyze(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        detections = self.detector.detect(frame_bgr)
        work, steps = self._restore(frame_bgr, detections)
        aligned, kept, kept_steps = [], [], []
        for d, st in zip(detections, steps, strict=True):
            try:
                aligned.append(align_5pt(work, d.kps5))
            except ValueError:  # degenerate keypoints, e.g. eyes overlapping in full profile
                continue
            kept.append(d)
            kept_steps.append(st)
        detections, steps = kept, kept_steps
        embeddings = self.embedder.embed_batch(aligned)
        return [
            FaceResult(detection=d, aligned=a, embedding=e, enhanced=st)
            for d, a, e, st in zip(detections, aligned, embeddings, steps, strict=True)
        ]

    def _restore(self, frame_bgr: np.ndarray, detections) -> tuple[np.ndarray, list[list[str]]]:
        """Run NAFNet on each face's crop (native resolution, before the warp) in a frame copy."""
        if self.enhancer is None or not detections:
            return frame_bgr, [[] for _ in detections]
        h, w = frame_bgr.shape[:2]
        ratio = self.enhancer.s.expand_ratio
        work = frame_bgr.copy()
        steps = []
        for d in detections:
            x1, y1, x2, y2 = expand_bbox(d.bbox, ratio, w, h)
            restored, st = self.enhancer.enhance(frame_bgr[y1:y2, x1:x2])
            if st:
                work[y1:y2, x1:x2] = restored
            steps.append(st)
        return work, steps

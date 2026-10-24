import numpy as np

from core.aligner import align_5pt
from core.embedder import Embedder
from core.hybrid_detector import HybridDetector
from core.types import FaceResult


class FacePipeline:
    """detect -> align -> embed, shared by enrollment and attendance."""

    def __init__(self, detector: HybridDetector, embedder: Embedder):
        self.detector = detector
        self.embedder = embedder

    def analyze(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        detections = self.detector.detect(frame_bgr)
        aligned = [align_5pt(frame_bgr, d.kps5) for d in detections]
        embeddings = self.embedder.embed_batch(aligned)
        return [
            FaceResult(detection=d, aligned=a, embedding=e)
            for d, a, e in zip(detections, aligned, embeddings, strict=True)
        ]

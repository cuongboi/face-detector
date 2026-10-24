"""ArcFace (InsightFace buffalo_l w600k_r50) via onnxruntime, without the insightface package."""

import cv2
import numpy as np
import onnxruntime as ort

from utils.config import Settings
from utils.logger import get_logger

log = get_logger(__name__)


class Embedder:
    def __init__(self, settings: Settings):
        available = ort.get_available_providers()
        providers = [
            p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider") if p in available
        ]
        path = str(settings.model_path(settings.embedder_model))
        try:
            self.session = ort.InferenceSession(path, providers=providers)
        except Exception:  # CoreML can reject some graphs; CPU always works
            log.warning("CoreML provider failed, falling back to CPU", exc_info=True)
            self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        log.info("Embedder providers: %s", self.session.get_providers())

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        return self.embed_batch([aligned_bgr])[0]

    def embed_batch(self, faces_bgr: list[np.ndarray]) -> np.ndarray:
        if not faces_bgr:
            return np.zeros((0, 512), np.float32)
        blob = cv2.dnn.blobFromImages(
            faces_bgr, scalefactor=1 / 127.5, size=(112, 112),
            mean=(127.5, 127.5, 127.5), swapRB=True,
        )
        out = self.session.run(None, {self.input_name: blob})[0]
        return out / np.linalg.norm(out, axis=1, keepdims=True)

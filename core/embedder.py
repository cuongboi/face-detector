"""ArcFace (InsightFace buffalo_l w600k_r50) as a compiled Core ML model, on the Neural Engine.

Converted by scripts/convert_models.py with the (x - 127.5) / 127.5 normalization baked in,
so the input is plain RGB 0-255 NCHW.
"""

import threading

import coremltools as ct
import cv2
import numpy as np

from utils.config import Settings
from utils.logger import get_logger

log = get_logger(__name__)

MAX_BATCH = 16  # upper bound of the model's flexible batch dimension


class Embedder:
    def __init__(self, settings: Settings):
        path = settings.model_path(settings.embedder_model)
        self.model = ct.models.CompiledMLModel(str(path), compute_units=ct.ComputeUnit.ALL)
        self._lock = threading.Lock()
        self.embed_batch([np.zeros((112, 112, 3), np.uint8)])  # warm-up
        log.info("Embedder: %s (Core ML)", path.name)

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        return self.embed_batch([aligned_bgr])[0]

    def embed_batch(self, faces_bgr: list[np.ndarray]) -> np.ndarray:
        if not faces_bgr:
            return np.zeros((0, 512), np.float32)
        blob = cv2.dnn.blobFromImages(faces_bgr, scalefactor=1.0, size=(112, 112), swapRB=True)
        with self._lock:
            out = np.concatenate([
                self.model.predict({"input": blob[i : i + MAX_BATCH]})["output"]
                for i in range(0, len(blob), MAX_BATCH)
            ])
        return out / np.linalg.norm(out, axis=1, keepdims=True)

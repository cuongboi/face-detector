"""NAFNet restoration of face crops: SIDD denoising for dark frames, GoPro deblurring for blur.

Compiled Core ML models with a fixed square input (scripts/convert_models.py), so they run on the
Neural Engine (~15 ms per crop on M1); crops are reflect-padded up to that size.
Runs on face crops only (not whole frames).
"""

import threading

import coremltools as ct
import cv2
import numpy as np

from utils.config import Settings
from utils.face_utils import apply_clahe, sharpness
from utils.logger import get_logger

log = get_logger(__name__)


def mean_luma(img_bgr: np.ndarray) -> float:
    return float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[..., 0].mean())


class FaceEnhancer:
    def __init__(self, settings: Settings):
        self.s = settings
        self.size = settings.enhance_max_side  # the models' fixed input side
        self.denoiser = self._load(settings.denoise_model)
        self.deblurrer = self._load(settings.deblur_model)
        self._lock = threading.Lock()
        # Warm up now instead of on the first live frame.
        blank = np.zeros((64, 64, 3), np.uint8)
        self._run(self.denoiser, blank)
        self._run(self.deblurrer, blank)
        log.info("FaceEnhancer: Core ML %dx%d", self.size, self.size)

    def _load(self, name: str) -> ct.models.CompiledMLModel:
        path = str(self.s.model_path(name))
        return ct.models.CompiledMLModel(path, compute_units=ct.ComputeUnit.ALL)

    def _run(self, model: ct.models.CompiledMLModel, img_bgr: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        # Reflect padding keeps the border statistics natural; zeros would read as dark/blur.
        mode = cv2.BORDER_REFLECT_101 if min(h, w) > 1 else cv2.BORDER_REPLICATE
        padded = cv2.copyMakeBorder(img_bgr, 0, self.size - h, 0, self.size - w, mode)
        x = cv2.dnn.blobFromImage(padded, scalefactor=1 / 255, swapRB=True)
        y = model.predict({"input": x})["output"][0, :, :h, :w].transpose(1, 2, 0)[..., ::-1]
        return np.ascontiguousarray((y * 255).round().astype(np.uint8))

    def enhance(self, crop_bgr: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Restore a face crop if it is dark and/or blurry.

        Returns the (same-shape) image and the steps applied; no steps = input returned as is.
        """
        if crop_bgr.size == 0:
            return crop_bgr, []
        h, w = crop_bgr.shape[:2]
        scale = min(1.0, self.s.enhance_max_side / max(h, w))
        img = crop_bgr
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        steps = []
        with self._lock:
            if mean_luma(img) < self.s.enhance_dark_luma:
                # NAFNet removes sensor noise but doesn't brighten; CLAHE does that afterwards.
                img = apply_clahe(self._run(self.denoiser, img))
                steps.append("denoise")
            # Checked after denoising: dark, low-contrast crops otherwise always look blurry.
            if sharpness(img) < self.s.enhance_blur_sharpness:
                img = self._run(self.deblurrer, img)
                steps.append("deblur")

        if not steps:
            return crop_bgr, []
        if scale < 1:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        return img, steps

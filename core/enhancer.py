"""NAFNet restoration of face crops: SIDD denoising for dark frames, GoPro deblurring for blur.

Runs on face crops only (not whole frames) so it stays cheap on Apple Silicon (MPS).
"""

import threading

import cv2
import numpy as np
import torch

from core.nafnet_arch import GOPRO_WIDTH32, SIDD_WIDTH32, NAFNet
from utils.config import Settings
from utils.face_utils import apply_clahe, sharpness
from utils.logger import get_logger

log = get_logger(__name__)


def mean_luma(img_bgr: np.ndarray) -> float:
    return float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[..., 0].mean())


class FaceEnhancer:
    def __init__(self, settings: Settings):
        self.s = settings
        self.device = torch.device(settings.enhance_device)
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            self.device = torch.device("cpu")
        self.denoiser = self._load(settings.denoise_model, SIDD_WIDTH32)
        self.deblurrer = self._load(settings.deblur_model, GOPRO_WIDTH32)
        # MPS is not safe to drive from several Gradio worker threads at once.
        self._lock = threading.Lock()
        # First MPS call compiles kernels (~1s); do it now instead of on the first live frame.
        blank = np.zeros((64, 64, 3), np.uint8)
        self._run(self.denoiser, blank)
        self._run(self.deblurrer, blank)
        log.info("FaceEnhancer on %s", self.device)

    def _load(self, name: str, config: dict) -> NAFNet:
        model = NAFNet(**config)
        state = torch.load(self.s.model_path(name), map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("params", state))
        return model.eval().to(self.device)

    @torch.inference_mode()
    def _run(self, model: NAFNet, img_bgr: np.ndarray) -> np.ndarray:
        rgb = np.ascontiguousarray(img_bgr[..., ::-1])
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1)[None].float() / 255
        y = model(x).clamp_(0, 1)[0].permute(1, 2, 0).mul_(255).round_().byte().cpu().numpy()
        return np.ascontiguousarray(y[..., ::-1])

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

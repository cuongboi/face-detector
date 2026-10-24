import cv2
import numpy as np
import pytest
import torch

from core.nafnet_arch import NAFNet
from core.pipeline import FacePipeline
from core.types import FaceDetection
from utils.config import get_settings

s = get_settings()
weights_missing = not all(s.model_path(n).exists() for n in (s.denoise_model, s.deblur_model))


def _bright_sharp(rng) -> np.ndarray:
    return rng.integers(100, 255, (120, 100, 3), dtype=np.uint8)


def test_nafnet_keeps_odd_shapes():
    model = NAFNet(width=8, middle_blk_num=1, enc_blk_nums=(1, 1), dec_blk_nums=(1, 1)).eval()
    with torch.inference_mode():
        out = model(torch.rand(1, 3, 37, 53))
    assert out.shape == (1, 3, 37, 53)


@pytest.fixture(scope="module")
def enhancer():
    if weights_missing:
        pytest.skip("NAFNet weights not downloaded")
    from core.enhancer import FaceEnhancer

    return FaceEnhancer(s)


def test_enhancer_skips_good_crops(enhancer):
    crop = _bright_sharp(np.random.default_rng(0))
    out, steps = enhancer.enhance(crop)
    assert steps == [] and out is crop


def test_enhancer_denoises_dark_crops(enhancer):
    rng = np.random.default_rng(0)
    dark = np.clip(rng.normal(25, 12, (150, 130, 3)), 0, 255).astype(np.uint8)
    out, steps = enhancer.enhance(dark)
    assert "denoise" in steps
    assert out.shape == dark.shape and out.dtype == np.uint8


def test_enhancer_deblurs_blurry_crops(enhancer):
    rng = np.random.default_rng(0)
    blurry = cv2.GaussianBlur(_bright_sharp(rng), (21, 21), 6)
    out, steps = enhancer.enhance(blurry)
    assert steps == ["deblur"] and out.shape == blurry.shape


def test_enhancer_downscales_large_crops(enhancer):
    big = cv2.GaussianBlur(_bright_sharp(np.random.default_rng(0)), (21, 21), 6)
    big = cv2.resize(big, (400, 480))
    out, steps = enhancer.enhance(big)
    assert steps and out.shape == big.shape


class _FakeDetector:
    def detect(self, frame):
        kps = np.array([[40, 50], [70, 50], [55, 65], [43, 80], [67, 80]], np.float32)
        return [FaceDetection(bbox=np.array([30, 30, 80, 90], np.float32), score=0.9, kps5=kps)]


class _FakeEmbedder:
    def embed_batch(self, faces):
        return np.array([f.astype(np.float32).mean(axis=(0, 1)) for f in faces]).reshape(-1, 3)


class _FakeEnhancer:
    def __init__(self):
        self.s = s
        self.crops = []

    def enhance(self, crop):
        self.crops.append(crop.shape)
        return np.full_like(crop, 200), ["denoise"]


def test_pipeline_applies_enhancer_to_face_crop_only():
    frame = np.zeros((200, 200, 3), np.uint8)
    enh = _FakeEnhancer()
    res = FacePipeline(_FakeDetector(), _FakeEmbedder(), enh).analyze(frame)
    assert len(enh.crops) == 1 and res[0].enhanced == ["denoise"]
    assert res[0].aligned.mean() > 100  # aligned from the restored crop
    assert frame.max() == 0  # input frame untouched


def test_pipeline_without_enhancer():
    frame = np.zeros((200, 200, 3), np.uint8)
    res = FacePipeline(_FakeDetector(), _FakeEmbedder()).analyze(frame)
    assert res[0].enhanced == [] and res[0].aligned.max() == 0


def test_pipeline_drops_faces_that_cannot_be_aligned():
    class _ProfileDetector(_FakeDetector):
        def detect(self, frame):
            dets = super().detect(frame)
            dets[0].kps5 = np.full((5, 2), 50, np.float32)  # all keypoints collapsed
            return dets

    frame = np.zeros((200, 200, 3), np.uint8)
    assert FacePipeline(_ProfileDetector(), _FakeEmbedder()).analyze(frame) == []

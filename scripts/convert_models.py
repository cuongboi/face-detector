"""Convert the downloaded ONNX / PyTorch weights to compiled Core ML models (FP16, Neural Engine).

Saved as compiled .mlmodelc bundles at a stable path: macOS caches the Neural Engine build per
path, so the app loads them in ~0.1s (an .mlpackage is recompiled to a new temp path on every
load, ~15s for all three).

    uv run --group convert python scripts/convert_models.py

Each model is checked against its source model before being saved; re-running skips
models that already exist.
"""

import shutil
import sys
from pathlib import Path

import coremltools as ct
import cv2
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.embedder import MAX_BATCH as ARCFACE_MAX_BATCH  # noqa: E402
from core.nafnet_arch import GOPRO_WIDTH32, SIDD_WIDTH32, NAFNet  # noqa: E402
from utils.config import get_settings  # noqa: E402

# Source files, as written by scripts/download_models.py.
ARCFACE_ONNX = "w600k_r50.onnx"
NAFNET_SIDD_PTH = "NAFNet-SIDD-width32.pth"
NAFNET_GOPRO_PTH = "NAFNet-GoPro-width32.pth"

MIN_COSINE = 0.999
MIN_PSNR = 40.0


class _ArcFaceRGB255(nn.Module):
    """ArcFace with its (x - 127.5) / 127.5 input normalization baked in."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - 127.5) / 127.5)


class _NAFNetUnit(nn.Module):
    """NAFNet with the output clamp folded in, so callers only rescale to 0-255."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).clamp(0, 1)


def _smooth_images(n: int, size: int, high: float) -> np.ndarray:
    """(n, 3, size, size) bicubic-upsampled noise: closer to real crops than white noise, which
    is out of distribution for both models and exaggerates fp16 differences."""
    rng = np.random.default_rng(0)
    small = rng.uniform(0, high, (n, size // 8, size // 8, 3)).astype(np.float32)
    big = [cv2.resize(im, (size, size), interpolation=cv2.INTER_CUBIC) for im in small]
    return np.clip(np.stack(big), 0, high).transpose(0, 3, 1, 2).copy()


def _convert(module: nn.Module, example: torch.Tensor, shape) -> ct.models.MLModel:
    traced = torch.jit.trace(module.eval(), example)
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS13,
    )


def _save(model: ct.models.MLModel, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(model.get_compiled_model_path(), tmp)
    tmp.rename(dest)
    print(f"✓ {dest.name}")


def convert_arcface(src: Path, dest: Path) -> None:
    import onnxruntime as ort
    from onnx2torch import convert

    model = _ArcFaceRGB255(convert(str(src)))
    shape = (ct.RangeDim(1, ARCFACE_MAX_BATCH, default=1), 3, 112, 112)
    ml = _convert(model, torch.rand(1, 3, 112, 112) * 255, shape)

    x = _smooth_images(ARCFACE_MAX_BATCH, 112, 255)
    sess = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    ref = sess.run(None, {sess.get_inputs()[0].name: (x - 127.5) / 127.5})[0]
    out = ml.predict({"input": x})["output"]
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    cos = (ref * out).sum(axis=1)
    print(f"  ArcFace cosine vs ONNX: min {cos.min():.5f}  mean {cos.mean():.5f}")
    if cos.min() < MIN_COSINE:
        raise SystemExit(f"ArcFace Core ML drifts from ONNX (min cosine {cos.min():.5f})")
    _save(ml, dest)


def convert_nafnet(src: Path, config: dict, dest: Path, size: int) -> None:
    net = NAFNet(**config)
    state = torch.load(src, map_location="cpu", weights_only=True)
    net.load_state_dict(state.get("params", state))
    model = _NAFNetUnit(net).eval()
    ml = _convert(model, torch.rand(1, 3, size, size), (1, 3, size, size))

    x = _smooth_images(1, size, 1)
    with torch.inference_mode():
        ref = model(torch.from_numpy(x)).numpy()
    out = ml.predict({"input": x})["output"]
    psnr = 10 * np.log10(1 / max(float(np.mean((ref - out) ** 2)), 1e-12))
    print(f"  {src.stem} PSNR vs PyTorch: {psnr:.1f} dB")
    if psnr < MIN_PSNR:
        raise SystemExit(f"{src.name}: Core ML drifts from PyTorch ({psnr:.1f} dB)")
    _save(ml, dest)


def main() -> None:
    s = get_settings()
    size = s.enhance_max_side
    jobs = [
        (ARCFACE_ONNX, s.embedder_model, convert_arcface),
        (NAFNET_SIDD_PTH, s.denoise_model, lambda a, b: convert_nafnet(a, SIDD_WIDTH32, b, size)),
        (NAFNET_GOPRO_PTH, s.deblur_model, lambda a, b: convert_nafnet(a, GOPRO_WIDTH32, b, size)),
    ]
    for src_name, dest_name, fn in jobs:
        src, dest = s.model_path(src_name), s.model_path(dest_name)
        if dest.exists():
            print(f"✓ {dest_name} (exists)")
        elif not src.exists():
            print(f"✗ {dest_name}: {src_name} missing, run scripts/download_models.py")
        else:
            print(f"→ {src_name} -> {dest_name}")
            fn(src, dest)


if __name__ == "__main__":
    main()

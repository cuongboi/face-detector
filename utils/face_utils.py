import base64
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def expand_bbox(
    bbox: np.ndarray, ratio: float, width: int, height: int
) -> tuple[int, int, int, int]:
    """Grow an (x1, y1, x2, y2) box by `ratio` on each side, square it, clip to image."""
    x1, y1, x2, y2 = bbox.astype(float)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * ratio)
    nx1, ny1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    nx2, ny2 = int(min(width, cx + side / 2)), int(min(height, cy + side / 2))
    return nx1, ny1, nx2, ny2


def encode_snapshot(img_bgr: np.ndarray, width: int, quality: int = 85) -> str:
    """JPEG (downscaled to `width`, never upscaled) as base64 text for the database."""
    h, w = img_bgr.shape[:2]
    if w > width:
        img_bgr = cv2.resize(img_bgr, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_snapshot(b64: str) -> np.ndarray:
    """Base64 JPEG -> RGB (for Gradio)."""
    buf = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """Equalize luminance only; helps backlit / low-light faces."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a, b = cv2.split(lab)
    l_ch = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_ch)
    return cv2.cvtColor(cv2.merge((l_ch, a, b)), cv2.COLOR_LAB2BGR)


def sharpness(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def draw_face(
    img_bgr: np.ndarray, bbox: np.ndarray, label: str, color: tuple[int, int, int]
) -> None:
    x1, y1, x2, y2 = bbox.astype(int)
    # Sized for a 1280px-wide view, so labels stay readable when 4K previews are downscaled.
    k = max(1.0, img_bgr.shape[1] / 1280)
    thick = max(round(2 * k), (x2 - x1) // 60)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thick)
    if label:
        scale, font_thick, pad = 0.6 * k, max(2, round(2 * k)), round(6 * k)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, font_thick)
        ty = max(th + pad, y1)
        cv2.rectangle(img_bgr, (x1, ty - th - pad), (x1 + tw + pad, ty), color, -1)
        cv2.putText(
            img_bgr, label, (x1 + pad // 2, ty - pad // 2 - 1), cv2.FONT_HERSHEY_SIMPLEX,
            scale, (0, 0, 0), font_thick,
        )


@lru_cache(maxsize=8)
def _font(path: Path, size: int) -> ImageFont.FreeTypeFont | None:
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return None


def draw_badge(
    img_bgr: np.ndarray, text: str, color: tuple[int, int, int], font_path: Path,
    top: int = 0,
) -> int:
    """Filled badge in the top-right corner, `top` px down (to stack several).

    PIL for Unicode text; cv2 fallback w/o font. Returns the y just below the badge.
    """
    h, w = img_bgr.shape[:2]
    size, pad = max(16, w // 40), max(8, w // 100)
    font = _font(font_path, size)
    if font is None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size / 30, 2)
        x1, y1 = max(0, w - tw - 3 * pad), top + pad
        cv2.rectangle(img_bgr, (x1, y1), (w - pad, y1 + th + 2 * pad), color, -1)
        cv2.putText(
            img_bgr, text, (x1 + pad, y1 + th + pad), cv2.FONT_HERSHEY_SIMPLEX,
            size / 30, (255, 255, 255), 2,
        )
        return y1 + th + 2 * pad
    left, top_, right, bottom = font.getbbox(text)
    tw, th = right - left, bottom - top_
    y1 = min(top + pad, max(0, h - th - 2 * pad))
    box = (max(0, w - tw - 3 * pad), y1, w - pad, y1 + th + 2 * pad)
    # Only convert the badge region, not the whole frame.
    x1, y1, x2, y2 = box
    roi = Image.fromarray(cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(roi)
    rgb = (color[2], color[1], color[0])
    draw.rounded_rectangle((0, 0, x2 - x1 - 1, y2 - y1 - 1), radius=pad, fill=rgb)
    draw.text((pad - left, pad - top_), text, font=font, fill=(255, 255, 255))
    img_bgr[y1:y2, x1:x2] = cv2.cvtColor(np.asarray(roi), cv2.COLOR_RGB2BGR)
    return y2

import cv2
import numpy as np


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
    thick = max(2, (x2 - x1) // 60)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thick)
    if label:
        scale = 0.6
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        ty = max(th + 6, y1)
        cv2.rectangle(img_bgr, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
        cv2.putText(
            img_bgr, label, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2
        )

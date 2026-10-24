import cv2
import numpy as np

# Canonical 5-point template for ArcFace models at 112x112 (InsightFace).
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# MediaPipe FaceMesh indices, ordered image-left to image-right to match the template.
MP_LEFT_EYE = (33, 133)
MP_RIGHT_EYE = (362, 263)
MP_NOSE = 1
MP_MOUTH_LEFT = 61
MP_MOUTH_RIGHT = 291


def kps5_from_mediapipe(landmarks: np.ndarray) -> np.ndarray:
    pts = landmarks[:, :2]
    return np.stack(
        [
            pts[list(MP_LEFT_EYE)].mean(axis=0),
            pts[list(MP_RIGHT_EYE)].mean(axis=0),
            pts[MP_NOSE],
            pts[MP_MOUTH_LEFT],
            pts[MP_MOUTH_RIGHT],
        ]
    ).astype(np.float32)


def align_5pt(img_bgr: np.ndarray, kps5: np.ndarray, size: int = 112) -> np.ndarray:
    """Similarity-warp a face so its 5 keypoints land on the ArcFace template."""
    dst = ARCFACE_TEMPLATE * (size / 112.0)
    matrix, _ = cv2.estimateAffinePartial2D(kps5.astype(np.float32), dst, method=cv2.LMEDS)
    if matrix is None:
        raise ValueError("Could not estimate alignment transform")
    return cv2.warpAffine(img_bgr, matrix, (size, size), borderValue=0)

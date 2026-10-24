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


def kps5_from_vision(regions: dict[str, np.ndarray]) -> np.ndarray:
    """5 ArcFace keypoints from Vision landmark regions (each (N, 2) in frame pixels, y down).

    Vision names eyes from the viewer's side, but pairs are sorted by x anyway so the order
    always matches the template (image-left first).
    """

    def center(pupil: str, eye: str) -> np.ndarray:
        pts = regions.get(pupil)
        return (pts if pts is not None and len(pts) else regions[eye]).mean(axis=0)

    eyes = [center("leftPupil", "leftEye"), center("rightPupil", "rightEye")]
    eyes.sort(key=lambda p: p[0])
    # noseCrest runs from between the eyes down to the tip; its lowest point is the tip.
    crest = regions["noseCrest"]
    nose = crest[crest[:, 1].argmax()]
    lips = regions["outerLips"]
    mouth = [lips[lips[:, 0].argmin()], lips[lips[:, 0].argmax()]]
    return np.stack([*eyes, nose, *mouth]).astype(np.float32)


def align_5pt(img_bgr: np.ndarray, kps5: np.ndarray, size: int = 112) -> np.ndarray:
    """Similarity-warp a face so its 5 keypoints land on the ArcFace template."""
    dst = ARCFACE_TEMPLATE * (size / 112.0)
    matrix, _ = cv2.estimateAffinePartial2D(kps5.astype(np.float32), dst, method=cv2.LMEDS)
    if matrix is None:
        raise ValueError("Could not estimate alignment transform")
    return cv2.warpAffine(img_bgr, matrix, (size, size), borderValue=0)

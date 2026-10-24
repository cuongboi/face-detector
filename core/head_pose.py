"""Head pose (yaw, pitch) in degrees.

Convention used everywhere: yaw > 0 when the nose points towards image right,
pitch > 0 when the face looks up.
"""

import math

import numpy as np


def pose_from_vision(yaw: float, pitch: float) -> tuple[float, float]:
    """From a VNFaceObservation (radians). Vision's yaw already matches; its pitch is + = down."""
    return math.degrees(yaw), -math.degrees(pitch)


def pose_from_kps5(kps5: np.ndarray) -> tuple[float, float]:
    """Coarse estimate from 5 keypoints, used when Vision reports no pose for the face."""
    le, re, nose, ml, mr = kps5.astype(float)
    eye_mid = (le + re) / 2
    mouth_mid = (ml + mr) / 2
    ied = np.linalg.norm(re - le)
    face_h = mouth_mid[1] - eye_mid[1]
    if ied < 1e-3 or face_h < 1e-3:
        return 0.0, 0.0
    # Nose tip protrudes ~0.6 inter-eye distance; its lateral shift ~ depth * sin(yaw).
    mid_x = (eye_mid[0] + mouth_mid[0]) / 2
    yaw = math.degrees(math.asin(np.clip((nose[0] - mid_x) / (0.6 * ied), -1, 1)))
    # Frontal nose sits ~55% of the way from eye line to mouth line; looking up moves it up.
    ratio = (nose[1] - eye_mid[1]) / face_h
    pitch = math.degrees(math.asin(np.clip((0.55 - ratio) / 0.45, -1, 1)))
    return yaw, pitch

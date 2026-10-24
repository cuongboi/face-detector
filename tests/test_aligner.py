import cv2
import numpy as np

from core.aligner import ARCFACE_TEMPLATE, align_5pt, kps5_from_vision
from core.head_pose import pose_from_kps5, pose_from_vision


def test_align_maps_keypoints_onto_template():
    # Template scaled x3, rotated 20° and shifted, inside a larger image.
    center = ARCFACE_TEMPLATE.mean(axis=0)
    rot = cv2.getRotationMatrix2D(tuple(map(float, center)), 20, 3.0)
    rot[:, 2] += (200, 150)
    kps = cv2.transform(ARCFACE_TEMPLATE[None], rot)[0]
    img = np.zeros((600, 600, 3), np.uint8)
    for x, y in kps:
        cv2.circle(img, (int(x), int(y)), 6, (255, 255, 255), -1)

    out = align_5pt(img, kps)

    assert out.shape == (112, 112, 3)
    for x, y in ARCFACE_TEMPLATE:
        assert out[int(round(y)), int(round(x))].max() > 200


def test_frontal_kps_gives_near_zero_pose():
    yaw, pitch = pose_from_kps5(ARCFACE_TEMPLATE)
    assert abs(yaw) < 5 and abs(pitch) < 10


def test_kps_yaw_sign_follows_nose():
    kps = ARCFACE_TEMPLATE.copy()
    kps[2, 0] += 12  # nose towards image right
    assert pose_from_kps5(kps)[0] > 15


def test_vision_pose_convention():
    yaw, pitch = pose_from_vision(np.radians(30), np.radians(10))  # Vision: + pitch = down
    assert abs(yaw - 30) < 1e-6 and abs(pitch + 10) < 1e-6


def test_kps5_from_vision_order():
    # Vision's left/right swapped on purpose: output must still be image-left first.
    regions = {
        "leftEye": np.array([[70, 50], [76, 50]], np.float32),
        "rightEye": np.array([[36, 50], [42, 50]], np.float32),
        "noseCrest": np.array([[56, 50], [56, 60], [56, 72]], np.float32),
        "outerLips": np.array([[70, 92], [56, 90], [42, 92], [56, 96]], np.float32),
    }
    kps = kps5_from_vision(regions)
    np.testing.assert_allclose(kps, [[39, 50], [73, 50], [56, 72], [42, 92], [70, 92]])

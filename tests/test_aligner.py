import cv2
import numpy as np

from core.aligner import ARCFACE_TEMPLATE, align_5pt
from core.head_pose import pose_from_kps5, pose_from_matrix


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


def test_matrix_pose_signs():
    def rot_y(deg):
        r = np.radians(deg)
        m = np.eye(4)
        m[0, 0], m[0, 2], m[2, 0], m[2, 2] = np.cos(r), np.sin(r), -np.sin(r), np.cos(r)
        return m

    yaw, pitch = pose_from_matrix(rot_y(30))
    assert abs(yaw - 30) < 1e-6 and abs(pitch) < 1e-6

"""Apple Vision detector; macOS only (Vision ships with the OS, no model download)."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.types import FaceDetection
from utils.config import get_settings

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="Apple Vision is macOS-only")
s = get_settings()
FACE = cv2.imread(str(Path(__file__).parent / "data" / "face.jpg"))  # NASA, public domain


@pytest.fixture(scope="module")
def detector():
    from core.vision_detector import VisionDetector

    return VisionDetector(s)


def test_blank_frame(detector):
    assert detector.detect(np.zeros((720, 1280, 3), np.uint8)) == []


def test_face_with_keypoints_and_pose(detector):
    (det,) = detector.detect(FACE)
    x1, y1, x2, y2 = det.bbox
    le, re, nose, ml, mr = det.kps5
    assert det.landmarks.shape == (76, 2)
    assert all(x1 <= x <= x2 and y1 <= y <= y2 for x, y in det.kps5)
    assert le[0] < re[0] and ml[0] < mr[0]  # template order: image-left first
    assert max(le[1], re[1]) < nose[1] < min(ml[1], mr[1])
    assert abs(det.yaw) < 20 and abs(det.pitch) < 25


def test_small_far_face_found_by_tiles(detector):
    # ~45px face in a 1080p frame: below what Vision finds on the full frame alone.
    frame = cv2.resize(cv2.GaussianBlur(FACE, (0, 0), 25), (1920, 1080))
    small = cv2.resize(FACE, (220, 220), interpolation=cv2.INTER_AREA)
    frame[600:820, 300:520] = small
    faces = detector.detect(frame)
    assert len(faces) == 1 and 30 < faces[0].size < 70


def test_nms_drops_partial_box_inside_full_one():
    from core.vision_detector import nms

    full = FaceDetection(bbox=np.array([0, 0, 100, 100], np.float32), score=0.8)
    part = FaceDetection(bbox=np.array([0, 40, 60, 100], np.float32), score=0.9)
    other = FaceDetection(bbox=np.array([300, 0, 400, 100], np.float32), score=0.5)
    kept = [d for d, _ in nms([(full, 1), (part, 2), (other, 3)], 0.4)]
    assert kept == [part, other]


def test_tile_grid_stays_in_bounds():
    from core.vision_detector import tile_grid

    tiles = tile_grid(3, 0.25)
    assert len(tiles) == 9 and tile_grid(1, 0.25) == []
    assert all(0 <= x and 0 <= y and x + w <= 1 and y + h <= 1 for x, y, w, h in tiles)
    assert max(x + w for x, _, w, _ in tiles) == pytest.approx(1)


def test_multiple_faces_keep_their_own_keypoints(detector):
    # Vision returns landmarks in a shuffled order; each box must stay paired with its face.
    frame = np.full((1080, 1920, 3), 90, np.uint8)
    small = cv2.resize(FACE, (400, 400))
    for x in (40, 760, 1480):
        frame[300:700, x : x + 400] = small
    for _ in range(5):
        faces = detector.detect(frame)
        assert len(faces) == 3
        for det in faces:
            x1, y1, x2, y2 = det.bbox
            assert all(x1 <= x <= x2 and y1 <= y <= y2 for x, y in det.kps5)

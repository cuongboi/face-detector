import numpy as np

from core.enrollment import EnrollmentSession, classify_pose
from core.types import FaceDetection, FaceResult


def _face(yaw, pitch, emb):
    rng = np.random.default_rng(0)
    aligned = rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)  # noisy => sharp
    det = FaceDetection(bbox=np.array([0, 0, 120, 120], float), score=0.9, yaw=yaw, pitch=pitch)
    return FaceResult(det, aligned, emb / np.linalg.norm(emb))


def test_classify_pose(settings):
    assert classify_pose(0, 0, settings) == "front"
    assert classify_pose(30, 0, settings) == "left"
    assert classify_pose(-30, 0, settings) == "right"
    assert classify_pose(0, 25, settings) == "up"
    assert classify_pose(0, -25, settings) == "down"
    assert classify_pose(15, 0, settings) is None  # ambiguous band
    assert classify_pose(70, 0, settings) is None  # full profile rejected


def test_full_session_requires_front_first_and_hold(settings):
    emb = np.ones(512)
    s = EnrollmentSession(settings)
    assert s.offer(_face(30, 0, emb), 0.0).captured is None  # front must come first
    assert s.offer(_face(0, 0, emb), 1.0).captured is None  # hold starts
    assert s.offer(_face(0, 0, emb), 1.6).captured == "front"
    for pose, (y, p) in {"left": (30, 0), "right": (-30, 0), "up": (0, 25)}.items():
        assert s.offer(_face(y, p, emb), 0.0, manual=True).captured == pose
    assert not s.complete
    assert s.offer(_face(0, -25, emb), 0.0, manual=True).captured == "down"
    assert s.complete


def test_rejects_different_person(settings):
    s = EnrollmentSession(settings)
    s.offer(_face(0, 0, np.ones(512)), 0.0, manual=True)
    other = np.ones(512)
    other[:256] = -1  # orthogonal => cosine 0
    res = s.offer(_face(30, 0, other), 0.0, manual=True)
    assert res.captured is None and "left" not in s.captures

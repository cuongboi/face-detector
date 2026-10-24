"""Core ML embedder; macOS only, needs the converted model (scripts/download_models.py)."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.aligner import align_5pt
from utils.config import get_settings

s = get_settings()
FACE = cv2.imread(str(Path(__file__).parent / "data" / "face.jpg"))  # NASA, public domain
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin"
    or not all(s.model_path(m).exists() for m in (s.embedder_model, s.detector_model)),
    reason="Core ML is macOS-only; run scripts/download_models.py",
)


@pytest.fixture(scope="module")
def aligned():
    from core.scrfd_detector import ScrfdDetector

    (det,) = ScrfdDetector(s.model_copy(update={"pose_vision": False})).detect(FACE)
    return align_5pt(FACE, det.kps5)


@pytest.fixture(scope="module", params=[False, True], ids=["plain", "flip"])
def embedder(request):
    from core.embedder import Embedder

    return Embedder(s.model_copy(update={"embed_flip": request.param}))


def test_normalized_and_batch_matches_single(embedder, aligned):
    other = cv2.GaussianBlur(aligned, (0, 0), 2)
    batch = embedder.embed_batch([aligned, other])
    assert batch.shape == (2, 512)
    np.testing.assert_allclose(np.linalg.norm(batch, axis=1), 1, atol=1e-5)
    np.testing.assert_allclose(batch[0], embedder.embed(aligned), atol=1e-4)


def test_flip_tta_is_mirror_invariant(embedder, aligned):
    a, b = embedder.embed_batch([aligned, cv2.flip(aligned, 1)])
    if embedder.flip:
        assert float(a @ b) > 0.999  # sum of face + mirror is the same both ways
    else:
        assert float(a @ b) > 0.8  # same person, but not identical


def test_same_face_downscaled_still_matches(embedder, aligned):
    small = cv2.resize(cv2.resize(aligned, (28, 28), interpolation=cv2.INTER_AREA), (112, 112))
    a, b = embedder.embed_batch([aligned, small])
    assert float(a @ b) > s.match_threshold

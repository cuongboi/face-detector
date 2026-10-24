"""Smoke test on real models; skipped until scripts/download_models.py has run."""

import numpy as np
import pytest

from utils.config import get_settings

s = get_settings()
pytestmark = pytest.mark.skipif(
    not all(s.model_path(n).exists() for n in (s.yolo_model, s.landmarker_model)),
    reason="models not downloaded",
)


def test_detector_runs_on_blank_frame():
    from core.hybrid_detector import HybridDetector

    assert HybridDetector(s).detect(np.zeros((720, 1280, 3), np.uint8)) == []

"""Live preview: transparent overlays for webcams, recognition decoupled from RTSP display."""

import threading
import time

import numpy as np

import app
from utils.face_utils import draw_badge


def test_overlay_is_transparent_except_boxes():
    frame = np.zeros((2160, 3840, 3), np.uint8)
    canvas = app._overlay(frame)
    assert canvas.shape == (720, 1280, 4) and not canvas.any()  # preview size, fully clear
    app._render(canvas, [(np.array([300, 300, 900, 900], float), "Alice", (0, 255, 0))], 3840)
    alpha = canvas[..., 3]
    assert alpha[100, 100] == 255 and alpha[200, 200] == 0  # box edge at 300/3 = 100px
    assert alpha[500:, 400:].max() == 0


def test_overlay_mirrored_boxes_follow_mirrored_video():
    canvas = np.zeros((100, 200, 4), np.uint8)
    app._render(canvas, [(np.array([0, 10, 20, 30], float), "", (0, 255, 0))], 200, mirror=True)
    cols = np.flatnonzero(canvas[..., 3].any(axis=0))
    assert cols.min() >= 178 and cols.max() <= 199  # x 0-20 mirrored to 180-200


def test_badge_on_overlay_is_opaque():
    canvas = np.zeros((720, 1280, 4), np.uint8)
    draw_badge(canvas, "✓ Vào: Đặng Thị Ánh", (60, 170, 40), app.settings.badge_font)
    assert canvas[..., 3].max() == 255 and canvas[..., 3][700:, :100].max() == 0


class _FastCamera:
    """Produces a new frame every ms, like a camera that outpaces recognition."""

    def __init__(self):
        self.closed, self.seq = False, 0
        self.frame = np.zeros((10, 10, 3), np.uint8)

    def read(self, after_seq, timeout=1.0):
        time.sleep(0.001)
        self.seq += 1
        return self.frame, self.seq


def test_recognition_worker_never_blocks_display():
    calls, release = [], threading.Event()

    def slow_attend(frame):
        calls.append(1)
        release.wait(0.2)  # recognition far slower than the camera
        return [(np.array([1, 1, 5, 5], float), "x", (0, 0, 255))], {"__type__": "update"}

    cam = _FastCamera()
    worker = app.RecognitionWorker(cam, slow_attend)
    t0 = time.monotonic()
    shown = [worker.latest() for _ in range(50)]  # display side: returns immediately
    assert time.monotonic() - t0 < 0.1 and shown[0][0] == []
    release.set()
    deadline = time.monotonic() + 2
    while not worker.latest()[0] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert worker.latest()[0][0][1] == "x"
    worker.stop()
    cam.closed = True
    assert len(calls) < 50  # only the newest frames were recognized, not every one


def test_recognition_worker_hands_out_table_once():
    cam = _FastCamera()
    table = object()
    worker = app.RecognitionWorker(cam, lambda f: ([], table))
    deadline = time.monotonic() + 2
    recent = None
    while recent is not table and time.monotonic() < deadline:
        _, recent = worker.latest()
    worker.stop()
    cam.closed = True
    assert recent is table

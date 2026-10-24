import time

import cv2
import numpy as np

from core.rtsp import RtspCamera, redact


def test_redact_hides_password_only():
    assert redact("rtsp://admin:s3cret@10.0.0.5:554/live") == "rtsp://admin:***@10.0.0.5:554/live"
    assert redact("rtsp://10.0.0.5/live") == "rtsp://10.0.0.5/live"


def test_reads_newest_frames_and_stops(tmp_path):
    # FFmpeg opens a local file through the same code path as an RTSP URL.
    path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (160, 120))
    for i in range(50):
        writer.write(np.full((120, 160, 3), i * 5, np.uint8))
    writer.release()

    cam = RtspCamera(path, timeout_seconds=2, reconnect_seconds=0.1)
    try:
        frame, seq = cam.read(0, timeout=5)
        assert frame is not None and frame.shape == (120, 160, 3) and seq > 0
        nxt, seq2 = cam.read(seq, timeout=5)
        assert nxt is not None and seq2 > seq
    finally:
        cam.close()
    assert cam.closed
    assert cam.read(seq2, timeout=0.1)[0] is None  # no waiting once closed


def test_unreachable_camera_reports_error():
    cam = RtspCamera("rtsp://127.0.0.1:1/none", timeout_seconds=1, reconnect_seconds=0.1)
    try:
        deadline = time.monotonic() + 5
        while cam.error is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert cam.error and not cam.connected
        assert cam.read(0, timeout=0.2) == (None, 0)
    finally:
        cam.close()

"""Server-side RTSP camera (IP cameras can't be opened by the browser like a webcam)."""

import os
import threading
from urllib.parse import urlsplit, urlunsplit

# TCP avoids the smeared / grey frames UDP gives on lossy Wi-Fi. Must be set before FFmpeg opens.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
# FFmpeg prints a decode error per frame until the first keyframe after (re)connecting; drops
# and reconnects are logged by this module instead. Set to 32 (info) to debug a camera.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)

RTSP_SCHEMES = ("rtsp", "rtsps")


def redact(url: str) -> str:
    """Hide the password in rtsp://user:pass@host/... for logs and the UI."""
    parts = urlsplit(url)
    if parts.password is None:
        return url
    netloc = f"{parts.username}:***@{parts.hostname}" + (f":{parts.port}" if parts.port else "")
    return urlunsplit(parts._replace(netloc=netloc))


class RtspCamera:
    """Decodes the stream on a background thread and keeps only the newest frame.

    Reading in the consumer's loop would let FFmpeg buffer frames while the (slower) face
    pipeline runs, so the preview would fall further and further behind the live scene.
    Reconnects on its own after drops.
    """

    def __init__(self, url: str, timeout_seconds: float = 5.0, reconnect_seconds: float = 2.0):
        self.url = url
        self.timeout_ms = int(timeout_seconds * 1000)
        self.reconnect_seconds = reconnect_seconds
        self.connected = False
        self.error: str | None = None
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rtsp", daemon=True)
        self._thread.start()

    def _open(self) -> cv2.VideoCapture:
        timeouts = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.timeout_ms,
        ]
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, timeouts)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self) -> None:
        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                cap.release()
                self.error = "Không kết nối được camera, đang thử lại…"
                log.warning("RTSP open failed: %s", redact(self.url))
                self._stop.wait(self.reconnect_seconds)
                continue
            log.info("RTSP connected: %s", redact(self.url))
            self.connected, self.error = True, None
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                with self._cond:
                    self._frame, self._seq = frame, self._seq + 1
                    self._cond.notify_all()
            cap.release()
            self.connected = False
            if not self._stop.is_set():
                self.error = "Mất kết nối camera, đang kết nối lại…"
                log.warning("RTSP stream dropped: %s", redact(self.url))
                self._stop.wait(self.reconnect_seconds)
        with self._cond:
            self._cond.notify_all()

    def read(self, after_seq: int, timeout: float = 1.0) -> tuple[np.ndarray | None, int]:
        """Newest frame once it is newer than `after_seq` (else None after `timeout`)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != after_seq or self._stop.is_set(), timeout)
            if self._seq == after_seq:
                return None, after_seq
            return self._frame, self._seq

    def close(self) -> None:
        # Don't join: a blocked read can take up to the read timeout; the thread is a daemon.
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        return self._stop.is_set()

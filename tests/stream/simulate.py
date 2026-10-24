"""Simulate an IP camera: loop an mp4 file as a live RTSP stream.

Runs mediamtx (RTSP server) + ffmpeg in one Docker container, so nothing besides Docker
needs to be installed locally.

    uv run python tests/stream/simulate.py tests/stream/camera.mp4
    uv run python tests/stream/simulate.py clip.mp4 --port 8555 --path door --user u --password p
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402

log = logging.getLogger("rtsp-sim")

DEFAULT_IMAGE = "bluenviron/mediamtx:1.9.3-ffmpeg"
CONTAINER_PORT = 8554


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check)


def _lan_ip() -> str | None:
    # No packet is sent: connect() on UDP only picks the outbound interface.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return None


def _probe(video: Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise SystemExit(f"Cannot decode video: {video}")
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS) or 25.0,
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def _ffmpeg_cmd(filename: str, path: str, fps: float, transcode: bool) -> str:
    codec = (
        f"-c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p -g {round(fps)} -an"
        if transcode
        else "-c copy"
    )
    # -re: real-time pacing like a live camera; -stream_loop -1: loop forever.
    return (
        f"ffmpeg -hide_banner -loglevel error -re -stream_loop -1 -i /media/{json.dumps(filename)} "
        f"{codec} -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:{CONTAINER_PORT}/{path}"
    )


def _mediamtx_config(run_on_init: str, path: str, user: str | None, password: str | None) -> str:
    # JSON is valid YAML, so no YAML dependency is needed.
    if user:
        users = [
            {"user": "any", "pass": "", "ips": ["127.0.0.1/32", "::1/128"],
             "permissions": [{"action": "publish"}]},
            {"user": user, "pass": password or "", "ips": [],
             "permissions": [{"action": "read"}, {"action": "playback"}]},
        ]
    else:
        users = [{"user": "any", "pass": "", "ips": [],
                  "permissions": [{"action": "publish"}, {"action": "read"}]}]
    config = {
        "logLevel": "warn",
        "rtsp": True,
        "protocols": ["tcp"],
        "rtspAddress": f":{CONTAINER_PORT}",
        "rtmp": False,
        "hls": False,
        "webrtc": False,
        "srt": False,
        "api": False,
        "authMethod": "internal",
        "authInternalUsers": users,
        "paths": {path: {"runOnInit": run_on_init, "runOnInitRestart": True}},
    }
    return json.dumps(config, indent=2)


def _wait_ready(url: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        try:
            if cap.isOpened() and cap.read()[0]:
                return True
        finally:
            cap.release()
        time.sleep(0.5)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path, help="path to an mp4 (or any ffmpeg-readable) file")
    parser.add_argument("--port", type=int, default=8554, help="host RTSP port (default 8554)")
    parser.add_argument("--path", default="cam", help="stream path, rtsp://host:port/<path>")
    parser.add_argument("--host", default="0.0.0.0", help="host interface to bind")
    parser.add_argument("--user", help="require this username to read the stream")
    parser.add_argument("--password", help="password for --user")
    parser.add_argument("--transcode", action="store_true",
                        help="re-encode to H.264 (use if the file isn't H.264 or clients lag)")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="mediamtx docker image (-ffmpeg)")
    parser.add_argument("--timeout", type=float, default=20.0, help="readiness timeout seconds")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    video = args.video.expanduser().resolve()
    if not video.is_file():
        parser.error(f"file not found: {video}")
    if args.password and not args.user:
        parser.error("--password requires --user")
    path = args.path.strip("/")
    if not path or not path.replace("_", "").replace("-", "").replace("/", "").isalnum():
        parser.error("--path may only contain letters, digits, '-', '_' and '/'")
    if shutil.which("docker") is None:
        raise SystemExit("docker not found on PATH")
    if _docker("info", check=False).returncode != 0:
        raise SystemExit("Docker daemon is not running (start Docker Desktop)")

    width, height, fps, frames = _probe(video)
    log.info("Source %s: %dx%d @ %.2f fps, %d frames (%.1fs)",
             video.name, width, height, fps, frames, frames / fps if fps else 0)

    workdir = Path(tempfile.mkdtemp(prefix="rtsp-sim-"))
    config_file = workdir / "mediamtx.yml"
    config_file.write_text(_mediamtx_config(
        _ffmpeg_cmd(video.name, path, fps, args.transcode), path, args.user, args.password))

    name = f"of-rtsp-sim-{args.port}"
    _docker("rm", "-f", name, check=False)  # stale container from a crashed run
    run = _docker(
        "run", "-d", "--name", name,
        "-p", f"{args.host}:{args.port}:{CONTAINER_PORT}/tcp",
        "-v", f"{video.parent}:/media:ro",
        "-v", f"{config_file}:/mediamtx.yml:ro",
        args.image,
        check=False,
    )
    if run.returncode != 0:
        shutil.rmtree(workdir, ignore_errors=True)
        raise SystemExit(f"docker run failed:\n{run.stderr.strip()}")

    stopping = False

    def stop(*_: object) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        log.info("Stopping %s ...", name)
        _docker("rm", "-f", name, check=False)
        shutil.rmtree(workdir, ignore_errors=True)

    def on_signal(*_: object) -> None:
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_signal)
    try:
        creds = ""
        if args.user:
            creds = f"{quote(args.user, safe='')}:{quote(args.password or '', safe='')}@"
        local_url = f"rtsp://{creds}127.0.0.1:{args.port}/{path}"
        log.info("Waiting for stream (image pull may take a while on first run) ...")
        if not _wait_ready(local_url, args.timeout):
            logs = _docker("logs", name, check=False)
            log.error("Stream not ready after %.0fs. Container logs:\n%s%s",
                      args.timeout, logs.stdout, logs.stderr)
            if not args.transcode:
                log.error("Hint: retry with --transcode if the file isn't H.264/AAC.")
            return 1

        print("\nRTSP stream ready (looping, Ctrl+C to stop):")
        print(f"  local : {local_url}")
        lan = _lan_ip()
        if lan and args.host in ("0.0.0.0", lan):
            print(f"  LAN   : rtsp://{creds}{lan}:{args.port}/{path}")
        print(flush=True)

        while True:
            time.sleep(2)
            state = _docker("inspect", "-f", "{{.State.Running}}", name, check=False)
            if state.stdout.strip() != "true":
                log.error("Container exited unexpectedly")
                return 1
    except KeyboardInterrupt:
        return 0
    finally:
        stop()


if __name__ == "__main__":
    sys.exit(main())

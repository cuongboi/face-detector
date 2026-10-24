"""Download all models into data/models. Re-running skips files that already exist."""

import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import get_settings  # noqa: E402

YOLO_FACE_URL = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.pt"
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"


def _download(url: str, dest: Path) -> None:
    print(f"↓ {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.rename(dest)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    s = get_settings()
    s.models_dir.mkdir(parents=True, exist_ok=True)

    for url, name in ((YOLO_FACE_URL, s.yolo_model), (LANDMARKER_URL, s.landmarker_model)):
        dest = s.model_path(name)
        if not dest.exists():
            _download(url, dest)

    embedder = s.model_path(s.embedder_model)
    if not embedder.exists():
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "buffalo_l.zip"
            _download(BUFFALO_L_URL, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                member = next(m for m in zf.namelist() if m.endswith("w600k_r50.onnx"))
                with zf.open(member) as src, open(embedder, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    for name in (s.yolo_model, s.landmarker_model, s.embedder_model):
        path = s.model_path(name)
        print(f"✓ {name}  {path.stat().st_size / 1e6:.1f} MB  sha256={_sha256(path)[:16]}…")


if __name__ == "__main__":
    main()

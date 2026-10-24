"""Download all source models into data/models, then convert them to Core ML.

Re-running skips files that already exist. Needs the `convert` group:
    uv run --group convert python scripts/download_models.py
"""

import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.convert_models import ARCFACE_ONNX, NAFNET_GOPRO_PTH, NAFNET_SIDD_PTH  # noqa: E402
from scripts.convert_models import main as convert_all  # noqa: E402
from utils.config import get_settings  # noqa: E402

BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
# Official NAFNet width-32 checkpoints (megvii-research/NAFNet README), hosted on Google Drive.
NAFNET_SIDD_GDRIVE_ID = "1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ"
NAFNET_GOPRO_GDRIVE_ID = "1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj"


def _download(url: str, dest: Path) -> None:
    print(f"↓ {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.rename(dest)


def _download_gdrive(file_id: str, dest: Path) -> None:
    # Drive puts a virus-scan confirmation page in front of large files; gdown handles it.
    import gdown

    tmp = dest.with_suffix(dest.suffix + ".part")
    if gdown.download(id=file_id, output=str(tmp), quiet=False) is None:
        raise SystemExit(f"Download failed: {dest.name} (Google Drive id {file_id})")
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

    embedder = s.model_path(ARCFACE_ONNX)
    if not embedder.exists():
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "buffalo_l.zip"
            _download(BUFFALO_L_URL, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                member = next(m for m in zf.namelist() if m.endswith("w600k_r50.onnx"))
                with zf.open(member) as src, open(embedder, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    for file_id, name in (
        (NAFNET_SIDD_GDRIVE_ID, NAFNET_SIDD_PTH), (NAFNET_GOPRO_GDRIVE_ID, NAFNET_GOPRO_PTH)
    ):
        dest = s.model_path(name)
        if not dest.exists():
            _download_gdrive(file_id, dest)

    for name in (ARCFACE_ONNX, NAFNET_SIDD_PTH, NAFNET_GOPRO_PTH):
        path = s.model_path(name)
        print(f"✓ {name}  {path.stat().st_size / 1e6:.1f} MB  sha256={_sha256(path)[:16]}…")

    convert_all()


if __name__ == "__main__":
    main()

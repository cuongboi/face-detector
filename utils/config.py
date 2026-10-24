"""Central configuration. Every tunable lives here; override via env vars (FA_*) or .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FA_", env_file=ROOT / ".env", extra="ignore")

    # Paths
    data_dir: Path = ROOT / "data"
    models_dir: Path = ROOT / "data" / "models"
    db_path: Path = ROOT / "data" / "attendance.db"
    # Compiled Core ML models produced by scripts/convert_models.py
    embedder_model: str = "w600k_r50.mlmodelc"

    # Server. Phones only allow camera access over HTTPS, so serve TLS when a cert exists
    # (create one with scripts/make_cert.py).
    server_host: str = "0.0.0.0"
    server_port: int = 7860
    ssl_certfile: Path = ROOT / "data" / "certs" / "cert.pem"
    ssl_keyfile: Path = ROOT / "data" / "certs" / "key.pem"

    # Detection (Apple Vision, tuned for 5-10m). Vision only finds faces wider than ~5% of its
    # input, so the frame is also scanned as an NxN grid of overlapping tiles (3 -> ~40px faces
    # at 1080p, ~57ms on M1). 1 = full frame only (~10ms, faces >= ~100px at 1080p).
    vision_tile_grid: int = 3
    vision_tile_overlap: float = 0.25  # fraction of a tile shared with its neighbour
    vision_conf: float = 0.3
    vision_nms_iou: float = 0.4  # same face found in several tiles
    expand_ratio: float = 0.3
    min_face_size: int = 40

    # Restoration (NAFNet) of dark / blurry face crops before alignment
    enhance_enabled: bool = True
    denoise_model: str = "NAFNet-SIDD-width32.mlmodelc"
    deblur_model: str = "NAFNet-GoPro-width32.mlmodelc"
    enhance_dark_luma: float = 70.0  # mean L (0-255) below this => denoise + CLAHE
    enhance_blur_sharpness: float = 60.0  # Laplacian variance below this => deblur
    enhance_max_side: int = 192  # = the Core ML input size; bigger crops are downscaled first

    # Recognition
    qdrant_url: str = "http://localhost:6333"
    collection: str = "faces"
    embedding_dim: int = 512
    match_threshold: float = 0.5
    search_top_k: int = 10

    # Attendance
    # First-in / last-out: a better match within `in_window_seconds` of the check-in replaces
    # its photo (time kept); later matches write/overwrite the check-out.
    in_window_seconds: float = 60
    sighting_interval_seconds: float = 2  # min gap between writes while the check-in can improve
    cooldown_seconds: int = 60  # min gap between check-out overwrites
    vote_frames: int = 3
    track_iou: float = 0.3
    track_ttl_seconds: float = 2.0
    camera_id: str = "cam-0"
    # Raw frame saved (JPEG, base64) with each attendance record as proof.
    snapshot_width: int = 960
    snapshot_quality: int = 85
    # External IP camera for the attendance tab, e.g. rtsp://user:pass@192.168.1.10:554/stream1
    rtsp_url: str = ""
    rtsp_timeout_seconds: float = 5.0
    rtsp_reconnect_seconds: float = 2.0
    # Browser camera resolution asked for ("ideal": the browser falls back to the best the camera
    # supports, e.g. 4K on recent phones). Higher = farther faces, but more upload per frame.
    webcam_width: int = 3840
    webcam_height: int = 2160
    display_max_width: int = 1280  # previews are downscaled; detection uses full resolution
    # Check-in badge on the preview; TTF font so Vietnamese diacritics render.
    badge_font: Path = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    badge_seconds: float = 3.0

    # Enrollment pose gates (degrees)
    front_max_angle: float = 10.0
    side_yaw_min: float = 20.0
    tilt_pitch_min: float = 15.0
    max_angle: float = 45.0  # |pitch| limit
    full_yaw_min: float = 50.0  # |yaw| from here = full profile ("full_left"/"full_right")
    full_yaw_max: float = 85.0
    min_sharpness: float = 40.0
    same_person_min_sim: float = 0.3
    hold_seconds: float = 0.5

    def model_path(self, name: str) -> Path:
        return self.models_dir / name


@lru_cache
def get_settings() -> Settings:
    return Settings()

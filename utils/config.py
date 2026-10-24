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
    yolo_model: str = "yolov11n-face.pt"
    landmarker_model: str = "face_landmarker.task"
    embedder_model: str = "w600k_r50.onnx"

    # Detection (tuned for 5-10m)
    yolo_imgsz: int = 960
    yolo_device: str = "mps"  # Apple Silicon GPU; falls back to cpu if unavailable
    yolo_conf: float = 0.4
    expand_ratio: float = 0.3
    min_face_size: int = 40
    # Below this crop size MediaPipe is unreliable -> use YOLO keypoints directly
    mediapipe_min_crop: int = 96
    use_clahe: bool = True

    # Recognition
    qdrant_url: str = "http://localhost:6333"
    collection: str = "faces"
    embedding_dim: int = 512
    match_threshold: float = 0.5
    search_top_k: int = 10

    # Attendance
    cooldown_seconds: int = 60
    vote_frames: int = 3
    track_iou: float = 0.3
    track_ttl_seconds: float = 2.0
    camera_id: str = "cam-0"

    # Enrollment pose gates (degrees)
    front_max_angle: float = 10.0
    side_yaw_min: float = 20.0
    tilt_pitch_min: float = 15.0
    max_angle: float = 45.0
    min_sharpness: float = 40.0
    same_person_min_sim: float = 0.3
    hold_seconds: float = 0.5

    def model_path(self, name: str) -> Path:
        return self.models_dir / name


@lru_cache
def get_settings() -> Settings:
    return Settings()

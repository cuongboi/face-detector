import pytest

from utils.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, qdrant_url=":memory:", cooldown_seconds=60, vote_frames=3)

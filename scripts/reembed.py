"""Rebuild the active embedder's Qdrant collection from the enrollment crops in SQLite.

Run after switching FA_EMBEDDER_MODEL (or FA_EMBED_FLIP), so nobody has to enroll again:
    uv run python scripts/reembed.py
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.embedder import Embedder  # noqa: E402
from core.repository import Repository  # noqa: E402
from core.vector_store import VectorStore  # noqa: E402
from utils.config import get_settings  # noqa: E402


def reembed(repo: Repository, store: VectorStore, embedder: Embedder) -> int:
    """Returns the number of people indexed."""
    people: dict[tuple[int, str, str], dict] = defaultdict(dict)
    for pid, name, emp, pose, img in repo.person_faces():
        people[(pid, name, emp)][pose] = img
    store.recreate()
    for (pid, name, emp), crops in people.items():
        embs = embedder.embed_batch(list(crops.values()))
        store.add_faces(pid, name, emp, dict(zip(crops, embs, strict=True)))
    return len(people)


def main() -> None:
    s = get_settings()
    repo = Repository(s.db_path)
    store = VectorStore(s.qdrant_url, s.collection, s.embedding_dim)
    n = reembed(repo, store, Embedder(s))
    missing = len(repo.list_persons()) - n
    print(f"✓ {n} người -> {s.collection}")
    if missing:
        print(f"! {missing} người không có ảnh đăng ký lưu sẵn (đăng ký cũ): cần đăng ký lại")


if __name__ == "__main__":
    main()

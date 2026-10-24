from datetime import datetime

import numpy as np

from core.repository import Repository
from core.vector_store import VectorStore


def _unit(seed):
    v = np.random.default_rng(seed).normal(size=512)
    return v / np.linalg.norm(v)


def test_search_groups_poses_and_delete(settings):
    store = VectorStore(":memory:", "faces_test")
    a = {"front": _unit(1), "left": _unit(2)}
    store.add_faces(1, "Alice", "E1", a)
    store.add_faces(2, "Bob", "E2", {"front": _unit(3)})
    assert store.count() == 3

    m = store.search(a["left"])
    assert m.person_id == 1 and m.pose == "left" and m.score > 0.99

    store.delete_person(1)
    assert store.count() == 1
    assert store.search(a["left"]).person_id == 2


def test_repository_roundtrip():
    repo = Repository(":memory:")
    pid = repo.add_person("Alice", "E1")
    repo.record_sighting(pid, "Alice", "E1", 0.8, "cam-0", datetime.now(), lambda: None, 60)
    assert len(repo.history()) == 1 and len(repo.daily_summary()) == 1
    repo.delete_person(pid)
    assert repo.list_persons().empty and repo.history().empty  # cascade
    assert repo.daily_summary().empty


def test_search_batch_matches_each_face_in_order():
    store = VectorStore(":memory:", "faces_batch")
    assert store.search_batch(np.stack([_unit(1)])) == [None]  # empty collection
    store.add_faces(1, "Alice", "E1", {"front": _unit(1)})
    store.add_faces(2, "Bob", "E2", {"front": _unit(2)})
    store.add_faces(3, "Carol", "E3", {"front": _unit(3)})
    found = store.search_batch(np.stack([_unit(3), _unit(1), _unit(2)]))
    assert [m.person_id for m in found] == [3, 1, 2]
    assert store.search_batch(np.zeros((0, 512))) == []


def test_badges_stack_downwards():
    from pathlib import Path

    from utils.face_utils import draw_badge

    img = np.zeros((720, 1280, 3), np.uint8)
    font = Path("/nonexistent.ttf")  # cv2 fallback
    y1 = draw_badge(img, "A", (0, 255, 0), font)
    y2 = draw_badge(img, "B", (0, 255, 0), font, y1)
    assert 0 < y1 < y2 < 720

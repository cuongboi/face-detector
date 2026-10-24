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
    repo.log_attendance(pid, "Alice", "E1", 0.8, "cam-0")
    assert len(repo.history()) == 1
    repo.delete_person(pid)
    assert repo.list_persons().empty and repo.history().empty  # cascade

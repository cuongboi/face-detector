import numpy as np

from core.attendance import AttendanceService, FaceState
from core.vector_store import Match


class FakeRepo:
    def __init__(self):
        self.logs = []

    def log_attendance(self, person_id, name, employee_id, score, camera_id, when=None):
        self.logs.append((person_id, score))


BOX = np.array([100, 100, 200, 200], float)
ALICE = Match(1, "Alice", "E1", 0.7)
BOB = Match(2, "Bob", "E2", 0.7)


def test_logs_after_consecutive_votes_then_cooldown(settings):
    repo = FakeRepo()
    svc = AttendanceService(settings, repo)
    states = [svc.update([(BOX, ALICE)], t)[0].state for t in (0.0, 0.1, 0.2, 0.3)]
    assert states == [FaceState.VERIFYING, FaceState.VERIFYING, FaceState.CHECKED_IN,
                      FaceState.COOLDOWN]
    assert len(repo.logs) == 1
    # After cooldown, a fresh stable track logs again.
    for t in (100.0, 100.1, 100.2):
        svc.update([(BOX, ALICE)], t)
    assert len(repo.logs) == 2


def test_flicker_and_low_score_do_not_log(settings):
    repo = FakeRepo()
    svc = AttendanceService(settings, repo)
    for t, m in enumerate([ALICE, BOB, ALICE, BOB, Match(1, "Alice", "E1", 0.2)]):
        svc.update([(BOX, m)], t * 0.1)
    assert repo.logs == []


def test_separate_tracks_for_separate_faces(settings):
    repo = FakeRepo()
    svc = AttendanceService(settings, repo)
    other = np.array([400, 100, 500, 200], float)
    for t in (0.0, 0.1, 0.2):
        svc.update([(BOX, ALICE), (other, BOB)], t)
    assert sorted(pid for pid, _ in repo.logs) == [1, 2]

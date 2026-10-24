import numpy as np
import pytest

from core.attendance import AttendanceService, FaceState
from core.vector_store import Match


class FakeRepo:
    """Records calls; answers like Repository.record_sighting with a scripted kind."""

    def __init__(self, kinds=None):
        self.calls = []
        self.images = []
        self.kinds = list(kinds or [])

    def record_sighting(self, person_id, name, employee_id, score, camera_id, when, image_fn,
                        in_window_seconds):
        self.calls.append((person_id, score))
        kind = self.kinds.pop(0) if self.kinds else "in"
        if kind:
            self.images.append(image_fn())
        return kind


BOX = np.array([100, 100, 200, 200], float)
ALICE = Match(1, "Alice", "E1", 0.7)
BOB = Match(2, "Bob", "E2", 0.7)


def test_votes_then_throttled_writes(settings):
    repo = FakeRepo(["in", None, "out"])
    svc = AttendanceService(settings, repo)

    def seen_until(t):  # a face streamed at 10 fps (tracks expire after track_ttl_seconds)
        return svc.update([(BOX, ALICE)], t)[0].state

    states = [seen_until(t) for t in (0.0, 0.1, 0.2, 0.3)]
    assert states == [FaceState.VERIFYING, FaceState.VERIFYING, FaceState.CHECKED_IN,
                      FaceState.COOLDOWN]
    assert len(repo.calls) == 1
    # Within the check-in window the repo is asked again every sighting_interval_seconds.
    for t in np.arange(0.4, 2.2, 0.1):
        seen_until(t)
    assert seen_until(2.25) == FaceState.COOLDOWN and len(repo.calls) == 2  # repo: not better
    for t in np.arange(2.35, 4.2, 0.1):
        seen_until(t)
    assert seen_until(4.3) == FaceState.CHECKED_OUT and len(repo.calls) == 3
    # After a check-out, the next write waits cooldown_seconds.
    end = 4.3 + settings.cooldown_seconds
    for t in np.arange(4.4, end - 0.1, 0.5):
        seen_until(t)
    assert len(repo.calls) == 3
    seen_until(end + 0.01)
    assert len(repo.calls) == 4


def test_flicker_and_low_score_do_not_write(settings):
    repo = FakeRepo()
    svc = AttendanceService(settings, repo)
    for t, m in enumerate([ALICE, BOB, ALICE, BOB, Match(1, "Alice", "E1", 0.2)]):
        svc.update([(BOX, m)], t * 0.1)
    assert repo.calls == []


def test_separate_tracks_for_separate_faces(settings):
    repo = FakeRepo()
    svc = AttendanceService(settings, repo)
    other = np.array([400, 100, 500, 200], float)
    for t in (0.0, 0.1, 0.2):
        svc.update([(BOX, ALICE), (other, BOB)], t)
    assert sorted(pid for pid, _ in repo.calls) == [1, 2]


def test_written_record_carries_960px_photo_encoded_once_per_frame(settings):
    from utils.face_utils import decode_snapshot

    repo = FakeRepo()
    svc = AttendanceService(settings, repo)
    other = np.array([400, 100, 500, 200], float)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    for t in (0.0, 0.1, 0.2):
        svc.update([(BOX, ALICE), (other, BOB)], t, frame=frame)
    assert len(repo.images) == 2 and repo.images[0] is repo.images[1]
    assert decode_snapshot(repo.images[0]).shape == (540, 960, 3)


def test_same_person_on_two_faces_only_best_one_counts(settings):
    repo = FakeRepo()
    svc = AttendanceService(settings, repo)
    other = np.array([400, 100, 500, 200], float)
    lookalike = Match(1, "Alice", "E1", 0.6)
    for t in (0.0, 0.1, 0.2):
        statuses = svc.update([(BOX, ALICE), (other, lookalike)], t)
    assert statuses[1].match is None and statuses[1].state == FaceState.UNKNOWN
    assert len(repo.calls) == 1 and repo.calls[0][1] == pytest.approx(0.7)


def test_three_people_in_one_frame_all_written(settings):
    repo = FakeRepo()
    svc = AttendanceService(settings, repo)
    carol = Match(3, "Carol", "E3", 0.8)
    boxes = [BOX, np.array([400, 100, 500, 200], float), np.array([700, 100, 800, 200], float)]
    for t in (0.0, 0.1, 0.2):
        statuses = svc.update(list(zip(boxes, [ALICE, BOB, carol], strict=True)), t)
    assert [s.state for s in statuses] == [FaceState.CHECKED_IN] * 3
    assert sorted(pid for pid, _ in repo.calls) == [1, 2, 3]

import sqlite3
from datetime import date, datetime

import numpy as np

from core.repository import Repository
from utils.face_utils import decode_snapshot, encode_snapshot

WINDOW = 60


def _repo() -> Repository:
    repo = Repository(":memory:")
    repo.add_person("Alice", "E1")
    return repo


def _sight(repo, hh, mm, ss, score, image="img", day=24):
    calls = []

    def image_fn():
        calls.append(1)
        return image

    kind = repo.record_sighting(
        1, "Alice", "E1", score, "cam", datetime(2026, 10, day, hh, mm, ss), image_fn, WINDOW
    )
    return kind, len(calls)


def _record(repo, day=24) -> dict:
    return repo.daily_summary(date(2026, 10, day)).to_dict("records")[0]


def test_first_in_better_in_then_last_out():
    repo = _repo()
    assert _sight(repo, 8, 0, 0, 0.6, "in1") == ("in", 1)
    # Better match inside the first minute: photo + score replaced, check-in time kept.
    assert _sight(repo, 8, 0, 30, 0.8, "in2") == ("in_update", 1)
    assert _sight(repo, 8, 0, 40, 0.7, "x") == (None, 0)  # not better: nothing, no encode
    rec = _record(repo)
    assert rec["check_in"] == "2026-10-24T08:00:00" and rec["in_score"] == 0.8
    assert rec["check_out"] == ""
    assert repo.snapshot("E1", "2026-10-24", "in") == "in2"

    # A minute after check-in: check-out, then every later match overwrites it.
    assert _sight(repo, 8, 1, 10, 0.5, "out1") == ("out", 1)
    assert _sight(repo, 12, 0, 0, 0.45, "out_lower") == ("out", 1)  # any score overwrites
    assert _sight(repo, 17, 0, 0, 0.55, "out2") == ("out", 1)
    rec = _record(repo)
    assert rec["check_in"] == "2026-10-24T08:00:00"
    assert rec["check_out"] == "2026-10-24T17:00:00" and rec["out_score"] == 0.55
    assert repo.snapshot("E1", "2026-10-24", "out") == "out2"
    assert repo.snapshot("E1", "2026-10-24", "in") == "in2"
    assert len(repo.history()) == 5  # event log: every write, not the skipped match

    # Next day starts a new record.
    assert _sight(repo, 8, 0, 0, 0.6, day=25) == ("in", 1)
    assert len(repo.daily_summary()) == 2


def test_clear_history_by_day():
    repo = _repo()
    _sight(repo, 8, 0, 0, 0.6)
    _sight(repo, 8, 0, 0, 0.6, day=25)
    assert repo.clear_history(date(2026, 10, 24)) == 1
    assert len(repo.daily_summary()) == 1 and len(repo.history()) == 1


def test_snapshot_encoding_960px_and_no_upscale():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[:, :960] = 255
    img = decode_snapshot(encode_snapshot(frame, 960))
    assert img.shape == (540, 960, 3) and img[:, :400].mean() > 250 and img[:, 560:].mean() < 5
    assert decode_snapshot(encode_snapshot(np.zeros((480, 640, 3), np.uint8), 960)).shape[1] == 640


def test_old_event_log_migrates_to_daily_records(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE persons (id INTEGER PRIMARY KEY, name TEXT, employee_id TEXT UNIQUE,"
        " created_at TEXT);"
        "INSERT INTO persons VALUES (1, 'Old', 'E9', '2026-01-01');"
        "CREATE TABLE attendance (id INTEGER PRIMARY KEY, person_id INTEGER, name TEXT,"
        " employee_id TEXT, timestamp TEXT, score REAL, camera_id TEXT);"
        "INSERT INTO attendance VALUES (1, 1, 'Old', 'E9', '2026-01-01T08:00:00', 0.6, 'cam');"
        "INSERT INTO attendance VALUES (2, 1, 'Old', 'E9', '2026-01-01T17:00:00', 0.7, 'cam');"
        "INSERT INTO attendance VALUES (3, 1, 'Old', 'E9', '2026-01-02T08:00:00', 0.6, 'cam');"
    )
    conn.commit()
    conn.close()
    recs = Repository(path).daily_summary().to_dict("records")
    assert [(r["day"], r["check_in"][11:], r["check_out"][11:]) for r in recs] == [
        ("2026-01-02", "08:00:00", ""),
        ("2026-01-01", "08:00:00", "17:00:00"),
    ]

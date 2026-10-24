"""SQLite: source of truth for people and attendance history.

`daily_attendance` holds one first-in / last-out record per person per day (with photos);
`attendance` is the event log of every write to it (no photos).
"""

import sqlite3
import threading
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    employee_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    employee_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    score REAL NOT NULL,
    camera_id TEXT NOT NULL,
    image TEXT  -- legacy (per-event photos); photos now live in daily_attendance
);
CREATE INDEX IF NOT EXISTS idx_attendance_ts ON attendance(timestamp);
CREATE TABLE IF NOT EXISTS daily_attendance (
    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    day TEXT NOT NULL,
    name TEXT NOT NULL,
    employee_id TEXT NOT NULL,
    check_in TEXT NOT NULL,
    in_score REAL NOT NULL,
    in_image TEXT,  -- base64 JPEG of the raw frame
    check_out TEXT,
    out_score REAL,
    out_image TEXT,
    camera_id TEXT NOT NULL,
    PRIMARY KEY (person_id, day)
);
"""

# One row per person-day from the legacy event log: first event = in, last (if later) = out.
MIGRATE_DAILY = """
INSERT INTO daily_attendance (person_id, day, name, employee_id, check_in, in_score, in_image,
                              check_out, out_score, out_image, camera_id)
SELECT d.person_id, d.day, f.name, f.employee_id, f.timestamp, f.score, f.image,
       CASE WHEN l.id != f.id THEN l.timestamp END, CASE WHEN l.id != f.id THEN l.score END,
       CASE WHEN l.id != f.id THEN l.image END, f.camera_id
FROM (SELECT person_id, date(timestamp) AS day, MIN(id) AS first_id, MAX(id) AS last_id
      FROM attendance GROUP BY person_id, day) d
JOIN attendance f ON f.id = d.first_id
JOIN attendance l ON l.id = d.last_id
"""


class Repository:
    def __init__(self, db_path: Path | str):
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(attendance)")}
        if "image" not in columns:  # databases created before snapshots existed
            self.conn.execute("ALTER TABLE attendance ADD COLUMN image TEXT")
        if not self.conn.execute("SELECT 1 FROM daily_attendance LIMIT 1").fetchone():
            self.conn.execute(MIGRATE_DAILY)
        self.conn.commit()
        self._lock = threading.Lock()

    def add_person(self, name: str, employee_id: str) -> int:
        with self._lock, self.conn:
            cur = self.conn.execute(
                "INSERT INTO persons (name, employee_id, created_at) VALUES (?, ?, ?)",
                (name, employee_id, datetime.now().isoformat(timespec="seconds")),
            )
            return int(cur.lastrowid)

    def get_person_by_employee_id(self, employee_id: str) -> tuple | None:
        with self._lock:
            return self.conn.execute(
                "SELECT id, name, employee_id FROM persons WHERE employee_id = ?", (employee_id,)
            ).fetchone()

    def delete_person(self, person_id: int) -> None:
        with self._lock, self.conn:
            self.conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))

    def list_persons(self) -> pd.DataFrame:
        with self._lock:
            return pd.read_sql_query(
                "SELECT id, name, employee_id, created_at FROM persons ORDER BY id", self.conn
            )

    def record_sighting(
        self, person_id: int, name: str, employee_id: str, score: float, camera_id: str,
        when: datetime, image_fn: Callable[[], str | None], in_window_seconds: float,
    ) -> str | None:
        """Apply one confirmed match to the person's record for that day.

        - first match of the day                     -> "in"
        - within `in_window_seconds` of the check-in,
          only if the score beats the stored one     -> "in_update" (check-in time kept)
        - after the window, any score                 -> "out" (time, score, photo overwritten)
        Returns None when nothing was written; `image_fn` is only called on a write.
        """
        ts = when.isoformat(timespec="seconds")
        day = when.date().isoformat()
        score = round(score, 4)
        with self._lock, self.conn:
            row = self.conn.execute(
                "SELECT check_in, in_score FROM daily_attendance WHERE person_id = ? AND day = ?",
                (person_id, day),
            ).fetchone()
            if row is None:
                kind = "in"
                self.conn.execute(
                    "INSERT INTO daily_attendance (person_id, day, name, employee_id, check_in,"
                    " in_score, in_image, camera_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (person_id, day, name, employee_id, ts, score, image_fn(), camera_id),
                )
            elif (when - datetime.fromisoformat(row[0])).total_seconds() < in_window_seconds:
                if score <= row[1]:
                    return None
                kind = "in_update"
                self.conn.execute(
                    "UPDATE daily_attendance SET name = ?, employee_id = ?, in_score = ?,"
                    " in_image = ?, camera_id = ? WHERE person_id = ? AND day = ?",
                    (name, employee_id, score, image_fn(), camera_id, person_id, day),
                )
            else:
                kind = "out"
                self.conn.execute(
                    "UPDATE daily_attendance SET name = ?, employee_id = ?, check_out = ?,"
                    " out_score = ?, out_image = ? WHERE person_id = ? AND day = ?",
                    (name, employee_id, ts, score, image_fn(), person_id, day),
                )
            self.conn.execute(
                "INSERT INTO attendance (person_id, name, employee_id, timestamp, score, camera_id)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (person_id, name, employee_id, ts, score, camera_id),
            )
            return kind

    def history(self, day: date | None = None, limit: int | None = None) -> pd.DataFrame:
        query = "SELECT name, employee_id, timestamp, score, camera_id FROM attendance"
        params: list = []
        if day is not None:
            query += " WHERE date(timestamp) = ?"
            params.append(day.isoformat())
        query += " ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {int(limit)}"
        with self._lock:
            return pd.read_sql_query(query, self.conn, params=params)

    def daily_summary(self, day: date | None = None) -> pd.DataFrame:
        """First-in / last-out records, one row per person per day (no photos)."""
        query = (
            "SELECT day, name, employee_id, check_in, COALESCE(check_out, '') AS check_out,"
            " in_score, out_score FROM daily_attendance"
        )
        params: list = []
        if day is not None:
            query += " WHERE day = ?"
            params.append(day.isoformat())
        query += " ORDER BY day DESC, check_in"
        with self._lock:
            return pd.read_sql_query(query, self.conn, params=params)

    def snapshot(self, employee_id: str, day: str, kind: str) -> str | None:
        """Base64 photo of a record's check-in (`kind="in"`) or check-out (`"out"`)."""
        column = {"in": "in_image", "out": "out_image"}[kind]
        with self._lock:
            row = self.conn.execute(
                f"SELECT {column} FROM daily_attendance WHERE employee_id = ? AND day = ?",
                (employee_id, day),
            ).fetchone()
        return row[0] if row else None

    def clear_history(self, day: date | None = None) -> int:
        """Delete records (and their event log) for `day`, or everything. Returns records."""
        where, params = "", []
        if day is not None:
            where, params = " WHERE day = ?", [day.isoformat()]
        with self._lock, self.conn:
            log_where = where.replace("day", "date(timestamp)")
            self.conn.execute("DELETE FROM attendance" + log_where, params)
            return self.conn.execute("DELETE FROM daily_attendance" + where, params).rowcount

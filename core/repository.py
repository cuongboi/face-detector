"""SQLite: source of truth for people and attendance history."""

import sqlite3
import threading
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
    camera_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attendance_ts ON attendance(timestamp);
"""


class Repository:
    def __init__(self, db_path: Path | str):
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
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

    def log_attendance(
        self, person_id: int, name: str, employee_id: str, score: float, camera_id: str,
        when: datetime | None = None,
    ) -> None:
        ts = (when or datetime.now()).isoformat(timespec="seconds")
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO attendance (person_id, name, employee_id, timestamp, score, camera_id)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (person_id, name, employee_id, ts, round(score, 4), camera_id),
            )

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

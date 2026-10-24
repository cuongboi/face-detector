"""Turns per-frame matches into first-in / last-out attendance records.

A face must be matched to the same person for `vote_frames` consecutive frames of one track
before it counts. Each confirmed match goes to `Repository.record_sighting` (in, better in
within the first minute, then out), throttled per person: every `sighting_interval_seconds`
while the check-in can still improve, every `cooldown_seconds` once it is updating check-out.
"""

import itertools
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

import numpy as np

from core.vector_store import Match
from utils.config import Settings
from utils.face_utils import encode_snapshot, iou


class FaceState(StrEnum):
    UNKNOWN = "unknown"
    VERIFYING = "verifying"
    CHECKED_IN = "checked_in"  # check-in written (or improved) on this frame
    CHECKED_OUT = "checked_out"  # check-out written on this frame
    COOLDOWN = "cooldown"  # recognized, nothing written


@dataclass
class FaceStatus:
    bbox: np.ndarray
    state: FaceState
    match: Match | None


@dataclass
class _Track:
    id: int
    bbox: np.ndarray
    last_seen: float
    votes: deque = field(default_factory=deque)


def _one_face_per_person(
    observations: list[tuple[np.ndarray, Match | None]], threshold: float
) -> list[tuple[np.ndarray, Match | None]]:
    """Drop matches below `threshold`; if several faces in one frame match the same person,
    only the highest-scoring one keeps it (the others are someone else, e.g. a look-alike)."""
    best: dict[int, float] = {}
    for _, m in observations:
        if m is not None and m.score >= threshold:
            best[m.person_id] = max(best.get(m.person_id, -1.0), m.score)
    out, taken = [], set()
    for bbox, m in observations:
        keep = (
            m is not None and m.score >= threshold and m.person_id not in taken
            and m.score == best[m.person_id]
        )
        if keep:
            taken.add(m.person_id)
        out.append((bbox, m if keep else None))
    return out


class AttendanceService:
    def __init__(self, settings: Settings, repository):
        self.s = settings
        self.repo = repository
        self._tracks: list[_Track] = []
        self._ids = itertools.count()
        self._next_write: dict[int, float] = {}  # person_id -> earliest time for next write
        self._frame: np.ndarray | None = None
        self._snapshot: str | None = None

    def update(
        self, observations: list[tuple[np.ndarray, Match | None]], now: float,
        frame: np.ndarray | None = None,
    ) -> list[FaceStatus]:
        """`frame` (raw BGR) is the photo saved with records written on this call."""
        self._frame, self._snapshot = frame, None
        self._tracks = [t for t in self._tracks if now - t.last_seen <= self.s.track_ttl_seconds]
        statuses, used = [], set()
        for bbox, match in _one_face_per_person(observations, self.s.match_threshold):
            track = self._assign(bbox, now, used)
            track.votes.append(match)
            while len(track.votes) > self.s.vote_frames:
                track.votes.popleft()
            statuses.append(FaceStatus(bbox, self._decide(track, now), match))
        return statuses

    def _assign(self, bbox: np.ndarray, now: float, used: set[int]) -> _Track:
        best, best_iou = None, self.s.track_iou
        for t in self._tracks:
            if t.id in used:
                continue
            overlap = iou(t.bbox, bbox)
            if overlap >= best_iou:
                best, best_iou = t, overlap
        if best is None:
            best = _Track(next(self._ids), bbox, now, deque())
            self._tracks.append(best)
        best.bbox, best.last_seen = bbox, now
        used.add(best.id)
        return best

    def _decide(self, track: _Track, now: float) -> FaceState:
        current = track.votes[-1]
        if current is None:
            return FaceState.UNKNOWN
        pid = current.person_id
        if now < self._next_write.get(pid, float("-inf")):
            return FaceState.COOLDOWN
        stable = len(track.votes) >= self.s.vote_frames and all(
            v is not None and v.person_id == pid for v in track.votes
        )
        if not stable:
            return FaceState.VERIFYING
        score = float(np.mean([v.score for v in track.votes]))
        kind = self.repo.record_sighting(
            pid, current.name, current.employee_id, score, self.s.camera_id,
            datetime.fromtimestamp(now), self._snapshot_once, self.s.in_window_seconds,
        )
        wait = self.s.cooldown_seconds if kind == "out" else self.s.sighting_interval_seconds
        self._next_write[pid] = now + wait
        if kind == "out":
            return FaceState.CHECKED_OUT
        return FaceState.CHECKED_IN if kind else FaceState.COOLDOWN

    def _snapshot_once(self) -> str | None:
        """Encode this frame's photo at most once, and only when a record is written."""
        if self._snapshot is None and self._frame is not None:
            self._snapshot = encode_snapshot(
                self._frame, self.s.snapshot_width, self.s.snapshot_quality
            )
        return self._snapshot

    def forget_person(self, person_id: int) -> None:
        self._next_write.pop(person_id, None)

    def reset_cooldowns(self) -> None:
        self._next_write.clear()

"""Turns per-frame matches into attendance records.

A face must be matched to the same person for `vote_frames` consecutive frames of one
track before it is logged, then that person is on cooldown.
"""

import itertools
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

import numpy as np

from core.vector_store import Match
from utils.config import Settings
from utils.face_utils import iou


class FaceState(StrEnum):
    UNKNOWN = "unknown"
    VERIFYING = "verifying"
    CHECKED_IN = "checked_in"  # logged on this frame
    COOLDOWN = "cooldown"  # recognized, already logged recently


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


class AttendanceService:
    def __init__(self, settings: Settings, repository):
        self.s = settings
        self.repo = repository
        self._tracks: list[_Track] = []
        self._ids = itertools.count()
        self._last_logged: dict[int, float] = {}

    def update(
        self, observations: list[tuple[np.ndarray, Match | None]], now: float
    ) -> list[FaceStatus]:
        self._tracks = [t for t in self._tracks if now - t.last_seen <= self.s.track_ttl_seconds]
        statuses, used = [], set()
        for bbox, match in observations:
            if match is not None and match.score < self.s.match_threshold:
                match = None
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
        last = self._last_logged.get(pid)
        if last is not None and now - last < self.s.cooldown_seconds:
            return FaceState.COOLDOWN
        stable = len(track.votes) >= self.s.vote_frames and all(
            v is not None and v.person_id == pid for v in track.votes
        )
        if not stable:
            return FaceState.VERIFYING
        score = float(np.mean([v.score for v in track.votes]))
        self.repo.log_attendance(
            pid, current.name, current.employee_id, score, self.s.camera_id,
            when=datetime.fromtimestamp(now),
        )
        self._last_logged[pid] = now
        return FaceState.CHECKED_IN

    def forget_person(self, person_id: int) -> None:
        self._last_logged.pop(person_id, None)

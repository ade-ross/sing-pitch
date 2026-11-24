"""Replay of a saved Take's line: scores revealed as a clock passes them. Pure; the UI supplies the clock."""
from __future__ import annotations

from typing import Sequence

from .take_engine import FrameScore


class Replay:
    def __init__(self, scores: Sequence[FrameScore]):
        self._scores = list(scores)
        self._origin = self._scores[0].t if self._scores else 0.0
        self._next = 0

    @property
    def origin_s(self) -> float:
        """The Take time of its first frame: advance() counts elapsed_s from here."""
        return self._origin

    @property
    def finished(self) -> bool:
        return self._next >= len(self._scores)

    def advance(self, elapsed_s: float) -> list[FrameScore]:
        """Scores whose time (since the Take began) is up to elapsed_s and not yet returned."""
        start = self._next
        while self._next < len(self._scores) and self._scores[self._next].t - self._origin <= elapsed_s:
            self._next += 1
        return self._scores[start:self._next]

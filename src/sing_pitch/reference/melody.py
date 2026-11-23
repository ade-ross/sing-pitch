"""The Melody Reference: the pitch line RMVPE extracted from a song's vocal stem, looked up at song time t."""
from __future__ import annotations

from typing import ClassVar

import numpy as np

from ..library.ports import ReferenceLine
from ..music import A4_MIDI
from ..config import A4_HZ
from .base import Target

MIN_CONFIDENCE = 0.3      # RMVPE activation below this = reference unreliable (04-rmvpe notes); ticket #7 refines


def scored_mask(line: ReferenceLine, min_confidence: float = MIN_CONFIDENCE) -> np.ndarray:
    """Which reference frames are scored. For now a plain confidence threshold.
    Ticket #7 replaces this with Unscored stretches (pass its mask to Melody(scored=...))."""
    return np.asarray(line.confidence) >= min_confidence


class Melody:
    """Scores the voice against the original singer's line. t is song time (the engine has already removed
    the latency). Frames where the reference is unreliable, or outside the song, give an unscored Target."""

    kind: ClassVar[str] = "melody"

    def __init__(self, line: ReferenceLine, song_id: str, scored: np.ndarray | None = None):
        self.line = line
        self.song_id = song_id
        self._t = np.asarray(line.t, dtype=np.float64)
        self._midi = A4_MIDI + 12.0 * np.log2(np.maximum(np.asarray(line.hz, dtype=np.float64), 1e-6) / A4_HZ)
        self.scored = scored_mask(line) if scored is None else np.asarray(scored, dtype=bool)
        hop = float(np.median(np.diff(self._t))) if len(self._t) > 1 else 0.01
        self._tolerance = 1.5 * hop

    @classmethod
    def from_song(cls, song) -> Melody:
        return cls(song.load_reference(), song.id)

    def midi_line(self) -> tuple[np.ndarray, np.ndarray]:
        """(song time, MIDI) of the line, NaN where unscored: what the live view draws."""
        return self._t, np.where(self.scored, self._midi, np.nan)

    def target(self, t: float, sung_midi: float) -> Target:
        if not len(self._t):
            return Target(None)
        j = int(np.searchsorted(self._t, t))
        if j == len(self._t) or (j > 0 and t - self._t[j - 1] < self._t[j] - t):
            j -= 1
        if abs(self._t[j] - t) > self._tolerance or not self.scored[j]:
            return Target(None)
        return Target(float(self._midi[j]))

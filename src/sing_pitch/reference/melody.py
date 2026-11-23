"""The Melody Reference: the pitch line RMVPE extracted from a song's vocal stem, looked up at song time t."""
from __future__ import annotations

from typing import ClassVar

import numpy as np

from ..library.ports import ReferenceLine
from ..music import A4_MIDI
from ..config import A4_HZ
from .base import Target

MIN_CONFIDENCE = 0.3      # RMVPE activation below this = reference unreliable (04-rmvpe notes)
FILL_GAP_S = 0.12         # a dip shorter than this between two reliable frames is a consonant, not a stretch
MIN_SCORED_S = 0.12       # a reliable run shorter than this is a blip: too short to judge a voice against


def _runs(flags: np.ndarray) -> list[tuple[int, int, bool]]:
    """(start, end, value) of each run of equal values, end exclusive."""
    edges = np.flatnonzero(np.diff(flags.astype(np.int8))) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(flags)]])
    return [(int(a), int(b), bool(flags[a])) for a, b in zip(starts, ends)]


def scored_mask(
    line: ReferenceLine, min_confidence: float = MIN_CONFIDENCE, fill_gap_s: float = FILL_GAP_S,
    min_scored_s: float = MIN_SCORED_S,
) -> np.ndarray:
    """Which reference frames are scored; the rest are Unscored stretches. Confidence >= min_confidence, then
    smoothed so the mask does not flicker: low-confidence gaps shorter than fill_gap_s between two reliable
    frames count as reliable, then reliable runs shorter than min_scored_s count as unreliable."""
    mask = _reliable(line, min_confidence)
    if len(mask) < 2:
        return mask
    hop = float(np.median(np.diff(np.asarray(line.t, dtype=np.float64))))
    # compare whole frames: the float32 hop (0.0099999998) would put a 12-frame run just under 0.12 s
    fill_frames, min_frames = round(fill_gap_s / hop), round(min_scored_s / hop)
    for start, end, value in _runs(mask):
        if not value and start > 0 and end < len(mask) and end - start < fill_frames:
            mask[start:end] = True
    for start, end, value in _runs(mask):
        if value and end - start < min_frames:
            mask[start:end] = False
    return mask


def _reliable(line: ReferenceLine, min_confidence: float) -> np.ndarray:
    """Frames whose own Hz can be trusted: confident, and a real positive frequency."""
    hz = np.asarray(line.hz, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return (np.asarray(line.confidence) >= min_confidence) & np.isfinite(hz) & (hz > 0)


def _midi_line(line: ReferenceLine, min_confidence: float = MIN_CONFIDENCE) -> np.ndarray:
    """MIDI per frame from the reliable frames only. RMVPE's Hz at low confidence is arbitrary (99 to 1316 Hz on
    silence plus noise; 04-rmvpe notes), so frames the smoothing filled are interpolated between their reliable
    neighbours and held at the edges, never read from their own Hz. NaN when no frame is reliable."""
    ok = _reliable(line, min_confidence)
    if not ok.any():
        return np.full(len(ok), np.nan)
    hz = np.asarray(line.hz, dtype=np.float64)
    midi = A4_MIDI + 12.0 * np.log2(hz[ok] / A4_HZ)
    return np.interp(np.arange(len(ok)), np.flatnonzero(ok), midi)


class Melody:
    """Scores the voice against the original singer's line. t is song time (the engine has already removed
    the latency). Frames where the reference is unreliable, or outside the song, give an unscored Target."""

    kind: ClassVar[str] = "melody"

    def __init__(self, line: ReferenceLine, song_id: str, scored: np.ndarray | None = None):
        self.line = line
        self.song_id = song_id
        self._t = np.asarray(line.t, dtype=np.float64)
        self._midi = _midi_line(line)
        self.scored = scored_mask(line) if scored is None else np.asarray(scored, dtype=bool)
        hop = float(np.median(np.diff(self._t))) if len(self._t) > 1 else 0.01
        self._tolerance = 1.5 * hop

    @classmethod
    def from_song(cls, song) -> Melody:
        return cls(song.load_reference(), song.id)

    def unscored_spans(self) -> list[tuple[float, float]]:
        """(start, end) in song time of every Unscored stretch: what the live and replay views grey out."""
        if not len(self._t):
            return []
        hop = self._tolerance / 1.5
        return [(float(self._t[a] - hop / 2), float(self._t[b - 1] + hop / 2))
                for a, b, scored in _runs(self.scored) if not scored]

    def midi_line(self) -> tuple[np.ndarray, np.ndarray]:
        """(song time, MIDI) of the line, NaN where unscored: what the live view draws."""
        return self._t, np.where(self.scored, self._midi, np.nan)

    def target(self, t: float, sung_midi: float) -> Target:
        if not len(self._t):
            return Target(None)
        j = int(np.searchsorted(self._t, t))
        if j == len(self._t) or (j > 0 and t - self._t[j - 1] < self._t[j] - t):
            j -= 1
        if abs(self._t[j] - t) > self._tolerance or not self.scored[j] or not np.isfinite(self._midi[j]):
            return Target(None)
        return Target(float(self._midi[j]))

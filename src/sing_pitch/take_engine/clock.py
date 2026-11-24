"""Song clocks: which song time a voice frame is scored against. A voice frame at input time t answers what the
singer heard one calibrated latency earlier; the clock turns t into that song time (None = no song was playing)."""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from ..pitch import PitchFrame


class SongLog(Protocol):
    """What LogClock reads (playback.transport.SongTimeLog)."""

    def song_time(self, output_s: float) -> float | None: ...


class SongClock(Protocol):
    offset_s: float      # t - song time at the Take start: drawn on the plot and kept in takes.csv (latency_offset_s)

    def __call__(self, t: float) -> float | None: ...


class LinearClock:
    """Song time = t - offset_s. Nearest note and Key (offset 0), and Melody Takes saved before the song-time log
    (their row's latency_offset_s), which were played linearly from start to end."""

    def __init__(self, offset_s: float = 0.0):
        self.offset_s = offset_s

    def __call__(self, t: float) -> float | None:
        return t - self.offset_s


class LogClock:
    """The song time the Transport rendered at output time t - latency_s, where the output clock read
    output_at_input_zero_s when the input clock was at 0. Correct across pauses (later: seeks, loops, Speed)."""

    def __init__(self, log: SongLog, latency_s: float, output_at_input_zero_s: float):
        self._log = log
        self.latency_s = latency_s
        self.output_at_input_zero_s = output_at_input_zero_s
        self.offset_s = latency_s - output_at_input_zero_s

    def __call__(self, t: float) -> float | None:
        return self._log.song_time(t - self.offset_s)


class SampledClock:
    """A saved Take's own clock: the song time of each of its frames (None = paused), read between and just beyond
    them. Draws the Reference on a replay of a Take that was jumped into or paused."""

    def __init__(self, t: Sequence[float], song_times: Sequence[float | None]):
        self._t = np.asarray(t, dtype=np.float64)
        self._song = np.array([np.nan if s is None else s for s in song_times], dtype=np.float64)
        valid = np.flatnonzero(np.isfinite(self._song))
        self.offset_s = float(self._t[valid[0]] - self._song[valid[0]]) if len(valid) else 0.0

    def __call__(self, t: float) -> float | None:
        if not len(self._t):
            return None
        if t <= self._t[0]:
            edge, at = self._song[0], self._t[0]
        elif t >= self._t[-1]:
            edge, at = self._song[-1], self._t[-1]
        else:
            value = float(np.interp(t, self._t, self._song))
            return None if np.isnan(value) else value
        return None if np.isnan(edge) else float(edge + (t - at))


def linear_song_times(frames: Sequence[PitchFrame], offset_s: float = 0.0) -> list[float]:
    """Per-frame song times of a linearly played Take (t - offset_s)."""
    return [f.t - offset_s for f in frames]

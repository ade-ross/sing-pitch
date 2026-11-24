"""Song clocks: which song time a voice frame is scored against. A voice frame at input time t answers what the
singer heard one calibrated latency earlier; the clock turns t into that song time (None = no song was playing)."""
from __future__ import annotations

from typing import Protocol, Sequence

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


def linear_song_times(frames: Sequence[PitchFrame], offset_s: float = 0.0) -> list[float]:
    """Per-frame song times of a linearly played Take (t - offset_s)."""
    return [f.t - offset_s for f in frames]

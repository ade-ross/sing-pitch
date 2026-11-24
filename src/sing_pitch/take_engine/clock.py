"""Song clocks: which song time a voice frame is scored against. A voice frame at input time t answers what the
singer heard one calibrated latency earlier; the clock turns t into that song time (None = no song was playing)."""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from ..pitch import PitchFrame


class SongLog(Protocol):
    """What LogClock reads (playback.transport.SongTimeLog)."""

    def song_time(self, output_s: float) -> float | None: ...

    def tail(self) -> tuple[float, float | None, float] | None: ...


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

    def tail(self) -> tuple[float, float | None, float] | None:
        """(voice time, song time, rate) where the rendered song ends, for projecting the song that is still to be
        heard: the voice time is ahead of the live voice by about the latency. Song time None = paused there."""
        tail = self._log.tail()
        return None if tail is None else (tail[0] + self.offset_s, tail[1], tail[2])


class SampledClock:
    """A saved Take's own clock: the song time of each of its frames (None = paused), read between and just beyond
    them. Draws the Reference on a replay of a Take that was jumped into or paused. Beyond the last frame the song goes
    on at the rate it had there (the median step of its last frames, so a Take at Speed 50 % goes on at 0.5, not at
    1.0), and ends at song_end_s when that is given."""

    TAIL_FRAMES = 20     # how many of the last frames set the rate beyond the end

    def __init__(self, t: Sequence[float], song_times: Sequence[float | None], song_end_s: float | None = None):
        self._t = np.asarray(t, dtype=np.float64)
        self._song = np.array([np.nan if s is None else s for s in song_times], dtype=np.float64)
        self._song_end_s = song_end_s
        valid = np.flatnonzero(np.isfinite(self._song))
        self.offset_s = float(self._t[valid[0]] - self._song[valid[0]]) if len(valid) else 0.0
        self._rate = self._tail_rate()

    def _tail_rate(self) -> float:
        t, song = self._t[-self.TAIL_FRAMES:], self._song[-self.TAIL_FRAMES:]
        with np.errstate(invalid="ignore", divide="ignore"):
            steps = np.diff(song) / np.diff(t)
        steps = steps[np.isfinite(steps)]
        rate = float(np.median(steps)) if len(steps) else 1.0
        return rate if rate > 0 else 1.0

    def __call__(self, t: float) -> float | None:
        if not len(self._t):
            return None
        if t <= self._t[0]:
            edge, at, rate = self._song[0], self._t[0], 1.0
        elif t >= self._t[-1]:
            edge, at, rate = self._song[-1], self._t[-1], self._rate
        else:
            value = float(np.interp(t, self._t, self._song))
            return None if np.isnan(value) else value
        if np.isnan(edge):
            return None
        value = float(edge + (t - at) * rate)
        return None if self._song_end_s is not None and value > self._song_end_s else value


def project_song_time(dt: np.ndarray, song_s: float, rate: float, loop_end_s: float | None = None,
                      loop_pass_start_s: float = 0.0, song_end_s: float | None = None) -> np.ndarray:
    """The song time that will be heard `dt` output seconds after the song was at `song_s` playing at `rate` (song
    seconds per output second). Inside a loop the song runs to loop_end_s and wraps to loop_pass_start_s (the Lead-in),
    again and again; outside one it is NaN (no song) past song_end_s. Assumes what is playing now keeps playing: a
    seek, pause or Speed change that has not been rendered yet is not foreseen."""
    dt = np.asarray(dt, dtype=np.float64)
    song = song_s + dt * rate
    if loop_end_s is not None and rate > 0 and song_s < loop_end_s:
        pass_s = max(loop_end_s - loop_pass_start_s, 1e-6)
        over = song - loop_end_s                        # song seconds past the Section end, as if it ran on
        wrapped = loop_pass_start_s + np.mod(over, pass_s)
        return np.where(over >= 0, wrapped, song)
    if song_end_s is not None:
        song = np.where(song > song_end_s, np.nan, song)
    return song


def linear_song_times(frames: Sequence[PitchFrame], offset_s: float = 0.0) -> list[float]:
    """Per-frame song times of a linearly played Take (t - offset_s)."""
    return [f.t - offset_s for f in frames]

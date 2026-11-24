"""The plot's x axis: song time as m:ss, like the Timeline, not the Take's own clock."""
from __future__ import annotations

import weakref
from typing import Callable

import numpy as np
import pyqtgraph as pg


SPACINGS = (0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0)     # song seconds between ticks
MAX_TICKS = 10


class SongTimeAxis(pg.AxisItem):
    """Tick labels are the song time heard at each tick, asked of the window (set_source): a Melody Take that began
    at 1:10 labels its first tick 1:10, a looped Section shows its own times, a Speed 50 % Take shows song time
    passing at half the pace of the ticks. The source is held weakly: the window owns this axis."""

    def __init__(self):
        super().__init__("bottom")
        self._source: weakref.WeakMethod | None = None
        self._labels: dict[float, str] = {}

    def set_source(self, source: Callable[[np.ndarray], np.ndarray]) -> None:
        """source(view times) -> seconds (NaN where no song is heard); a bound method."""
        self._source = weakref.WeakMethod(source)

    def tickValues(self, minVal, maxVal, size):
        """Ticks at whole song seconds (so a label is exactly where its second is), found from the song time at the
        two ends of the view and taken as a straight line between them. Where that fails (no song at an end, a jump
        or a loop wrap inside the view) the ordinary ticks are used and labelled with the song time asked of the window."""
        self._labels = {}
        source = None if self._source is None else self._source()
        ends = np.array([minVal, maxVal], dtype=float)
        song = ends if source is None else np.asarray(source(ends), dtype=float)
        if not np.isfinite(song).all() or song[1] <= song[0] or maxVal <= minVal:
            return super().tickValues(minVal, maxVal, size)
        span = float(song[1] - song[0])
        step = next((c for c in SPACINGS if span / c <= MAX_TICKS), SPACINGS[-1])
        rate = span / (maxVal - minVal)                       # song seconds per view second
        first = int(np.ceil(song[0] / step - 1e-9))
        seconds = [k * step for k in range(first, int(np.floor(song[1] / step + 1e-9)) + 1)]
        values = [float(minVal + (t - song[0]) / rate) for t in seconds]
        self._labels = {v: format_song_time(t, step < 1.0) for v, t in zip(values, seconds)}
        return [(step / rate, values)]

    def tickStrings(self, values, scale, spacing):
        labelled = [self._labels.get(float(v)) for v in values]
        if all(label is not None for label in labelled):
            return labelled
        x = np.asarray(values, dtype=float)
        source = None if self._source is None else self._source()
        seconds = x if source is None else np.asarray(source(x * scale), dtype=float)
        steps = np.abs(np.diff(seconds))
        steps = steps[np.isfinite(steps) & (steps > 0)]
        tenths = len(steps) > 0 and float(steps.min()) < 1.0            # ticks closer than a second: show tenths
        return [format_song_time(t, tenths) for t in seconds]


def format_song_time(seconds: float, tenths: bool = False) -> str:
    """m:ss (m:ss.s when tenths); nothing before the song starts or where there is no song."""
    if not np.isfinite(seconds) or seconds < -1e-9:
        return ""
    if tenths:
        whole, rest = divmod(round(float(seconds), 1), 60.0)
        return f"{int(whole)}:{rest:04.1f}"
    whole = int(round(float(seconds)))
    return f"{whole // 60}:{whole % 60:02d}"

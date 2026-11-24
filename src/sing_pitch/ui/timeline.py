"""The Timeline: the whole song on one strip, the reference melody drawn on it, a playhead, click to jump.

It is a pyqtgraph plot whose x axis is SONG time in seconds (original tempo). Everything that is not "show the song
and say where it was clicked" is left to the window: the widget only emits `seek_requested(song_s)`.

For Sections (ticket #13): `plot` is the PlotItem, so a `pg.LinearRegionItem` (draggable edge handles included) can
be added with `timeline.plot.addItem(region)`; a region takes the mouse itself, a click anywhere else still seeks.
Dragging on empty strip is routed to `_TimelineView.mouseDragEvent`, which is where "drag across the timeline to mark
a Section" goes (it gets the song time through `timeline.song_time_at(ev.pos())`)."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from ..reference import Melody

LINE_STEP_S = 0.05          # the strip draws the reference at this resolution, however long the song
PLAYHEAD_PEN = pg.mkPen("#f5f5f5", width=2)
LINE_PEN = pg.mkPen("#5dade2", width=2)


class _TimelineView(pg.ViewBox):
    """The strip's view box: a click seeks, a drag is left alone (for #13) and nothing zooms or pans."""

    def __init__(self, owner: Timeline):
        super().__init__(enableMenu=False)
        self._owner = owner
        self.setMouseEnabled(False, False)

    def mouseClickEvent(self, ev) -> None:
        if ev.button() == QtCore.Qt.LeftButton:
            ev.accept()
            self._owner.seek_requested.emit(self._owner.song_time_at(ev.pos()))
        else:
            ev.ignore()

    def mouseDragEvent(self, ev, axis=None) -> None:
        ev.ignore()                  # ticket #13: drag across the strip marks a Section

    def wheelEvent(self, ev, axis=None) -> None:
        ev.ignore()


class Timeline(pg.GraphicsLayoutWidget):
    seek_requested = QtCore.Signal(float)        # song time (s) the user clicked

    def __init__(self):
        super().__init__()
        self.setBackground("#1e1e1e")
        self.setFixedHeight(110)
        self._view = _TimelineView(self)
        self.plot: pg.PlotItem = self.addPlot(viewBox=self._view)
        self._view.disableAutoRange()
        self.plot.hideButtons()
        self.plot.hideAxis("left")
        self.plot.getAxis("bottom").setLabel("song time", units="s")
        self._line = self.plot.plot(pen=LINE_PEN, connect="finite")
        self._playhead = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=PLAYHEAD_PEN)
        self._playhead.setZValue(10)
        self.plot.addItem(self._playhead)
        self._duration = 1.0
        self.set_duration(1.0)

    def set_melody(self, melody: Melody | None, duration_s: float | None = None) -> None:
        """Draw the reference melody over the whole song (None clears the strip)."""
        if melody is None:
            self._line.setData([], [])
            return
        t, midi = melody.midi_line()
        end = float(t[-1]) if len(t) else 0.0
        self.set_duration(max(duration_s or 0.0, end, 1.0))
        step = max(1, round(LINE_STEP_S / float(np.median(np.diff(t))))) if len(t) > 1 else 1
        t, midi = t[::step], midi[::step]
        self._line.setData(t, midi)
        if np.isfinite(midi).any():
            lo, hi = float(np.nanmin(midi)), float(np.nanmax(midi))
            self.plot.setYRange(lo - 2, hi + 2, padding=0)

    def set_duration(self, duration_s: float) -> None:
        self._duration = float(duration_s)
        self.plot.setLimits(xMin=0.0, xMax=self._duration)      # limits first: they clamp the range
        self.plot.setXRange(0.0, self._duration, padding=0)

    @property
    def duration_s(self) -> float:
        return self._duration

    def set_playhead(self, song_s: float) -> None:
        self._playhead.setPos(song_s)

    @property
    def playhead_s(self) -> float:
        return float(self._playhead.value())

    def song_time_at(self, view_pos) -> float:
        """Song time (s, clamped to the song) of a position in the view box's own coordinates (a mouse event's pos)."""
        x = self._view.mapToView(view_pos).x()
        return float(min(max(x, 0.0), self._duration))

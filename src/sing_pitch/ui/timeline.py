"""The Timeline: the whole song on one strip, the reference melody drawn on it, a playhead, click to jump.

It is a pyqtgraph plot whose x axis is SONG time in seconds (original tempo). Everything that is not "show the song
and say where it was clicked" is left to the window: the widget only emits `seek_requested(song_s)`.

Sections: a left-button drag across the strip marks one (a `pg.LinearRegionItem` whose edges are handles that can be
dragged, and whose body can be dragged as a whole). A finished drag emits `section_marked(start, end)`, a finished
edit of the shown Section `section_changed(start, end)`; the window decides what they mean (save, loop). A click outside the Section still seeks."""
from __future__ import annotations

import weakref

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from ..reference import Melody

LINE_STEP_S = 0.05          # the strip draws the reference at this resolution, however long the song
PLAYHEAD_PEN = pg.mkPen("#f5f5f5", width=2)
LINE_PEN = pg.mkPen("#5dade2", width=2)
SECTION_BRUSH = pg.mkBrush(241, 196, 15, 50)
MIN_SECTION_S = 0.2         # a shorter drag is a shaky click, not a Section


class _TimelineView(pg.ViewBox):
    """The strip's view box: a click seeks, a drag marks a Section, nothing zooms or pans."""

    def __init__(self, owner: Timeline):
        super().__init__(enableMenu=False)
        # weak: the strip owns this box, so a strong ref back is a cycle that only gc frees (ui/gui_gc.py: why not)
        self._owner_ref = weakref.ref(owner)
        self.setMouseEnabled(False, False)

    @property
    def _owner(self) -> Timeline:
        return self._owner_ref()

    def mouseClickEvent(self, ev) -> None:
        if ev.button() == QtCore.Qt.LeftButton:
            ev.accept()
            self._owner.seek_requested.emit(self._owner.song_time_at(ev.pos()))
        else:
            ev.ignore()

    def mouseDragEvent(self, ev, axis=None) -> None:
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        ev.accept()
        a, b = self._owner.song_time_at(ev.buttonDownPos()), self._owner.song_time_at(ev.pos())
        self._owner._drag_section(a, b, ev.isFinish())

    def wheelEvent(self, ev, axis=None) -> None:
        ev.ignore()


class Timeline(pg.GraphicsLayoutWidget):
    seek_requested = QtCore.Signal(float)        # song time (s) the user clicked
    section_marked = QtCore.Signal(float, float)    # (start, end) of a new Section dragged across the strip
    section_changed = QtCore.Signal(float, float)   # (start, end) after he moved the shown Section's edges or body

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
        self.region = pg.LinearRegionItem((0.0, 1.0), brush=SECTION_BRUSH, swapMode="sort")
        self.region.setZValue(5)
        self.region.setVisible(False)
        self.plot.addItem(self.region)
        self.region.sigRegionChangeFinished.connect(self._region_edited)
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

    @property
    def section(self) -> tuple[float, float] | None:
        if not self.region.isVisible():
            return None
        start, end = self.region.getRegion()
        return float(start), float(end)

    def set_section(self, start_s: float | None, end_s: float | None = None) -> None:
        """Show a Section (None hides it). Does not emit section_changed."""
        if start_s is None:
            self.region.setVisible(False)
            return
        self.region.blockSignals(True)
        self.region.setRegion((min(start_s, end_s), max(start_s, end_s)))
        self.region.blockSignals(False)
        self.region.setBounds((0.0, self._duration))
        self.region.setVisible(True)

    def _drag_section(self, a: float, b: float, finished: bool) -> None:
        if abs(b - a) < MIN_SECTION_S:
            return
        self.set_section(a, b)
        if finished:
            self.section_marked.emit(*self.section)

    def _region_edited(self, *_) -> None:
        """An edge or the body was moved: an edge dropped closer than MIN_SECTION_S to the other is pushed back to
        MIN_SECTION_S (inside the song) before the edit is reported."""
        if not self.region.isVisible():
            return
        start, end = self.section
        if end - start < MIN_SECTION_S:
            width = min(MIN_SECTION_S, self._duration)
            end = min(max(end, start + width), self._duration)
            start = max(0.0, min(start, end - width))
            self.set_section(start, end)
        self.section_changed.emit(*self.section)

    def song_time_at(self, view_pos) -> float:
        """Song time (s, clamped to the song) of a position in the view box's own coordinates (a mouse event's pos)."""
        x = self._view.mapToView(view_pos).x()
        return float(min(max(x, 0.0), self._duration))

"""Live view: pitch line on a note grid coloured by Band, current note + cents, Take summary on Stop."""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..audio_input import AudioInput, SoundDeviceInput, list_inputs, pick_input
from ..config import data_dir
from ..live import LiveRunner
from ..music import NOTE_NAMES
from ..pitch import PitchTracker
from ..reference import Band, NearestNote
from ..take_engine import FrameScore, TakeRecord

WINDOW_S = 10.0       # seconds of line on screen
TICK_MS = 33          # ~30 Hz redraw
HALF_RANGE = 8        # semitones shown above and below the centre
BAND_COLOUR = {Band.GREEN: "#2ecc71", Band.YELLOW: "#f1c40f", Band.RED: "#e74c3c"}
BRUSH = {b: pg.mkBrush(c) for b, c in BAND_COLOUR.items()}

pg.setConfigOptions(antialias=False)


def describe(record: TakeRecord) -> str:
    s = record.summary
    if s.n_scored == 0:
        return f"Take {record.id}: no voiced sound heard. Saved."
    return (
        f"In tune {s.pct_in_tune:.0f}%   |   median {s.median_abs_cents:.0f} cents off   |   "
        f"tends {s.tendency}   |   range {s.lowest_note}-{s.highest_note}   |   "
        f"{s.n_scored * 0.016:.1f} s sung   |   saved as {record.id}"
    )


class MainWindow(QtWidgets.QWidget):
    def __init__(
        self,
        tracker: PitchTracker,
        home: Path | None = None,
        input_factory: Callable[[int], AudioInput] = SoundDeviceInput,
    ):
        super().__init__()
        self.setWindowTitle("sing-pitch")
        self.resize(1000, 620)
        self._tracker, self._home, self._input_factory = tracker, home or data_dir(), input_factory
        self._runner: LiveRunner | None = None
        self._t: deque[float] = deque()
        self._midi: deque[float] = deque()
        self._band: deque[Band | None] = deque()
        self._centre = 60.0

        self.device_box = QtWidgets.QComboBox()
        self._fill_devices()
        self.button = QtWidgets.QPushButton("Start")
        self.button.clicked.connect(self.toggle)
        self.history_button = QtWidgets.QPushButton("History")
        self.history_button.clicked.connect(self.open_history)
        self.history_window = None
        self.note_label = QtWidgets.QLabel("-")
        self.note_label.setStyleSheet("font-size: 40px; font-weight: bold;")
        self.summary_label = QtWidgets.QLabel("Press Start and sing.")
        self.summary_label.setWordWrap(True)

        self.plot_widget = pg.PlotWidget(background="#1e1e1e")
        plot = self.plot_widget.getPlotItem()
        plot.setMouseEnabled(False, False)
        plot.hideButtons()
        plot.showGrid(y=True, alpha=0.35)
        self._curve = plot.plot(pen=pg.mkPen("#777777", width=1), connect="finite")
        self._dots = pg.ScatterPlotItem(size=6, pen=None)
        plot.addItem(self._dots)
        self._plot = plot
        self._set_y(self._centre)

        top = QtWidgets.QHBoxLayout()
        for w in (QtWidgets.QLabel("Input"), self.device_box, self.button, self.history_button):
            top.addWidget(w)
        top.addStretch(1)
        top.addWidget(self.note_label)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.plot_widget, 1)
        layout.addWidget(self.summary_label)

        self._timer = QtCore.QTimer(self)
        self._timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._timer.timeout.connect(self.tick)

    def _fill_devices(self) -> None:
        try:
            devices = list_inputs()
        except Exception as exc:  # PortAudio missing or no devices: the window still opens
            self.device_box.addItem(f"no input ({exc})", None)
            return
        for d in devices:
            self.device_box.addItem(d["name"], d["index"])
        best = pick_input(devices)
        if best is not None:
            self.device_box.setCurrentIndex(self.device_box.findData(best))

    def _set_y(self, centre: float) -> None:
        self._centre = centre
        lo, hi = int(round(centre)) - HALF_RANGE, int(round(centre)) + HALF_RANGE
        self._plot.setYRange(lo, hi, padding=0)
        ticks = [(m, f"{NOTE_NAMES[m % 12]}{m // 12 - 1}") for m in range(lo - 1, hi + 2)]
        self._plot.getAxis("left").setTicks([ticks])

    def open_history(self) -> None:
        from .history_view import HistoryView

        if self.history_window is None:
            self.history_window = HistoryView(self._home)
        else:
            self.history_window.reload()
        self.history_window.show()
        self.history_window.raise_()

    def toggle(self) -> None:
        if self._runner is None:
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        index = self.device_box.currentData()
        try:
            self._runner = LiveRunner(self._input_factory(index), self._tracker, NearestNote(), self._home)
            self._runner.start()
        except Exception as exc:
            self._runner = None
            self.summary_label.setText(f"Could not open the input: {exc}")
            return
        self._t.clear(), self._midi.clear(), self._band.clear()
        self.button.setText("Stop")
        self.device_box.setEnabled(False)
        self.summary_label.setText("Singing...")
        self._timer.start(TICK_MS)

    def _stop(self) -> None:
        self._timer.stop()
        record = self._runner.stop()
        self.tick(drain_only=True)
        self._runner = None
        self.button.setText("Start")
        self.device_box.setEnabled(True)
        self.summary_label.setText(describe(record))

    def ingest(self, scores: list[FrameScore]) -> None:
        for s in scores:
            self._t.append(s.t)
            self._midi.append(np.nan if s.midi is None else s.midi)
            self._band.append(s.band)
        if scores:
            while self._t[-1] - self._t[0] > WINDOW_S:
                self._t.popleft(), self._midi.popleft(), self._band.popleft()
            last = scores[-1]
            if last.band is None:
                self.note_label.setText("-")
            else:
                self.note_label.setText(f"{last.note_name}  {last.cents:+.0f} c")
            voiced = [s.midi for s in scores if s.midi is not None]
            if voiced and abs(voiced[-1] - self._centre) > HALF_RANGE - 2:
                self._set_y(voiced[-1])

    def tick(self, drain_only: bool = False) -> None:
        if self._runner is not None:
            self.ingest(self._runner.drain())
        if not self._t:
            return
        t, midi = np.fromiter(self._t, float), np.fromiter(self._midi, float)
        self._curve.setData(t, midi)
        voiced = ~np.isnan(midi)
        bands = [b for b, v in zip(self._band, voiced) if v]
        self._dots.setData(t[voiced], midi[voiced], brush=[BRUSH[b] for b in bands])
        self._plot.setXRange(t[-1] - WINDOW_S, t[-1], padding=0)

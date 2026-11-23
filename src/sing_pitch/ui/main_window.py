"""Live view: pitch line on a note grid coloured by Band, current note + cents, Take summary on Stop."""
from __future__ import annotations

from collections import deque
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..audio_input import AudioInput, SoundDeviceInput, list_inputs, pick_input
from ..config import data_dir
from ..live import LiveRunner
from ..music import NOTE_NAMES, note_name
from ..pitch import PitchTracker
from ..playback import DronePlayer
from ..reference import Band, Key, NearestNote, Reference
from ..replay import Replay
from ..take_engine import FrameScore, TakeRecord, replay_scores
from ..take_store import last_take_id, load_frames, saved_reference

WINDOW_S = 10.0       # seconds of line on screen
TICK_MS = 33          # ~30 Hz redraw
HALF_RANGE = 8        # semitones shown above and below the centre
BAND_COLOUR = {Band.GREEN: "#2ecc71", Band.YELLOW: "#f1c40f", Band.RED: "#e74c3c"}
BRUSH = {b: pg.mkBrush(c) for b, c in BAND_COLOUR.items()}

pg.setConfigOptions(antialias=False)


def steadiness_text(s) -> str:
    if s.steadiness_cents is None:
        return "steadiness: no held note"
    return f"steadiness {s.steadiness_cents:.0f} cents"


def describe(record: TakeRecord) -> str:
    s = record.summary
    if s.n_scored == 0:
        return f"Take {record.id}: no voiced sound heard. Saved."
    key = f"{record.key}   |   {s.out_of_key_count} out of key   |   " if record.reference_kind == "key" else ""
    return (
        f"{key}In tune {s.pct_in_tune:.0f}%   |   median {s.median_abs_cents:.0f} cents off   |   "
        f"tends {s.tendency}   |   {steadiness_text(s)}   |   range {s.lowest_note}-{s.highest_note}   |   "
        f"{s.n_scored * 0.016:.1f} s sung   |   saved as {record.id}"
    )


class MainWindow(QtWidgets.QWidget):
    def __init__(
        self,
        tracker: PitchTracker,
        home: Path | None = None,
        input_factory: Callable[[int], AudioInput] = SoundDeviceInput,
        drone_factory: Callable[[], DronePlayer] = DronePlayer,
    ):
        super().__init__()
        self.setWindowTitle("sing-pitch")
        self.resize(1000, 620)
        self._tracker, self._home, self._input_factory = tracker, home or data_dir(), input_factory
        self._drone_factory = drone_factory
        self._drone: DronePlayer | None = None
        self._runner: LiveRunner | None = None
        self._replay: Replay | None = None
        self._replay_started = 0.0
        self._t: deque[float] = deque()
        self._midi: deque[float] = deque()
        self._band: deque[Band | None] = deque()
        self._out: deque[bool] = deque()          # True = sung note outside the Key
        self._centre = 60.0

        self.device_box = QtWidgets.QComboBox()
        self._fill_devices()
        self.reference_box = QtWidgets.QComboBox()
        self.reference_box.addItems(["Nearest note", "Key"])
        self.key_box = QtWidgets.QComboBox()
        for k in Key.all():
            self.key_box.addItem(k.name, k)
        self.drone_box = QtWidgets.QCheckBox("Drone")
        self.reference_box.currentTextChanged.connect(self._reference_changed)
        self._reference_changed()
        self.button = QtWidgets.QPushButton("Start")
        self.button.clicked.connect(self.toggle)
        self.replay_button = QtWidgets.QPushButton("Replay last Take")
        self.replay_button.clicked.connect(self.toggle_replay)
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
        for w in (QtWidgets.QLabel("Input"), self.device_box, QtWidgets.QLabel("Reference"), self.reference_box,
                  self.key_box, self.drone_box, self.button, self.replay_button, self.history_button):
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

    def _reference_changed(self) -> None:
        is_key = self.reference_box.currentText() == "Key"
        self.key_box.setEnabled(is_key)
        self.drone_box.setEnabled(is_key)

    def _reference(self) -> Reference:
        if self.reference_box.currentText() == "Key":
            return self.key_box.currentData()
        return NearestNote()

    def _set_y(self, centre: float) -> None:
        self._centre = centre
        lo, hi = int(round(centre)) - HALF_RANGE, int(round(centre)) + HALF_RANGE
        self._plot.setYRange(lo, hi, padding=0)
        ticks = [(m, f"{NOTE_NAMES[m % 12]}{m // 12 - 1}") for m in range(lo - 1, hi + 2)]
        self._plot.getAxis("left").setTicks([ticks])

    def toggle_replay(self) -> None:
        """Play the most recent saved Take's pitch line back through the live view."""
        if self._replay is not None:
            self._end_replay()
            return
        take_id = last_take_id(self._home)
        if take_id is None:
            self.summary_label.setText("No saved Take to replay yet.")
            return
        self._replay = Replay(replay_scores(load_frames(self._home, take_id), saved_reference(self._home, take_id)))
        self._replay_started = time.monotonic()
        self._t.clear(), self._midi.clear(), self._band.clear(), self._out.clear()
        self.replay_button.setText("Stop replay")
        self.button.setEnabled(False)
        self.summary_label.setText(f"Replaying Take {take_id}...")
        self._timer.start(TICK_MS)

    def _end_replay(self) -> None:
        self._timer.stop()
        self._replay = None
        self.replay_button.setText("Replay last Take")
        self.button.setEnabled(True)

    def open_history(self) -> None:
        from .history_view import HistoryView

        if self.history_window is None:
            self.history_window = HistoryView(self._home)
        else:
            self.history_window.reload()
        self.history_window.show()
        self.history_window.raise_()

    def toggle(self) -> None:
        if self._replay is not None:
            return
        if self._runner is None:
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        index = self.device_box.currentData()
        try:
            reference = self._reference()
            self._runner = LiveRunner(self._input_factory(index), self._tracker, reference, self._home)
            self._runner.start()
        except Exception as exc:
            self._runner = None
            self.summary_label.setText(f"Could not open the input: {exc}")
            return
        self._t.clear(), self._midi.clear(), self._band.clear(), self._out.clear()
        self.button.setText("Stop")
        for w in (self.device_box, self.reference_box, self.key_box, self.drone_box):
            w.setEnabled(False)
        self.replay_button.setEnabled(False)
        self.summary_label.setText("Singing...")
        if reference.kind == "key" and self.drone_box.isChecked():
            try:
                self._drone = self._drone_factory()
                self._drone.start(reference)
            except Exception as exc:  # the Take goes on without the Drone
                self._drone = None
                self.summary_label.setText(f"Singing... (Drone unavailable: {exc})")
        self._timer.start(TICK_MS)

    def _stop(self) -> None:
        self._timer.stop()
        if self._drone is not None:
            self._drone.stop()
            self._drone = None
        record = self._runner.stop()
        self.tick(drain_only=True)
        self._runner = None
        self.button.setText("Start")
        self.device_box.setEnabled(True)
        self.reference_box.setEnabled(True)
        self._reference_changed()
        self.replay_button.setEnabled(True)
        self.summary_label.setText(describe(record))

    def ingest(self, scores: list[FrameScore]) -> None:
        for s in scores:
            self._t.append(s.t)
            self._midi.append(np.nan if s.midi is None else s.midi)
            self._band.append(s.band)
            self._out.append(s.in_key is False)
        if scores:
            while self._t[-1] - self._t[0] > WINDOW_S:
                self._t.popleft(), self._midi.popleft(), self._band.popleft(), self._out.popleft()
            last = scores[-1]
            if last.band is None:
                self.note_label.setText("-")
            elif last.in_key is False:
                self.note_label.setText(f"{note_name(round(last.midi))}  out of key")
            else:
                self.note_label.setText(f"{last.note_name}  {last.cents:+.0f} c")
            voiced = [s.midi for s in scores if s.midi is not None]
            if voiced and abs(voiced[-1] - self._centre) > HALF_RANGE - 2:
                self._set_y(voiced[-1])

    def tick(self, drain_only: bool = False) -> None:
        if self._runner is not None:
            self.ingest(self._runner.drain())
        elif self._replay is not None:
            self.ingest(self._replay.advance(time.monotonic() - self._replay_started))
        if not self._t:
            return
        t, midi = np.fromiter(self._t, float), np.fromiter(self._midi, float)
        self._curve.setData(t, midi)
        voiced = ~np.isnan(midi)
        bands = [b for b, v in zip(self._band, voiced) if v]
        out = [o for o, v in zip(self._out, voiced) if v]
        self._dots.setData(t[voiced], midi[voiced], brush=[BRUSH[b] for b in bands],
                           symbol=["x" if o else "o" for o in out],   # out-of-key notes are drawn as big crosses
                           size=[11 if o else 6 for o in out])
        self._plot.setXRange(t[-1] - WINDOW_S, t[-1], padding=0)
        if self._replay is not None and self._replay.finished:
            self._end_replay()
            self.summary_label.setText("Replay finished.")

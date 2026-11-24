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
from ..calibration import CalibrationRun
from ..library import Library
from ..live import LiveRunner, Playback
from ..music import NOTE_NAMES, note_name
from ..pitch import PitchTracker
from ..playback import BackingPlayer, DronePlayer
from ..playback.outputs import OutputKind, default_output_name, headphone_warning, output_kind
from ..reference import Band, Key, Melody, NearestNote, Reference
from ..replay import Replay
from ..settings import DEFAULT_OUTPUT, Settings
from ..take_engine import FrameScore, TakeRecord, replay_scores
from ..take_store import last_take_id, load_frames, saved_latency, saved_reference, saved_song_times

WINDOW_S = 10.0       # seconds of line on screen
TICK_MS = 33          # ~30 Hz redraw
END_GRACE_S = 1.0     # Melody: keep listening this long after the Backing ends
HALF_RANGE = 8        # semitones shown above and below the centre
BAND_COLOUR = {Band.GREEN: "#2ecc71", Band.YELLOW: "#f1c40f", Band.RED: "#e74c3c"}
UNSCORED_COLOUR = "#8a8a8a"   # sung where the Reference has nothing reliable: shown, never scored
BRUSH = {b: pg.mkBrush(c) for b, c in BAND_COLOUR.items()}
BRUSH[None] = pg.mkBrush(UNSCORED_COLOUR)
UNSCORED_SPAN_BRUSH = pg.mkBrush(138, 138, 138, 70)   # Unscored stretch of the Melody, greyed on the plot
REFERENCE_PEN = pg.mkPen("#5dade2", width=4)


def _default_output() -> tuple[str | None, OutputKind]:
    name = default_output_name()
    return name, output_kind(name) if name else "unknown"

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
    if record.reference_kind == "melody":
        key = f"{record.song_id}   |   octave shift {s.octave_shift:+d}   |   "
    return (
        f"{key}In tune {s.pct_in_tune:.0f}%   |   median {s.median_abs_cents:.0f} cents off   |   "
        f"tends {s.tendency}   |   {steadiness_text(s)}   |   range {s.lowest_note}-{s.highest_note}   |   "
        f"{s.n_scored * 0.016:.1f} s sung   |   saved as {record.id}"
    )


def _volume_slider(tip: str) -> QtWidgets.QSlider:
    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider.setRange(0, 100)
    slider.setValue(100)
    slider.setMaximumWidth(140)
    slider.setToolTip(tip)
    return slider


class MainWindow(QtWidgets.QWidget):
    def __init__(
        self,
        tracker: PitchTracker,
        home: Path | None = None,
        input_factory: Callable[[int], AudioInput] = SoundDeviceInput,
        drone_factory: Callable[[], DronePlayer] = DronePlayer,
        backing_factory: Callable[[Path, Path], Playback] = BackingPlayer.from_file,
        output_info: Callable[[], tuple[str | None, OutputKind]] = _default_output,
        calibration_player_factory: Callable | None = None,
    ):
        super().__init__()
        self.setWindowTitle("sing-pitch")
        self.resize(1000, 620)
        self._tracker, self._home, self._input_factory = tracker, home or data_dir(), input_factory
        self._drone_factory = drone_factory
        self._drone: DronePlayer | None = None
        self._backing_factory = backing_factory
        self._backing: Playback | None = None
        self._output_info = output_info
        self._calibration_player_factory = calibration_player_factory
        self._calibration: CalibrationRun | None = None
        self._settings = Settings.load(self._home)
        self._melody: Melody | None = None           # drawn beside the live line during Melody Takes and replays
        self._melody_latency = 0.0
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
        self.reference_box.addItems(["Nearest note", "Key", "Melody"])
        self.key_box = QtWidgets.QComboBox()
        for k in Key.all():
            self.key_box.addItem(k.name, k)
        self.drone_box = QtWidgets.QCheckBox("Drone")
        self.song_box = QtWidgets.QComboBox()
        self.reload_songs()
        self.guide_box = QtWidgets.QCheckBox("Guide vocal")
        self.guide_slider = _volume_slider("Guide vocal volume")
        self.backing_slider = _volume_slider("Backing volume")
        for w in (self.guide_box, self.guide_slider, self.backing_slider):
            (w.toggled if w is self.guide_box else w.valueChanged).connect(self._apply_gains)
        self.calibrate_button = QtWidgets.QPushButton("Calibrate latency")
        self.calibrate_button.clicked.connect(self.start_calibration)
        self.output_label = QtWidgets.QLabel("")
        self.output_label.setWordWrap(True)
        self.output_label.setStyleSheet("color: #e67e22;")
        self.headphones_button = QtWidgets.QPushButton("These are headphones")
        self.headphones_button.clicked.connect(self.confirm_headphones)
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
        self._unscored = pg.BarGraphItem(x0=[], x1=[], y0=[], y1=[], brush=UNSCORED_SPAN_BRUSH, pen=None)
        self._unscored.setZValue(-5)
        plot.addItem(self._unscored)
        self._ref_curve = plot.plot(pen=REFERENCE_PEN, connect="finite")
        self._curve = plot.plot(pen=pg.mkPen("#777777", width=1), connect="finite")
        self._dots = pg.ScatterPlotItem(size=6, pen=None)
        plot.addItem(self._dots)
        self._plot = plot
        self._set_y(self._centre)

        top = QtWidgets.QHBoxLayout()
        for w in (QtWidgets.QLabel("Input"), self.device_box, QtWidgets.QLabel("Reference"), self.reference_box,
                  self.key_box, self.drone_box, self.song_box, self.button, self.replay_button, self.history_button,
                  self.calibrate_button):
            top.addWidget(w)
        top.addStretch(1)
        top.addWidget(self.note_label)
        mix = QtWidgets.QHBoxLayout()
        for w in (self.guide_box, self.guide_slider, QtWidgets.QLabel("Backing"), self.backing_slider):
            mix.addWidget(w)
        mix.addStretch(1)
        warn = QtWidgets.QHBoxLayout()
        warn.addWidget(self.output_label, 1)
        warn.addWidget(self.headphones_button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(mix)
        layout.addLayout(warn)
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

    def reload_songs(self) -> None:
        self.song_box.clear()
        for song in Library(self._home).list_songs():
            self.song_box.addItem(song.meta.title, song.id)
        if not self.song_box.count():
            self.song_box.addItem("no songs: sing-pitch import <file or link>", None)

    def _reference_changed(self) -> None:
        is_key = self.reference_box.currentText() == "Key"
        is_melody = self.reference_box.currentText() == "Melody"
        self.key_box.setEnabled(is_key)
        self.drone_box.setEnabled(is_key)
        self.song_box.setEnabled(is_melody)
        for w in (self.guide_box, self.guide_slider, self.backing_slider):
            w.setEnabled(is_melody)
        self.update_output_warning()

    def _apply_gains(self, *_) -> None:
        """Slider and checkbox changes reach the player at once, also mid-Take; it ramps them within a block."""
        if self._backing is not None:
            self._backing.set_gains(*self._gains())

    def _gains(self) -> tuple[float, float]:
        """(Backing, Guide vocal) linear gains from the controls; the Guide vocal is 0 while its box is unticked."""
        return self.backing_slider.value() / 100, self.guide_slider.value() / 100 if self.guide_box.isChecked() else 0.0

    def update_output_warning(self) -> None:
        """Melody only: warn when the Backing would go to speakers (it leaks into the mic)."""
        text = None
        if self.reference_box.currentText() == "Melody":
            name, kind = self._output_info()
            if not self._settings.headphones_confirmed(name):
                text = headphone_warning(kind, name)
        self.output_label.setText(text or "")
        self.output_label.setVisible(bool(text))
        self.headphones_button.setVisible(bool(text) and "Cannot tell" in (text or ""))

    def confirm_headphones(self) -> None:
        name, _ = self._output_info()
        if name:
            self._settings.confirm_headphones(name)
            self._settings.save()
        self.update_output_warning()

    def _reference(self) -> Reference:
        if self.reference_box.currentText() == "Key":
            return self.key_box.currentData()
        if self.reference_box.currentText() == "Melody":
            song_id = self.song_box.currentData()
            if song_id is None:
                raise ValueError("no song in the library yet: import one with `sing-pitch import <file or link>`")
            return Melody.from_song(Library(self._home).get(song_id))
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
        try:
            reference = saved_reference(self._home, take_id)       # a Melody Take whose song was removed raises here
            latency = saved_latency(self._home, take_id)
            song_times = saved_song_times(self._home, take_id)
            replay = Replay(replay_scores(load_frames(self._home, take_id), reference, song_times=song_times))
        except Exception as exc:
            self.summary_label.setText(f"Cannot replay Take {take_id}: {exc}")
            return
        self._replay = replay
        self._replay_started = time.monotonic()
        self._show_melody(reference, latency)
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

    def _show_melody(self, reference: Reference, latency_s: float) -> None:
        self._melody = reference if reference.kind == "melody" else None
        self._melody_latency = latency_s
        self._ref_curve.setData([], [])
        self._unscored.setOpts(x0=[], x1=[], y0=[], y1=[])
        if self._melody is not None:
            _, midi = self._melody.midi_line()
            if np.isfinite(midi).any():
                self._set_y(float(np.nanmedian(midi)))

    def _start(self) -> None:
        index = self.device_box.currentData()
        try:
            reference = self._reference()
        except Exception as exc:
            self.summary_label.setText(f"Cannot start: {exc}")
            return
        backing, latency = None, 0.0
        try:
            if reference.kind == "melody":
                output, _ = self._output_info()
                latency = self._settings.latency_for(output)
                song = Library(self._home).get(reference.song_id)
                backing = self._backing_factory(song.backing_path, song.vocals_path)
                backing.set_gains(*self._gains())
            self._runner = LiveRunner(self._input_factory(index), self._tracker, reference, self._home,
                                      backing=backing, latency_s=latency)
            self._runner.start()
        except Exception as exc:
            self._runner = None
            self.summary_label.setText(f"Could not open the input or the output: {exc}")
            return
        self._backing = backing
        self._show_melody(reference, latency)
        self._t.clear(), self._midi.clear(), self._band.clear(), self._out.clear()
        self.button.setText("Stop")
        for w in (self.device_box, self.reference_box, self.key_box, self.drone_box, self.song_box,
                  self.calibrate_button):
            w.setEnabled(False)
        self.replay_button.setEnabled(False)
        self.summary_label.setText("Singing...")
        if reference.kind == "melody" and not self._settings.is_calibrated(self._output_info()[0]):
            self.summary_label.setText(f"Singing... (latency not calibrated for this output: using {latency:.2f} s)")
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
        try:
            record, error = self._runner.stop(), None
            self._backing = None
            self.tick(drain_only=True)
        except Exception as exc:        # the worker thread crashed: the Take is lost, say why
            record, error = None, exc
        self._backing = None
        self._runner = None
        self.button.setText("Start")
        self.device_box.setEnabled(True)
        self.reference_box.setEnabled(True)
        self.calibrate_button.setEnabled(True)
        self._reference_changed()
        self.replay_button.setEnabled(True)
        self.summary_label.setText(describe(record) if record else f"The Take failed: {error}")

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

    def start_calibration(self) -> None:
        """Clicks in the headphones; clap or sing "ta" on each. The measured delay becomes the default latency
        for the current output device."""
        if self._runner is not None or self._replay is not None or self._calibration is not None:
            return
        try:
            kwargs = {"player_factory": self._calibration_player_factory} if self._calibration_player_factory else {}
            self._calibration = CalibrationRun(self._input_factory(self.device_box.currentData()), **kwargs)
            self._calibration.start()
        except Exception as exc:
            self._calibration = None
            self.summary_label.setText(f"Calibration could not start: {exc}")
            return
        for w in (self.button, self.replay_button, self.calibrate_button):
            w.setEnabled(False)
        self.summary_label.setText("Calibrating: clap or sing 'ta' exactly on each click you hear (8 clicks)...")
        QtCore.QTimer.singleShot(int(1000 * self._calibration.duration_s) + 300, self.finish_calibration)

    def finish_calibration(self) -> None:
        if self._calibration is None:
            return
        run, self._calibration = self._calibration, None
        result = run.finish()
        for w in (self.button, self.replay_button, self.calibrate_button):
            w.setEnabled(True)
        output, _ = self._output_info()
        if result.latency_s is None:
            self.summary_label.setText(f"Calibration failed: heard {result.matched} of 8 answers. Try again, louder.")
            return
        self._settings.set_latency(output, result.latency_s)
        self._settings.save()
        self.summary_label.setText(
            f"Latency for {output or DEFAULT_OUTPUT}: {1000 * result.latency_s:.0f} ms "
            f"({result.matched}/8 answers, spread {1000 * result.spread_s:.0f} ms). Saved as the default.")

    def _draw_melody(self, t_end: float) -> None:
        """The reference line on the voice clock (song time + latency), over the visible window."""
        if self._melody is None:
            return
        latency = self._melody_latency
        if self._runner is not None and self._runner.latency_s is not None:
            latency = self._runner.latency_s
        t, midi = self._melody.midi_line()
        x = t + latency
        keep = (x >= t_end - WINDOW_S) & (x <= t_end + 1.0)
        self._ref_curve.setData(x[keep], midi[keep])
        spans = [(a + latency, b + latency) for a, b in self._melody.unscored_spans()
                 if b + latency >= t_end - WINDOW_S and a + latency <= t_end + 1.0]
        self._unscored.setOpts(x0=[a for a, _ in spans], x1=[b for _, b in spans],
                               y0=[0.0] * len(spans), y1=[127.0] * len(spans))

    def tick(self, drain_only: bool = False) -> None:
        if self._runner is not None:
            try:
                self.ingest(self._runner.drain())
            except Exception:           # the worker thread died: stop() ends the Take and reports why
                self._stop()
                return
            b = self._backing
            if b is not None and not drain_only and getattr(b, "finished", False) \
                    and b.position_s() > b.duration_s + END_GRACE_S:
                self._stop()            # the song is over (plus time for the late voice to arrive)
                return
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
        self._draw_melody(t[-1])
        if self._replay is not None and self._replay.finished:
            self._end_replay()
            self.summary_label.setText("Replay finished.")

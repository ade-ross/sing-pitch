"""Live view: pitch line on a note grid coloured by Band, current note + cents, Take summary on Stop."""
from __future__ import annotations

from collections import deque
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..audio_input import AudioInput, SoundDeviceInput, list_inputs, pick_input
from ..config import SAMPLE_RATE, data_dir
from ..calibration import CalibrationRun
from ..library import Library, PlayerSettings, player_settings
from ..live import LiveRunner, Playback
from ..music import NOTE_NAMES, note_name
from ..pitch import PitchTracker
from ..playback import BackingPlayer, DronePlayer
from ..playback.transport import LEAD_IN_S, Loop
from ..playback.stretch import SpeedCache, Stretcher, time_stretch
from ..playback.outputs import OutputKind, default_output_name, headphone_warning, output_kind
from ..reference import Band, Key, Melody, NearestNote, Reference
from ..reference.melody import NotesMelody
from ..reference.notes import PER_PHRASE, format_length
from ..replay import Replay
from ..settings import DEFAULT_OUTPUT, Settings
from ..take_engine import (AUTO, FrameScore, LinearClock, PauseFold, SampledClock, SongClock, TakeRecord, pinned,
                           project_song_time, replay_scores)
from ..take_store import (last_take_id, load_frames, saved_clock_times, saved_latency, saved_octave,
                          saved_reference, saved_song_times)
from .timeline import Timeline
from .speed_prep import SpeedPreparer

PAST_S = 4.0          # the plot is centred on now: this much of the voice and the Reference behind the now-line...
FUTURE_S = 4.0        # ...and this much of the Reference ahead of it
TICK_MS = 33          # ~30 Hz redraw
SPEED_MIN, SPEED_MAX, SPEED_STEP = 50, 120, 5     # Speed in percent of the original
SPEED_DEBOUNCE_MS = 400   # +/- pressed several times: stretch only the Speed he stops on
END_GRACE_S = 1.0     # Melody: keep listening this long after the Backing ends
JUMP_S = 5.0          # the arrow keys jump this far
DRAW_STEP_S = 0.01    # the reference line is drawn at this resolution on the voice clock
HALF_RANGE = 8        # semitones shown above and below the centre
BAND_COLOUR = {Band.GREEN: "#2ecc71", Band.YELLOW: "#f1c40f", Band.RED: "#e74c3c"}
UNSCORED_COLOUR = "#8a8a8a"   # sung where the Reference has nothing reliable: shown, never scored
BRUSH = {b: pg.mkBrush(c) for b, c in BAND_COLOUR.items()}
BRUSH[None] = pg.mkBrush(UNSCORED_COLOUR)
UNSCORED_SPAN_BRUSH = pg.mkBrush(138, 138, 138, 70)   # Unscored stretch of the Melody, greyed on the plot
REFERENCE_PEN = pg.mkPen("#5dade2", width=4)
GUIDE_PEN = pg.mkPen(93, 173, 226, 70, width=2)     # the Reference an octave above and below: fainter than the blue line
NOTE_LABEL_COLOR = (235, 245, 255)                  # Notes mode: the note name over each bar
NOTE_GAP_SAMPLES = 2                                # Notes mode: blank draw samples at the end of a bar, so bars stay apart
NOTE_SLIDER_PHRASE = 41                             # the Minimum note length slider: 1..40 tenths of a second, 41 per Phrase


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
        mode = f" ({s.octave_mode})" if s.octave_mode else ""
        notes = f"notes {format_length(record.min_note_s)}   |   " if record.note_mode == "on" else ""
        key = f"{record.song_id}   |   {notes}octave shift {s.octave_shift:+d}{mode}   |   "
    return (
        f"{key}In tune {s.pct_in_tune:.0f}%   |   median {s.median_abs_cents:.0f} cents off   |   "
        f"tends {s.tendency}   |   {steadiness_text(s)}   |   range {s.lowest_note}-{s.highest_note}   |   "
        f"{s.n_scored * 0.016:.1f} s sung   |   saved as {record.id}"
    )


def pass_text(record: TakeRecord) -> str:
    """One Pass's result, shown while the loop goes on."""
    s = record.summary
    if s.n_scored == 0:
        return f"Pass {record.pass_index} of {record.section_name}: nothing sung in the Section."
    return (f"Pass {record.pass_index} of {record.section_name}: In tune {s.pct_in_tune:.0f}%   |   "
            f"median {s.median_abs_cents:.0f} cents off   |   tends {s.tendency}   |   saved as {record.id}")


def length_of_slider(value: int) -> float:
    """The Minimum note length a slider position stands for: tenths of a second, the far end one note per Phrase."""
    return PER_PHRASE if value >= NOTE_SLIDER_PHRASE else value / 10


def slider_of_length(min_note_s: float) -> int:
    return NOTE_SLIDER_PHRASE if not np.isfinite(min_note_s) else min(40, max(1, round(10 * min_note_s)))


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
        stretcher: Stretcher = time_stretch,
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
        self._speed_ready: tuple[str, int, tuple] | None = None   # (song id, percent, stems) loaded for the chosen Speed
        self._speed_job: tuple[int, str, int] | None = None       # (job, song id, percent) being prepared
        self._speed_playing = 100                                 # the percent the player is set to (or will start at)
        home = self._home     # not self in the lambda: window <-> preparer would be a cycle, freed by gc on any thread
        self._preparer = SpeedPreparer(lambda song_id: SpeedCache(Library(home).get(song_id).dir, SAMPLE_RATE,
                                                                  stretcher), self)
        self._preparer.done.connect(self._speed_prepared)
        self._speed_timer = QtCore.QTimer(self)
        self._speed_timer.setSingleShot(True)
        self._speed_timer.setInterval(SPEED_DEBOUNCE_MS)
        self._speed_timer.timeout.connect(self._prepare_speed)
        self._settings = Settings.load(self._home)
        self._melody: Melody | None = None           # drawn beside the live line during Melody Takes and replays
        self._melody_latency = 0.0
        self._replay_clock: SongClock | None = None  # the replayed Take's own song times, for drawing the reference
        self._paused = False
        self._runner: LiveRunner | None = None
        self._replay: Replay | None = None
        self._replay_started = 0.0
        self._t: deque[float] = deque()           # voice time of each drawn frame (the plot shows view time, see _fold)
        self._midi: deque[float] = deque()
        self._band: deque[Band | None] = deque()
        self._out: deque[bool] = deque()          # True = sung note outside the Key
        self._oct: deque[int] = deque()           # the Octave shift each frame was scored at
        self._oct_known = False                   # a scored frame has set it (until then the line sits at 0)
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
        self.song_box.currentIndexChanged.connect(self._restore_player_settings)   # first: Speed and Section read them
        self.song_box.currentIndexChanged.connect(self._speed_changed)
        self.guide_box = QtWidgets.QCheckBox("Guide vocal")
        self.guide_slider = _volume_slider("Guide vocal volume")
        self.backing_slider = _volume_slider("Backing volume")
        for w in (self.guide_box, self.guide_slider, self.backing_slider):
            (w.toggled if w is self.guide_box else w.valueChanged).connect(self._apply_gains)
            (w.toggled if w is self.guide_box else w.valueChanged).connect(self._remember_player_settings)
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.pause_button.setToolTip("Space. Left/Right arrows jump 5 s; click the timeline to jump anywhere.")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.timeline = Timeline()
        self.timeline.seek_requested.connect(self.seek)
        for key, action in ((QtCore.Qt.Key_Space, self.toggle_pause), (QtCore.Qt.Key_Left, self.jump_back),
                            (QtCore.Qt.Key_Right, self.jump_forward)):    # bound methods: a lambda keeps self alive
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.setContext(QtCore.Qt.WindowShortcut)
            shortcut.activated.connect(action)
        self.timeline.section_marked.connect(self._section_marked)
        self.timeline.section_changed.connect(self._section_edited)
        self.section_box = QtWidgets.QComboBox()
        self.section_box.setFocusPolicy(QtCore.Qt.NoFocus)
        self.section_box.setToolTip("Drag across the timeline to mark a Section; drag its edges to adjust it")
        self.section_box.currentIndexChanged.connect(self._section_chosen)
        self.section_name_edit = QtWidgets.QLineEdit()
        self.section_name_edit.setPlaceholderText("Section name")
        self.section_name_edit.setMaximumWidth(140)
        self.save_section_button = QtWidgets.QPushButton("Save")
        self.save_section_button.clicked.connect(self.save_section)
        self.rename_section_button = QtWidgets.QPushButton("Rename")
        self.rename_section_button.clicked.connect(self.rename_section)
        self.delete_section_button = QtWidgets.QPushButton("Delete")
        self.delete_section_button.clicked.connect(self.delete_section)
        self.loop_box = QtWidgets.QCheckBox("Loop (L)")
        self.loop_box.setFocusPolicy(QtCore.Qt.NoFocus)
        self.loop_box.setToolTip(f"Repeat the Section; each Pass starts {LEAD_IN_S:.0f} s early (never scored) and is "
                                 "saved as its own Take")
        self.loop_box.toggled.connect(self._loop_toggled)
        loop_key = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_L), self)
        loop_key.setContext(QtCore.Qt.WindowShortcut)
        loop_key.activated.connect(self.toggle_loop)                  # bound method: a lambda keeps self alive
        self.speed_box = QtWidgets.QSpinBox()
        self.speed_box.setRange(SPEED_MIN, SPEED_MAX)
        self.speed_box.setSingleStep(SPEED_STEP)
        self.speed_box.setValue(100)
        self.speed_box.setSuffix(" %")
        self.speed_box.setFocusPolicy(QtCore.Qt.NoFocus)         # so + and - stay shortcuts of the window
        self.speed_box.setToolTip("Speed, pitch unchanged (+ and - change it by 5 %)")
        self.speed_label = QtWidgets.QLabel("")
        self.octave_box = QtWidgets.QComboBox()
        for k in (None, *player_settings.OCTAVE_CHOICES):      # Auto, then the pinned shifts
            self.octave_box.addItem("Auto" if k is None else f"{k:+d}" if k else "0", k)
        self.octave_box.setFocusPolicy(QtCore.Qt.NoFocus)
        self.octave_box.setToolTip("Octave shift: Auto picks the octave you sing in for each Phrase; a number pins it "
                                   "for the whole Take. Faint lines show the melody one octave above and below.")
        self.octave_box.currentIndexChanged.connect(self._remember_player_settings)
        self.notes_box = QtWidgets.QCheckBox("Notes")
        self.notes_box.setFocusPolicy(QtCore.Qt.NoFocus)
        self.notes_box.setToolTip("Notes mode: show and score the melody as flat notes on exact semitones")
        self.note_length_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.note_length_slider.setRange(1, NOTE_SLIDER_PHRASE)
        self.note_length_slider.setValue(slider_of_length(player_settings.DEFAULT_MIN_NOTE_S))
        self.note_length_slider.setFixedWidth(120)
        self.note_length_slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.note_length_slider.setToolTip("Minimum note length: melody movements shorter than this merge into the "
                                           "neighbouring note; far right = one note per Phrase")
        self.note_length_label = QtWidgets.QLabel(format_length(player_settings.DEFAULT_MIN_NOTE_S))
        self.note_length_label.setMinimumWidth(70)
        self.notes_box.toggled.connect(self._notes_changed)                # bound methods: a lambda keeps self alive
        self.note_length_slider.valueChanged.connect(self._note_length_moved)
        self.note_length_slider.sliderReleased.connect(self._notes_changed)
        self.speed_box.valueChanged.connect(self._speed_changed)
        self.speed_box.valueChanged.connect(self._remember_player_settings)
        for key, slot in (("+", self.speed_up), ("=", self.speed_up), ("-", self.speed_down)):
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=slot)    # bound method: a lambda keeps self alive
        self.calibrate_button = QtWidgets.QPushButton("Calibrate latency")
        self.calibrate_button.clicked.connect(self.start_calibration)
        self.output_label = QtWidgets.QLabel("")
        self.output_label.setWordWrap(True)
        self.output_label.setStyleSheet("color: #e67e22;")
        self.headphones_button = QtWidgets.QPushButton("These are headphones")
        self.headphones_button.clicked.connect(self.confirm_headphones)
        self.reference_box.currentTextChanged.connect(self._reference_changed)
        self.song_box.currentIndexChanged.connect(self._preview_song)
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
        self._guide_curves = [plot.plot(pen=GUIDE_PEN, connect="finite") for _ in (12, -12)]   # Guide lines
        self._ref_curve = plot.plot(pen=REFERENCE_PEN, connect="finite")
        self._note_labels: list[pg.TextItem] = []      # Notes mode: a pool of note-name labels, reused every frame
        self._now_line = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen(255, 255, 255, 110, width=1))
        self._now_line.setZValue(5)
        plot.addItem(self._now_line)
        self._curve = plot.plot(pen=pg.mkPen("#777777", width=1), connect="finite")
        self._dots = pg.ScatterPlotItem(size=6, pen=None)
        plot.addItem(self._dots)
        self._plot = plot
        self._set_y(self._centre)
        self._set_now(0.0)

        top = QtWidgets.QHBoxLayout()
        for w in (QtWidgets.QLabel("Input"), self.device_box, QtWidgets.QLabel("Reference"), self.reference_box,
                  self.key_box, self.drone_box, self.song_box, self.button, self.replay_button, self.history_button,
                  self.calibrate_button):
            top.addWidget(w)
        top.addStretch(1)
        top.addWidget(self.note_label)
        mix = QtWidgets.QHBoxLayout()
        for w in (self.guide_box, self.guide_slider, QtWidgets.QLabel("Backing"), self.backing_slider,
                  self.pause_button, QtWidgets.QLabel("Speed"), self.speed_box, self.speed_label, QtWidgets.QLabel("Octave"),
                  self.octave_box, self.notes_box, self.note_length_slider, self.note_length_label):
            mix.addWidget(w)
        mix.addStretch(1)
        sections = QtWidgets.QHBoxLayout()
        for w in (QtWidgets.QLabel("Section"), self.section_box, self.section_name_edit, self.save_section_button,
                  self.rename_section_button, self.delete_section_button, self.loop_box):
            sections.addWidget(w)
        sections.addStretch(1)
        warn = QtWidgets.QHBoxLayout()
        warn.addWidget(self.output_label, 1)
        warn.addWidget(self.headphones_button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(mix)
        layout.addLayout(sections)
        layout.addLayout(warn)
        layout.addWidget(self.plot_widget, 1)
        layout.addWidget(self.timeline)
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
        self._restore_player_settings()
        self._preview_song()
        for w in (self.guide_box, self.guide_slider, self.backing_slider, self.speed_box, self.octave_box,
                  self.notes_box, self.note_length_slider, self.section_box, self.section_name_edit, self.save_section_button, self.rename_section_button,
                  self.delete_section_button, self.loop_box):
            w.setEnabled(is_melody)
        self.update_output_warning()
        self._speed_changed()

    def _preview_song(self, *_) -> None:
        """Idle with Melody chosen: the Timeline shows the chosen song's reference melody. Not while a Take runs."""
        if self._runner is not None or self._replay is not None:
            return
        melody = None
        if self.reference_box.currentText() == "Melody" and self.song_box.currentData() is not None:
            try:
                melody = self._in_notes_mode(Melody.from_song(Library(self._home).get(self.song_box.currentData())))
            except Exception:       # a song that cannot be read: Start says why
                melody = None
        self.timeline.set_melody(melody)
        self.timeline.setVisible(melody is not None)
        self.reload_sections()

    def seek(self, song_s: float) -> None:
        """Jump the running Melody Take's song to song_s (a click on the Timeline). A jump closes the Take and ends a
        loop."""
        if self._backing is None:
            return
        self._backing.seek(song_s)
        self._set_loop_box(False)
        self.timeline.set_playhead(song_s)

    # --- Sections and the loop ---------------------------------------------------------------------------------
    def _sections(self):
        """The chosen song's Section store, or None (no Melody song chosen)."""
        song_id = self._speed_song()
        return None if song_id is None else Library(self._home).sections(song_id)

    def reload_sections(self, select: str | None = None) -> None:
        """Fill the picker from the library: the unnamed mark first, then the saved Sections; show the chosen one
        (else the one the song was last left on)."""
        store = self._sections()
        if select is None and (player := self._player_store()) is not None:
            select = player.load().section
        self.section_box.blockSignals(True)
        self.section_box.clear()
        self.section_box.addItem("unsaved mark", None)
        for section in (store.saved() if store else []):
            self.section_box.addItem(section.name, section.name)
        if select is not None:
            self.section_box.setCurrentIndex(max(0, self.section_box.findData(select)))
        self.section_box.blockSignals(False)
        self._section_chosen()

    def _chosen_section(self):
        store = self._sections()
        if store is None:
            return None
        name = self.section_box.currentData()
        try:
            return store.mark if name is None else store.get(name)
        except StopIteration:
            return None

    def _section_chosen(self, *_) -> None:
        section = self._chosen_section()
        self.timeline.set_section(*((None,) if section is None else (section.start_s, section.end_s)))
        if self.section_box.currentData() is not None:
            self.section_name_edit.setText(self.section_box.currentData())
        self._remember_player_settings()
        self._apply_loop()

    def _section_marked(self, start_s: float, end_s: float) -> None:
        """A new drag on the strip: it becomes the unnamed mark (kept until replaced)."""
        store = self._sections()
        if store is None:
            return
        store.set_mark(start_s, end_s)
        self.timeline.set_section(start_s, end_s)
        if self.section_box.currentIndex() != 0:
            self.section_box.setCurrentIndex(0)            # shows the mark and re-applies the loop
        else:
            self._apply_loop()

    def _section_edited(self, start_s: float, end_s: float) -> None:
        """Edges moved: a saved Section keeps its name and takes the new span, otherwise the mark does."""
        store, name = self._sections(), self.section_box.currentData()
        if store is None:
            return
        if name is None:
            store.set_mark(start_s, end_s)
        else:
            store.save(name, start_s, end_s)
        self._apply_loop()

    def save_section(self) -> None:
        store, span, name = self._sections(), self.timeline.section, self.section_name_edit.text().strip()
        if store is None or span is None or not name:
            self.summary_label.setText("Mark a Section on the timeline and type its name to save it.")
            return
        store.save(name, *span)
        self.reload_sections(select=name)

    def rename_section(self) -> None:
        store, old, new = self._sections(), self.section_box.currentData(), self.section_name_edit.text().strip()
        if store is None or old is None:
            self.summary_label.setText("Pick a saved Section to rename.")
            return
        try:
            store.rename(old, new)
        except ValueError as exc:
            self.summary_label.setText(f"Cannot rename: {exc}")
            return
        self.reload_sections(select=new)

    def delete_section(self) -> None:
        store, name = self._sections(), self.section_box.currentData()
        if store is None or name is None:
            return
        store.delete(name)
        self.reload_sections()

    def toggle_loop(self) -> None:
        if self.loop_box.isEnabled():
            self.loop_box.setChecked(not self.loop_box.isChecked())

    def _loop(self) -> Loop | None:
        """The loop the controls ask for: the shown Section, named when it is a saved one."""
        span = self.timeline.section
        if not self.loop_box.isChecked() or span is None:
            return None
        return Loop(span[0], span[1], self.section_box.currentData(), LEAD_IN_S)

    def _loop_toggled(self, on: bool) -> None:
        if on and self.timeline.section is None:
            self._set_loop_box(False)
            self.summary_label.setText("Mark a Section on the timeline first (drag across it), then loop it.")
            return
        self._apply_loop()

    def _apply_loop(self) -> None:
        """The running Take's player loops what the controls show (a changed Section starts again at Pass 1)."""
        if self._backing is not None and hasattr(self._backing, "set_loop"):
            loop = self._loop()
            if loop != getattr(self._backing, "loop", None):
                self._backing.set_loop(loop)

    def _set_loop_box(self, on: bool) -> None:
        self.loop_box.blockSignals(True)
        self.loop_box.setChecked(on)
        self.loop_box.blockSignals(False)

    def jump_back(self) -> None:
        self.jump_by(-JUMP_S)

    def jump_forward(self) -> None:
        self.jump_by(JUMP_S)

    def jump_by(self, seconds: float) -> None:
        if self._backing is not None:
            self.seek(max(0.0, self._backing.position_s() + seconds))

    def toggle_pause(self) -> None:
        """Space: pause or resume the song of the running Melody Take. The Take goes on; the plot stands still while
        the song is paused (singing then is neither drawn nor scored) and goes on from the same place on resume."""
        if self._backing is None:
            return
        self._paused = not self._paused
        (self._backing.pause if self._paused else self._backing.play)()
        self.pause_button.setText("Play" if self._paused else "Pause")

    def _apply_gains(self, *_) -> None:
        """Slider and checkbox changes reach the player at once, also mid-Take; it ramps them within a block."""
        if self._backing is not None:
            self._backing.set_gains(*self._gains())

    # --- per-song player settings -------------------------------------------------------------------------------
    def _player_store(self):
        """The chosen song's player settings (Melody with a song chosen), else None."""
        song_id = self._speed_song()
        return None if song_id is None else Library(self._home).player_settings(song_id)

    def _remember_player_settings(self, *_) -> None:
        """Keep the controls as the chosen song's settings; the song reopens like this."""
        store = self._player_store()
        if store is not None:
            store.save(PlayerSettings(self.guide_box.isChecked(), self.guide_slider.value() / 100,
                                      self.backing_slider.value() / 100, self.speed_box.value(),
                                      self.section_box.currentData(), self.octave_box.currentData(),
                                      self.notes_box.isChecked(), self._note_length()))

    def _restore_player_settings(self, *_) -> None:
        """Put the chosen song's saved Guide vocal, volumes and Speed on the controls (the Section is picked by
        reload_sections). Not while a Take runs."""
        store = self._player_store()
        if store is None or self._runner is not None:
            return
        saved = store.load()
        for widget, value in ((self.guide_box, saved.guide_on), (self.guide_slider, round(100 * saved.guide_gain)),
                              (self.backing_slider, round(100 * saved.backing_gain)),
                              (self.speed_box, saved.speed_pct)):
            widget.blockSignals(True)
            (widget.setChecked if widget is self.guide_box else widget.setValue)(value)
            widget.blockSignals(False)
        self.octave_box.blockSignals(True)
        self.octave_box.setCurrentIndex(max(0, self.octave_box.findData(saved.octave)) if saved.octave is not None
                                        else 0)
        self.octave_box.blockSignals(False)
        for widget in (self.notes_box, self.note_length_slider):
            widget.blockSignals(True)
        self.notes_box.setChecked(saved.notes)
        self.note_length_slider.setValue(slider_of_length(saved.min_note_s))
        for widget in (self.notes_box, self.note_length_slider):
            widget.blockSignals(False)
        self.note_length_label.setText(format_length(self._note_length()))

    # --- Notes mode -------------------------------------------------------------------------------------------------
    def _note_length(self) -> float:
        return length_of_slider(self.note_length_slider.value())

    def _in_notes_mode(self, melody: Melody) -> Melody:
        """The Melody as the controls want it: its Notes-mode bars when Notes is ticked."""
        return melody.notes(self._note_length()) if self.notes_box.isChecked() else melody

    def _note_length_moved(self, *_) -> None:
        """The slider moved: the label follows at once; while it is being dragged the song is reloaded only on release."""
        self.note_length_label.setText(format_length(self._note_length()))
        if not self.note_length_slider.isSliderDown():
            self._notes_changed()

    def _notes_changed(self, *_) -> None:
        self.note_length_label.setText(format_length(self._note_length()))
        self._remember_player_settings()
        self._preview_song()

    def _octave_setting(self):
        """Auto, or the box's pinned shift; only a Melody has an original singer to be an octave away from."""
        k = self.octave_box.currentData()
        return AUTO if k is None or self.reference_box.currentText() != "Melody" else pinned(k)

    # --- Speed -------------------------------------------------------------------------------------------------
    def speed_up(self) -> None:
        self._step_speed(1)

    def speed_down(self) -> None:
        self._step_speed(-1)

    def _step_speed(self, direction: int) -> None:
        if self.speed_box.isEnabled():
            self.speed_box.setValue(self.speed_box.value() + direction * SPEED_STEP)      # stays within 50-120

    def _speed_song(self) -> str | None:
        return self.song_box.currentData() if self.reference_box.currentText() == "Melody" else None

    def _speed_changed(self, *_) -> None:
        """The chosen Speed or song changed (or the Reference did). 100 % and Speeds whose stems are already loaded
        apply at once; a Speed cached on disk loads in the background; a new one is stretched in the background
        after a short pause (so a run of +/- presses prepares only the Speed he stops on) while the player keeps its
        old Speed. The message says what is going on."""
        song_id, percent = self._speed_song(), self.speed_box.value()
        self._speed_timer.stop()
        self._speed_job = None
        if song_id is None:
            self.speed_label.setText("")
        elif percent == 100:
            self._speed_ready = None
            self._set_player_speed(100, None)
        elif self._speed_ready and self._speed_ready[:2] == (song_id, percent):
            self._set_player_speed(percent, self._speed_ready[2])
        elif self._preparer.is_cached(song_id, percent):
            self._prepare_speed()
        else:
            self.speed_label.setText(f"preparing {percent} %…")
            self._speed_timer.start()

    def _prepare_speed(self) -> None:
        song_id, percent = self._speed_song(), self.speed_box.value()
        if song_id is not None and percent != 100:
            self._speed_job = (self._preparer.start(song_id, percent), song_id, percent)

    def _speed_prepared(self, job: int, stems, error) -> None:
        if self._speed_job is None or self._speed_job[0] != job:
            return                                        # superseded: its result is cached on disk for next time
        _, song_id, percent = self._speed_job
        self._speed_job = None
        if error is not None:
            self.speed_label.setText(f"Speed {percent} % failed: {error}")
            self.speed_box.blockSignals(True)
            self.speed_box.setValue(self._speed_playing)
            self.speed_box.blockSignals(False)
            self._remember_player_settings()          # the failed Speed is not kept as the song's Speed
            return
        self._speed_ready = (song_id, percent, stems)
        self._set_player_speed(percent, stems)

    def _set_player_speed(self, percent: int, stems) -> None:
        """Hand the player this Speed. The Speed it already plays is not handed again: a speed_changed would split
        the running Take for nothing (during a Take, one song and percent always mean the same stems)."""
        self.speed_label.setText("")
        if self._backing is not None and percent == self._speed_playing:
            return
        self._speed_playing = percent
        if self._backing is not None:
            self._backing.set_speed(percent / 100, *(stems or ()))

    def _stems_for_start(self, song_id: str, percent: int):
        """The stretched stems a Take at `percent` starts with, or None while they are still being prepared."""
        if self._speed_ready and self._speed_ready[:2] == (song_id, percent):
            return self._speed_ready[2]
        if self._preparer.is_cached(song_id, percent):
            stems = self._preparer.load_now(song_id, percent)
            self._speed_ready = (song_id, percent, stems)
            return stems
        if self._speed_job is None and not self._speed_timer.isActive():
            self._speed_changed()
        return None

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
            return self._in_notes_mode(Melody.from_song(Library(self._home).get(song_id)))
        return NearestNote()

    def _set_now(self, now: float) -> None:
        """Centre the plot on voice time `now`: the now-line in the middle, PAST_S behind it and FUTURE_S ahead."""
        self._now_line.setValue(now)
        self._plot.setXRange(now - PAST_S, now + FUTURE_S, padding=0)

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
            frames = load_frames(self._home, take_id)
            scores = replay_scores(frames, reference, song_times=song_times, octave=saved_octave(self._home, take_id))
            replay = Replay(scores)
            octave_k = next((s.octave for s in scores if s.band is not None), 0)      # where the Take was scored
            replay_clock = SampledClock([f.t for f in frames], saved_clock_times(self._home, take_id),   # + Lead-in
                                        getattr(reference, "end_s", None))
        except Exception as exc:
            self.summary_label.setText(f"Cannot replay Take {take_id}: {exc}")
            return
        self._replay, self._replay_clock = replay, replay_clock
        self._replay_started = time.monotonic()
        self._show_melody(reference, latency, octave_k)
        self._t.clear(), self._midi.clear(), self._band.clear(), self._out.clear(), self._oct.clear()
        self._oct_known = False
        self._set_now(0.0)
        self.replay_button.setText("Stop replay")
        self.button.setEnabled(False)
        for w in (self.notes_box, self.note_length_slider):       # a replay is drawn as it was scored: they change nothing
            w.setEnabled(False)
        self.summary_label.setText(f"Replaying Take {take_id}...")
        self._timer.start(TICK_MS)

    def _end_replay(self) -> None:
        self._timer.stop()
        self._replay = self._replay_clock = None
        self.replay_button.setText("Replay last Take")
        self.button.setEnabled(True)
        for w in (self.notes_box, self.note_length_slider):
            w.setEnabled(self.reference_box.currentText() == "Melody")
        self._preview_song()                                      # the Timeline as the controls now want it

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

    def _show_melody(self, reference: Reference, latency_s: float, octave_k: int = 0) -> None:
        self._melody = reference if reference.kind == "melody" else None
        self._melody_latency = latency_s
        for curve in (self._ref_curve, *self._guide_curves):
            curve.setData([], [])
        self._show_labels([])
        self._unscored.setOpts(x0=[], x1=[], y0=[], y1=[])
        self.timeline.set_melody(self._melody, getattr(self._backing, "duration_s", None))
        self.timeline.setVisible(self._melody is not None)
        if self._melody is not None:
            _, midi = self._melody.midi_line()
            if np.isfinite(midi).any():
                self._set_y(float(np.nanmedian(midi)) + 12 * octave_k)    # a pinned Octave: look where he sings

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
                percent = self.speed_box.value()
                stems = None if percent == 100 else self._stems_for_start(reference.song_id, percent)
                if percent != 100 and stems is None:
                    self.summary_label.setText(f"Cannot start yet: preparing {percent} %…")
                    return
                song = Library(self._home).get(reference.song_id)
                backing = self._backing_factory(song.backing_path, song.vocals_path)
                backing.set_gains(*self._gains())
                if self._loop() is not None:                 # the Take starts on the Section's Lead-in, as Pass 1
                    backing.set_loop(self._loop())
                if stems is not None:
                    backing.set_speed(percent / 100, *stems)
                self._speed_playing = percent
            self._runner = LiveRunner(self._input_factory(index), self._tracker, reference, self._home,
                                      backing=backing, latency_s=latency, octave=self._octave_setting())
            self._runner.start()
        except Exception as exc:
            self._runner = None
            self.summary_label.setText(f"Could not open the input or the output: {exc}")
            return
        self._backing, self._paused = backing, False
        self._show_melody(reference, latency, self._octave_setting().k)
        self.pause_button.setText("Pause")
        self.pause_button.setEnabled(backing is not None)
        self._t.clear(), self._midi.clear(), self._band.clear(), self._out.clear(), self._oct.clear()
        self._oct_known = False
        self._set_now(0.0)
        self.button.setText("Stop")
        for w in (self.device_box, self.reference_box, self.key_box, self.drone_box, self.song_box,
                  self.octave_box, self.notes_box, self.note_length_slider, self.calibrate_button):
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
        self.pause_button.setEnabled(False)
        self.button.setText("Start")
        self.device_box.setEnabled(True)
        self.reference_box.setEnabled(True)
        self.calibrate_button.setEnabled(True)
        self._reference_changed()
        self.replay_button.setEnabled(True)
        self.summary_label.setText(describe(record) if record else f"The Take failed: {error}")

    def ingest(self, scores: list[FrameScore]) -> None:
        fold = self._fold()
        scores = [s for s in scores if not fold.paused(s.t)]     # sung while the song was paused: not drawn
        for s in scores:
            self._t.append(s.t)
            self._midi.append(np.nan if s.midi is None else s.midi)
            self._band.append(s.band)
            self._out.append(s.in_key is False)
            if not self._oct_known and s.band is not None:      # the first scored frame: the Phrase's octave is known,
                self._oct_known = True                           # so the silence before it is drawn at it too
                self._oct = deque(s.octave for _ in self._oct)
            self._oct.append(s.octave)
        if scores:
            end = fold.to_view(self._t[-1])
            while end - fold.to_view(self._t[0]) > PAST_S:
                self._t.popleft(), self._midi.popleft(), self._band.popleft(), self._out.popleft(), self._oct.popleft()
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

    def _draw_clock(self) -> SongClock:
        """Voice time -> song time for drawing: the running Take's clock (it knows every jump and pause), the replayed
        Take's own song times, else song time + latency."""
        if self._runner is not None and self._runner.song_clock is not None:
            return self._runner.song_clock
        return self._replay_clock or LinearClock(self._melody_latency)

    def _fold(self, clock: SongClock | None = None) -> PauseFold:
        """Voice time <-> the plot's view time: the draw clock's pauses folded away, so the plot stands still while the
        song is paused and resumes where it stopped (no pauses: the two are the same)."""
        spans = getattr(clock or self._draw_clock(), "paused_spans", None)
        return PauseFold(spans() if spans is not None else ())

    def _now_voice(self) -> float:
        """The voice time of the song heard now, where the now-line goes: the running Take's heard_t (the tracker's
        frames lag it, so the voice ends a little left of the line), a replay's own clock, else the newest frame. Never
        before the newest frame, so no voice is drawn right of the line."""
        newest = self._t[-1]
        if self._runner is not None:
            return max(newest, self._runner.heard_t)
        if self._replay is not None:
            return max(newest, self._replay.origin_s + time.monotonic() - self._replay_started)
        return newest

    def _octaves_at(self, x: np.ndarray, fold: PauseFold | None = None) -> np.ndarray:
        """The Octave shift in force at each view time x: that of the latest frame at or before it (the first frame's
        before them, the last frame's beyond them)."""
        if not self._t:
            return np.zeros(len(x))
        t = (fold or self._fold()).to_view(np.fromiter(self._t, float))
        octs = np.fromiter(self._oct, float)
        return octs[np.clip(np.searchsorted(t, x, side="right") - 1, 0, len(t) - 1)]

    def _song_times(self, x: np.ndarray, fold: PauseFold | None = None) -> np.ndarray:
        """The song time heard at each view time x of the plot (NaN = no song there). Up to what the Transport has
        rendered that is the song clock's own answer at the voice time shown there, so seeks, Speed changes and loop
        wraps show as they happened and pauses are folded away (see _fold). Beyond it (the right half, up to FUTURE_S
        ahead of now) nothing is known yet: the song is taken to go on from where the rendering ended at its Speed,
        wrapping at the end of a looped Section to the next Pass's Lead-in, and ending at the song end. A seek, pause,
        Speed change or Section change made inside those seconds is not foreseen (it shows once rendered); while paused
        the song goes on, on resume, from where the player waits."""
        clock = self._draw_clock()
        fold = fold or self._fold(clock)
        song = np.array([np.nan if (t := clock(v)) is None else t for v in fold.to_voice(x)])
        tail = getattr(clock, "tail", None)
        tail = tail() if tail is not None else None
        if tail is None:                                 # a replay's clock knows its whole Take, a linear one goes on
            return song
        x_end, song_end, rate = tail
        backing = self._backing
        if song_end is None:                             # paused: the song waits where the player stands
            if backing is None:
                return song
            song_end, rate = backing.position_s(), getattr(backing, "speed", 1.0)
        x_end = fold.to_view(x_end)                      # paused: the fold the pause stands at
        ahead = x > x_end
        duration = getattr(backing, "duration_s", None)
        loop = getattr(backing, "loop", None)
        loop_end = None if loop is None else min(loop.end_s, duration if duration is not None else loop.end_s)
        song[ahead] = project_song_time(x[ahead] - x_end, song_end, rate, loop_end,
                                        0.0 if loop is None else loop.pass_start_s, duration)
        return song

    def _sample_reference(self, x: np.ndarray,
                          fold: PauseFold | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(song time, MIDI, unscored) of the Melody at each view time x of the plot, past and future halves alike:
        the one place that turns the plot's x-range into the Reference, for the line, the Guide lines, the Unscored
        bands and whatever else is drawn in the right half."""
        song = self._song_times(x, fold)
        midi, unscored = self._melody.sample(song)
        return song, midi, unscored

    def _draw_melody(self, t_end: float) -> None:
        """The reference line over the plot's [t_end - PAST_S, t_end + FUTURE_S] (view time, see _fold)."""
        if self._melody is None:
            return
        x = np.arange(t_end - PAST_S, t_end + FUTURE_S + DRAW_STEP_S, DRAW_STEP_S)
        fold = self._fold()
        song, midi, unscored = self._sample_reference(x, fold)
        octaves = self._octaves_at(x, fold)
        shift = 12.0 * octaves                           # drawn where the Take is scored, not at the singer's octave
        if isinstance(self._melody, NotesMelody):
            midi = self._note_bars(x, song, midi, octaves, shift)
        else:
            self._show_labels([])
        self._ref_curve.setData(x, midi + shift)
        for curve, d in zip(self._guide_curves, (12, -12)):
            curve.setData(x, midi + shift + d)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], unscored.astype(np.int8), [0]])))
        starts, ends = edges[::2], edges[1::2]
        self._unscored.setOpts(x0=x[starts], x1=x[ends - 1] + DRAW_STEP_S, y0=[0.0] * len(starts),
                               y1=[127.0] * len(starts))

    def _note_bars(self, x: np.ndarray, song: np.ndarray, midi: np.ndarray, octaves: np.ndarray,
                   shift: np.ndarray) -> np.ndarray:
        """Notes mode: the sampled notes as separate bars (the line blanked just before a bar that follows a different
        note, or a new Octave shift, so neighbouring bars never join into a staircase), each labelled with its note
        name at its start, the left edge of the plot for a bar already under way. An Unscored dip inside a note breaks
        its line but neither trims the bar before it nor labels the note again after it."""
        index = self._melody.note_index_at(song)
        index[~np.isfinite(midi)] = -1
        midi = midi.copy()
        on = np.flatnonzero(index >= 0)                  # the samples that belong to a note
        if not len(on):
            self._show_labels([])
            return midi
        different = (index[on][1:] != index[on][:-1]) | (octaves[on][1:] != octaves[on][:-1])    # vs the previous note sample
        for i, j in zip(on[1:][different], on[:-1][different]):
            if i == j + 1:                               # bars side by side: break the line (not across a dip, already NaN)
                midi[max(0, i - NOTE_GAP_SAMPLES):i] = np.nan
        starts = np.concatenate([[on[0]], on[1:][different]])
        notes = self._melody.note_list
        labels = [(x[i], notes[index[i]].midi + shift[i], note_name(int(round(notes[index[i]].midi + shift[i]))))
                  for i in starts]
        self._show_labels(labels)
        return midi

    def _show_labels(self, labels: list[tuple[float, float, str]]) -> None:
        """Put (x, MIDI, text) labels on the plot from the pool: items are created once and reused, the rest hidden."""
        while len(self._note_labels) < len(labels):
            item = pg.TextItem("", color=NOTE_LABEL_COLOR, anchor=(0, 1))
            item.setZValue(4)
            self._plot.addItem(item)
            self._note_labels.append(item)
        for item, (lx, ly, text) in zip(self._note_labels, labels):
            if item.toPlainText() != text:
                item.setText(text)
            item.setPos(lx, ly)
            if not item.isVisible():
                item.setVisible(True)
        for item in self._note_labels[len(labels):]:
            if item.isVisible():
                item.setVisible(False)

    def tick(self, drain_only: bool = False) -> None:
        if self._runner is not None:
            try:
                self.ingest(self._runner.drain())
            except Exception:           # the worker thread died: stop() ends the Take and reports why
                self._stop()
                return
            for record in self._runner.drain_saved():
                self.summary_label.setText(pass_text(record) if record.pass_index is not None
                                           else f"Jumped. Take saved: {describe(record)}")
            b = self._backing
            if b is not None and not drain_only and getattr(b, "finished", False) \
                    and b.position_s() > b.duration_s + END_GRACE_S:
                self._stop()            # the song is over (plus time for the late voice to arrive)
                return
        elif self._replay is not None:
            self.ingest(self._replay.advance(time.monotonic() - self._replay_started))
        if self._backing is not None and not drain_only:
            self.timeline.set_playhead(self._backing.position_s())
        if not self._t:
            return
        fold = self._fold()
        t, midi = fold.to_view(np.fromiter(self._t, float)), np.fromiter(self._midi, float)
        self._curve.setData(t, midi)
        voiced = ~np.isnan(midi)
        bands = [b for b, v in zip(self._band, voiced) if v]
        out = [o for o, v in zip(self._out, voiced) if v]
        self._dots.setData(t[voiced], midi[voiced], brush=[BRUSH[b] for b in bands],
                           symbol=["x" if o else "o" for o in out],   # out-of-key notes are drawn as big crosses
                           size=[11 if o else 6 for o in out])
        now = fold.to_view(self._now_voice())
        self._set_now(now)
        self._draw_melody(now)
        if self._replay is not None and self._replay.finished:
            self._end_replay()
            self.summary_label.setText("Replay finished.")

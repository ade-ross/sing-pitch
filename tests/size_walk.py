"""Walk the main window through every state it can reach and report each time its size, minimum size or minimum size
hint is not what it was at first show (#52: a window that resizes itself under a running Take breaks the sync, #49).
Used by test_window_size_is_constant.py (offscreen) and by hand on the real macOS style (QT_QPA_PLATFORM=cocoa)."""
from __future__ import annotations

import numpy as np
from audio import SR, quiet
from melody_helpers import library_song, reference_line
from PySide6 import QtWidgets

from sing_pitch.audio_input import ArrayInput
from sing_pitch.pitch.swiftf0 import SwiftF0Tracker
from sing_pitch.playback.outputs import output_devices
from sing_pitch.ui import MainWindow
from sing_pitch.ui.main_window import FIELD_HINT, PAUSED_TEXT

LONG_TITLE = "Ed Sheeran - Shape of You (Official Music Video) [Remastered 4K HD] feat. Someone Else"
LONG_SECTION = "The very long Section name that nobody would ever type but a YouTube-length one might be"
LONG_DEVICE = "Aggregate Device with a Very Long Name: Scarlett 18i20 USB Audio Interface (Input and Output 1-8)"
WORD = "Y" * 400            # no spaces at all: a word-wrapped label would ask for its whole width (#66)
LONG = "A very long message that goes on and on and on, much wider than any window could ever be " * 4
NOTES = [(-900, 0.6), (-500, 0.6), (-200, 0.8)]

MIC = {"index": 0, "name": LONG_DEVICE, "max_input_channels": 1, "max_output_channels": 0}
BUDS = {"index": 1, "name": "WF-1000XM5", "max_input_channels": 1, "max_output_channels": 2}
SPEAKERS = {"index": 2, "name": "MacBook Pro Speakers " + LONG_DEVICE, "max_input_channels": 0, "max_output_channels": 2}
UNKNOWN = {"index": 3, "name": "Mystery Box", "max_input_channels": 0, "max_output_channels": 2}
DEVICES = [MIC, BUDS, SPEAKERS, UNKNOWN]


def kind_of(name):
    return "headphones" if name == "WF-1000XM5" else "unknown" if name == "Mystery Box" else "speakers"


class FakeBacking:
    duration_s, finished = 2.0, False

    def __init__(self, path=None, vocals=None, **kw):
        self.speed, self.paused = 1.0, False

    def pause(self):
        self.paused = True

    def play(self):
        self.paused = False

    def set_gains(self, *a): ...
    def set_speed(self, speed, *a):
        self.speed = speed
    def start(self): ...
    def stop(self): ...
    def position_s(self):
        return 0.0

    @property
    def song_log(self):
        from sing_pitch.playback import SongTimeLog

        log = SongTimeLog(SR)
        log.append(0, 0.0)
        return log


class FakePlayer:
    def __init__(self, audio, rate):
        pass

    def start(self): ...
    def position_s(self):
        return 256 / SR
    def stop(self): ...


def stretcher(audio, sr, speed):
    return audio


def build(home, title=LONG_TITLE, default_output=("Mystery Box", "unknown")) -> MainWindow:
    """A window with a long-titled song, long device names, every kind of output and fake audio."""
    library_song(home, reference_line(NOTES), title=title, seconds=2.0)
    library_song(home, reference_line(NOTES), title="Short", seconds=2.0)
    return MainWindow(SwiftF0Tracker(), home=home, input_factory=lambda _i: ArrayInput(quiet(3.0), SR, 256),
                      backing_factory=FakeBacking, calibration_player_factory=FakePlayer,
                      output_info=lambda: default_output,
                      input_devices=lambda: [d for d in DEVICES if d["max_input_channels"] > 0],
                      output_devices=lambda: output_devices(DEVICES), output_kind_of=kind_of)


def signature(win):
    return (win.size().toTuple(), win.minimumSize().toTuple(), win.minimumSizeHint().toTuple())


class Watch:
    """Records the window's size, minimum size and minimum size hint at first show and checks them at every step."""

    def __init__(self, win):
        self.win, self.steps, self.broken = win, 0, []
        QtWidgets.QApplication.processEvents()
        self.first = signature(win)

    def __call__(self, what):
        QtWidgets.QApplication.processEvents()
        self.steps += 1
        now = signature(self.win)
        if now != self.first:
            self.broken.append(f"{what}: {now} instead of {self.first} (size, minimumSize, minimumSizeHint)")
            self.first = now          # report each change once, not every step after it


def walk(win, check, take_id=None):
    """Every state: `check(what)` is called after each one."""
    w = win
    sections = w.section_box
    check("first show")

    # --- every Reference, Notes mode, every combo box with its longest text -----------------------------------------
    for ref in ("Nearest note", "Key", "Melody", "Key", "Nearest note", "Melody"):
        w.reference_box.setCurrentText(ref)
        check(f"Reference {ref}")
    for i in range(w.key_box.count()):
        w.key_box.setCurrentIndex(i)
        check(f"Key {w.key_box.currentText()}")
    for i in range(w.song_box.count()):
        w.song_box.setCurrentIndex(i)
        check(f"song {w.song_box.currentText()[:20]}")
    for i in range(w.device_box.count()):
        w.device_box.setCurrentIndex(i)
        check(f"input {w.device_box.currentText()[:20]}")
    for i in range(w.output_box.count()):     # speakers / unknown / headphones: the warning and its button come and go
        w.output_box.setCurrentIndex(i)
        check(f"output {w.output_box.currentText()[:20]} (warning '{w.output_label.text()[:20]}')")
    for ref in ("Key", "Nearest note", "Melody"):
        w.reference_box.setCurrentText(ref)
        check(f"Reference {ref} with an output warning")
    for text in (LONG, WORD, "Output is " + WORD + ", which looks like speakers."):
        w.output_label.setText(text)
        w.output_label.setVisible(True), w.headphones_button.setVisible(True)
        check(f"output note {text[:20]}")
    w.update_output_warning()
    w.refresh_devices()
    check("Refresh")
    w.drone_box.setChecked(True)
    check("Drone on")
    w.drone_box.setChecked(False)
    for notes in (True, False):
        w.notes_box.setChecked(notes)
        check(f"Notes {notes}")
    w.notes_box.setChecked(True)
    for v in range(w.note_length_slider.minimum(), w.note_length_slider.maximum() + 1, 9):
        w.note_length_slider.setValue(v)
        check(f"note length {w.note_length_label.text()}")
    w.note_length_slider.setValue(w.note_length_slider.maximum())
    check("note length Phrase")
    for v in range(w.tolerance_slider.minimum(), w.tolerance_slider.maximum() + 1, 20):
        w.tolerance_slider.setValue(v)
        check(f"Tolerance {w.tolerance_label.text()}")
    for k in range(w.octave_box.count()):
        w.octave_box.setCurrentIndex(k)
        check(f"Octave {w.octave_box.currentText()}")
    for k in (2, 0, 1):
        w.guide_box.setChecked(bool(k)), w.guide_slider.setValue(30 * k), w.backing_slider.setValue(40 * k)
        check("Guide vocal / Backing")

    # --- note read-out -----------------------------------------------------------------------------------------------
    for text in ("-", "A4  -1 c", "G#4  +39 c", "C#4  out of key", "D#4  out of key"):
        w.note_label.setText(text)
        check(f"note read-out {text}")

    # --- Speed and its preparation ----------------------------------------------------------------------------------
    for pct in (50, 70, 85, 120, 100):
        w.speed_box.setValue(pct)
        check(f"Speed {pct}")
        QtWidgets.QApplication.processEvents()
    for text in ("preparing 55 %…", "preparing 100 % … 100 %", f"Speed 55 % failed: {LONG}", ""):
        w.speed_label.setText(text)
        check(f"speed message {text[:20]}")
    for b in (w.speed_70_button, w.speed_85_button, w.speed_100_button):
        b.click()
        check(f"preset {b.text()}")
    for text in ("85 % from the next Pass", "100 % from the next Pass", "preparing 100 % … 100 % from the next Pass"):
        w.speed_label.setText(text)                      # in a loop the label says so while the old Speed plays (#72)
        check(f"speed message {text[:25]}")
    w.speed_label.setText("")

    # --- Sections: marks, names, collisions, loop, Count-in -----------------------------------------------------
    w.timeline.section_marked.emit(0.2, 0.9)
    check("mark")
    w.section_name_edit.setText(LONG_SECTION)
    check("long Section name typed")
    w.save_section_button.click()
    check("saved long Section")
    w.timeline.section_marked.emit(0.5, 1.5)
    check("second mark")
    w.section_name_edit.setText(LONG_SECTION)
    w.save_section_button.click()
    check("Save onto a taken name: Overwrite and Cancel appear")
    w.cancel_overwrite_button.click()
    check("Cancel")
    w.save_section_button.click()
    w.overwrite_button.click()
    check("Overwrite")
    w.section_name_edit.setText("B")
    w.rename_section_button.click()
    check("Rename")
    w.delete_section_button.click()
    check("Delete")
    w.timeline.section_marked.emit(0.3, 1.2)
    w.loop_box.setChecked(True)
    check("Loop on")
    for text in ("Loop · on", "Loop · Pass 12", "Loop · Pass 12 · last 100 %", LONG):
        w.loop_label.setText(text)
        check(f"loop label {text[:20]}")
    w.between_box.setCurrentText("Count-in")
    check("Count-in")
    for bpm in (10.0, 0.0, 500.0, 143.6, 72.0):
        w.bpm_box.setValue(bpm)
        check(f"BPM {bpm}: {w.tempo_label.text()[:30]}")
    for text in ("finding the tempo…", "tempo unknown — set BPM (Lead-in until then)", LONG):
        w.tempo_label.setText(text)
        check(f"tempo note {text[:20]}")
    w.beats_box.setValue(7)
    check("beats")
    w.clear_section()
    check("Clear")
    w.between_box.setCurrentText("Lead-in")
    w.loop_box.setChecked(False)

    # --- status messages ----------------------------------------------------------------------------------------------
    for text in ("Press Start and sing.", "Singing...", PAUSED_TEXT, FIELD_HINT, LONG, WORD, "line1\nline2\nline3\nline4", "Jumped. Take saved: " + LONG,
                 "Saved Section \"" + "Q" * 150 + "\".", "", "x"):
        w.summary_label.setText(text)
        check(f"status {text[:20]}")
    w.save_section_button.click()
    w.rename_section_button.click()
    w.delete_section_button.click()
    check("Section buttons")

    # --- calibration, Start/Stop, Pause, replay, History -------------------------------------------------------------
    w.start_calibration()
    check("calibration running")
    w.finish_calibration()
    check("calibration finished")
    # --- the Melody and Tolerance band previewed before Start (#71): every control that redraws it ---------------
    w.reference_box.setCurrentText("Melody")
    check("preview: Melody")
    w.timeline.section_marked.emit(0.3, 1.2)
    check("preview: Section marked")
    for v in (10, 50, 100):
        w.tolerance_slider.setValue(v)
        check(f"preview: Tolerance {v}")
    for k in range(w.octave_box.count()):
        w.octave_box.setCurrentIndex(k)
        check(f"preview: Octave {w.octave_box.currentText()}")
    for notes in (False, True):
        w.notes_box.setChecked(notes)
        check(f"preview: Notes {notes}")
    for i in range(w.song_box.count()):
        w.song_box.setCurrentIndex(i)
        check(f"preview: song {i}")
    w.clear_section()
    check("preview: Section cleared")
    w._set_start_waiting(True)                           # a Start waiting for its Speed says "Cancel start" (#72)
    check("Start waiting: Cancel start")
    w._set_start_waiting(False)
    check("Start no longer waiting")
    w.toggle()
    check("Start (Melody)")
    for _ in range(5):
        w.tick()
        QtWidgets.QApplication.processEvents()
    check("Take running")
    w.tolerance_slider.setValue(30)
    check("Tolerance moved mid-Take")
    w.section_name_edit.setFocus()
    w.field_focus_changed(w.section_name_edit, True)     # brown field and the hint on the status line (#72)
    check("text field focused mid-Take: brown hint")
    w.tick()
    check("hint, tick")
    w.field_focus_changed(w.section_name_edit, False)
    w.section_name_edit.clearFocus()
    check("text field left")
    w.toggle_pause()
    check("Pause")
    w.section_name_edit.setText(LONG_SECTION)
    w.field_focus_changed(w.section_name_edit, True)
    check("hint while paused")
    w.field_focus_changed(w.section_name_edit, False)
    w.section_name_edit.setText("")
    w.toggle_pause()
    check("Play")
    w.toggle()
    check("Stop")
    w.tolerance_slider.setValue(70)
    check("Tolerance after Stop: the preview replaces the Take's plot")
    w.open_history()
    check("History opened")
    w.history_window.reload()
    check("History reloaded")
    w.history_window.hide()
    check("History hidden")
    w.history_window.close()
    if take_id is not None:                      # replay: the Take's settings come on the controls and go again
        w.reference_box.setCurrentText("Nearest note")
        w.replay_take(take_id)
        check("replay started")
        w.tolerance_slider.setValue(60)
        for _ in range(5):
            w.tick()
        check("replaying")
        w.toggle_replay()
        check("replay stopped")

"""Sections and the loop in the window, with a fake Backing (no sound card): mark, save, rename, delete, L toggles the
loop, a per-Pass result shows while the loop goes on."""
import dataclasses
from datetime import datetime

from PySide6 import QtCore, QtWidgets
from audio import SR, quiet
from melody_helpers import library_song, reference_line
from test_window_timeline import NOTES, SeekableBacking

from sing_pitch.library import Library, Section
from sing_pitch.take_engine import TakeRecord, TakeSummary


class LoopingBacking(SeekableBacking):
    def __init__(self, path=None, vocals=None):
        super().__init__(path, vocals)
        self.loop = None

    def set_loop(self, loop):
        self.loop = loop
        self.calls.append(("loop", loop))

    def seek(self, song_s):
        self.loop = None
        super().seek(song_s)


def section_window(qtbot, tmp_path):
    from sing_pitch.audio_input import ArrayInput
    from sing_pitch.pitch.swiftf0 import SwiftF0Tracker
    from sing_pitch.ui import MainWindow

    library_song(tmp_path, reference_line(NOTES))
    backings = []

    def make(path, vocals=None):
        backings.append(LoopingBacking(path, vocals))
        return backings[-1]

    win = MainWindow(SwiftF0Tracker(), home=tmp_path, input_factory=lambda _i: ArrayInput(quiet(0.5), SR, 256),
                     backing_factory=make, output_info=lambda: ("WF-1000XM5", "headphones"))
    qtbot.addWidget(win)
    win.reference_box.setCurrentText("Melody")
    return win, backings


def song_id(win):
    return win.song_box.currentData()


def test_a_marked_section_is_kept_as_the_unnamed_mark_and_can_be_saved_renamed_and_deleted(qtbot, tmp_path):
    win, _ = section_window(qtbot, tmp_path)
    win.timeline.section_marked.emit(0.5, 1.5)            # what a drag on the strip emits
    assert Library(tmp_path).sections(song_id(win)).mark == Section(None, 0.5, 1.5)
    win.section_name_edit.setText("Chorus 1")
    win.save_section_button.click()
    assert Library(tmp_path).sections(song_id(win)).saved() == [Section("Chorus 1", 0.5, 1.5)]
    assert win.section_box.currentText() == "Chorus 1"
    win.timeline.section_changed.emit(0.4, 1.8)           # its edges dragged: the saved Section follows
    assert Library(tmp_path).sections(song_id(win)).saved() == [Section("Chorus 1", 0.4, 1.8)]
    win.section_name_edit.setText("Chorus")
    win.rename_section_button.click()
    assert [s.name for s in Library(tmp_path).sections(song_id(win)).saved()] == ["Chorus"]
    assert win.section_box.currentText() == "Chorus"
    win.delete_section_button.click()
    assert Library(tmp_path).sections(song_id(win)).saved() == []


def test_choosing_a_saved_section_shows_it_on_the_timeline_and_a_reopened_window_finds_it(qtbot, tmp_path):
    win, _ = section_window(qtbot, tmp_path)
    store = Library(tmp_path).sections(song_id(win))
    store.save("Verse", 0.2, 0.9)
    store.set_mark(1.0, 1.6)
    win.reload_sections()
    assert win.timeline.section == (1.0, 1.6)                     # the unnamed mark comes back
    win.section_box.setCurrentText("Verse")
    assert win.timeline.section == (0.2, 0.9)


def test_l_toggles_the_loop_on_the_running_take_and_a_jump_ends_it(qtbot, tmp_path):
    win, backings = section_window(qtbot, tmp_path)
    win.show()
    win.activateWindow()
    qtbot.waitUntil(lambda: QtWidgets.QApplication.activeWindow() is win)
    win.timeline.section_marked.emit(0.5, 1.5)
    win.toggle()
    qtbot.keyClick(win, QtCore.Qt.Key_L)
    loop = backings[0].loop
    assert win.loop_box.isChecked() and (loop.start_s, loop.end_s, loop.name) == (0.5, 1.5, None)
    assert abs(loop.lead_in_s - 2.0) < 1e-9
    qtbot.keyClick(win, QtCore.Qt.Key_L)
    assert backings[0].calls[-1] == ("loop", None) and not win.loop_box.isChecked()
    qtbot.keyClick(win, QtCore.Qt.Key_L)
    win.timeline.seek_requested.emit(0.2)                          # a jump ends the loop
    assert backings[0].loop is None and not win.loop_box.isChecked()
    win.toggle()


def test_the_loop_set_before_start_starts_the_take_on_the_section_and_needs_a_section(qtbot, tmp_path):
    win, backings = section_window(qtbot, tmp_path)
    win.loop_box.setChecked(True)                                  # nothing marked yet
    assert not win.loop_box.isChecked() and "Section" in win.summary_label.text()
    Library(tmp_path).sections(song_id(win)).save("Verse", 0.2, 0.9)
    win.reload_sections()
    win.section_box.setCurrentText("Verse")
    win.loop_box.setChecked(True)
    win.toggle()
    assert backings[0].calls[0] == ("loop", backings[0].loop) and backings[0].loop.name == "Verse"
    win.toggle()


def test_each_pass_result_is_shown_while_the_loop_goes_on(qtbot, tmp_path):
    win, _ = section_window(qtbot, tmp_path)
    win.toggle()
    summary = TakeSummary(300, 250, 6.0, 87.0, 12.0, "centred", "C4", "G4")
    record = TakeRecord("20261124-120000", datetime(2026, 11, 24, 12), "melody", (), summary, song_id="do-re-mi",
                        section_name="Chorus 1", pass_index=2)
    win._runner._saved_news.put(record)
    win.tick()
    assert "Pass 2" in win.summary_label.text() and "Chorus 1" in win.summary_label.text()
    assert "87%" in win.summary_label.text() and win.button.text() == "Stop"
    win._runner._saved_news.put(dataclasses.replace(record, section_name=None, pass_index=None))
    win.tick()
    assert win.summary_label.text().startswith("Jumped.")
    win.toggle()

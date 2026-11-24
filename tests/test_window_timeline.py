"""Timeline, Space and the arrow keys in the window, with a fake Backing (no sound card)."""
import numpy as np
from audio import SR, quiet
from melody_helpers import library_song, reference_line
from PySide6 import QtCore, QtWidgets
from test_melody_window import FakeBacking, window

NOTES = [(-900, 0.6), (-500, 0.6), (-200, 0.8)]


class SeekableBacking(FakeBacking):
    duration_s = 30.0

    def __init__(self, path=None, vocals=None):
        super().__init__(path, vocals)
        self.calls, self.position = [], 12.0

    def position_s(self):
        return self.position

    def seek(self, song_s):
        self.calls.append(("seek", song_s))

    def pause(self):
        self.calls.append(("pause",))

    def play(self):
        self.calls.append(("play",))


def running_window(qtbot, tmp_path):
    library_song(tmp_path, reference_line(NOTES))
    backings = []

    def make(path, vocals=None):
        backings.append(SeekableBacking(path, vocals))
        return backings[-1]

    from sing_pitch.audio_input import ArrayInput
    from sing_pitch.pitch.swiftf0 import SwiftF0Tracker
    from sing_pitch.ui import MainWindow

    win = MainWindow(SwiftF0Tracker(), home=tmp_path, input_factory=lambda _i: ArrayInput(quiet(0.5), SR, 256),
                     backing_factory=make, output_info=lambda: ("WF-1000XM5", "headphones"))
    qtbot.addWidget(win)
    win.reference_box.setCurrentText("Melody")
    return win, backings


def test_the_timeline_shows_the_chosen_song_before_start_and_hides_for_other_references(qtbot, tmp_path):
    win, _ = running_window(qtbot, tmp_path)
    x, _y = win.timeline._line.getData()
    assert not win.timeline.isHidden() and x is not None and len(x) > 10
    win.reference_box.setCurrentText("Key")
    assert win.timeline.isHidden()


def test_clicking_the_timeline_jumps_the_running_song_and_moves_the_playhead(qtbot, tmp_path):
    win, backings = running_window(qtbot, tmp_path)
    win.show()
    qtbot.waitExposed(win)
    win.seek(7.0)                                          # nothing runs yet: ignored
    win.toggle()
    assert backings[0].calls == []
    win.timeline.seek_requested.emit(20.5)                 # what a click on the strip emits
    assert backings[0].calls == [("seek", 20.5)] and win.timeline.playhead_s == 20.5
    win.tick()
    assert win.timeline.playhead_s == 12.0                 # the playhead follows the song position
    win.toggle()


def test_space_pauses_and_resumes_and_the_arrow_keys_jump_5_s(qtbot, tmp_path):
    win, backings = running_window(qtbot, tmp_path)
    win.show()
    win.activateWindow()
    qtbot.waitUntil(lambda: QtWidgets.QApplication.activeWindow() is win)    # window shortcuts need the active window
    qtbot.keyClick(win, QtCore.Qt.Key_Space)               # nothing runs yet: ignored
    win.toggle()
    win.button.setFocus()                                  # Space on a focused Stop button must pause, not stop
    assert win.pause_button.isEnabled() and win.pause_button.text() == "Pause"
    qtbot.keyClick(win, QtCore.Qt.Key_Space)
    assert backings[0].calls == [("pause",)] and win.pause_button.text() == "Play"
    qtbot.keyClick(win, QtCore.Qt.Key_Space)
    assert backings[0].calls[-1] == ("play",) and win.pause_button.text() == "Pause"
    qtbot.keyClick(win, QtCore.Qt.Key_Right)
    qtbot.keyClick(win, QtCore.Qt.Key_Left)
    backings[0].position = 2.0
    qtbot.keyClick(win, QtCore.Qt.Key_Left)                # 2 - 5 would be before the song start
    assert backings[0].calls[-3:] == [("seek", 17.0), ("seek", 7.0), ("seek", 0.0)]
    win.pause_button.click()
    assert backings[0].calls[-1] == ("pause",) and win.button.text() == "Stop"
    win.toggle()
    assert not win.pause_button.isEnabled()


def test_the_reference_line_follows_the_clock_across_a_jump(qtbot, tmp_path):
    """Drawn on the voice clock through the Take's own song clock, not as song time plus one constant."""
    from sing_pitch.take_engine import SampledClock

    win, _ = running_window(qtbot, tmp_path)
    reference = win._reference()
    win._show_melody(reference, 0.0)
    # the voice clock reads 10 s, the song was at 0..1 s and then jumped back to 0.0
    win._replay_clock = SampledClock([0.0, 1.0, 1.01, 2.0], [0.0, 1.0, 0.0, 0.99])
    win._draw_melody(1.5)
    x, y = win._ref_curve.getData()
    first = np.nanargmin(np.abs(x - 0.5)), np.nanargmin(np.abs(x - 1.5))
    assert abs(y[first[0]] - y[first[1]]) < 0.01           # same note 1 s apart on the voice clock: the jump was drawn
    assert not np.isnan(y[first[0]])

"""Speed in the window: the control, the +/- keys, the "preparing N %..." message while the stems are stretched in a
background thread, and the stretched stems reaching the Backing player. No sound card, no Rubber Band."""
import threading

import numpy as np
from audio import SR, quiet
from melody_helpers import library_song, reference_line
from test_melody_window import window
from test_speed_cache import CountingStretcher

from PySide6 import QtCore

NOTES = [(-900, 0.6), (-500, 0.6), (-200, 0.8)]


class GatedStretcher(CountingStretcher):
    """Waits for the test to open the gate, so the test can look at the window while a stretch is in flight."""

    def __init__(self):
        super().__init__()
        self.gate = threading.Event()

    def __call__(self, audio, sample_rate, speed):
        assert self.gate.wait(10), "the test never opened the gate"
        return super().__call__(audio, sample_rate, speed)


def melody_window(qtbot, tmp_path, stretcher, backings=None):
    song = library_song(tmp_path, reference_line(NOTES), seconds=2.0)
    win = window(qtbot, tmp_path, quiet(3.0), backings=backings, stretcher=stretcher)
    win.reference_box.setCurrentText("Melody")
    return win, song


def test_the_speed_control_runs_from_50_to_120_in_steps_of_5_and_acts_with_plus_and_minus(qtbot, tmp_path):
    win, _ = melody_window(qtbot, tmp_path, CountingStretcher())
    box = win.speed_box
    assert (box.minimum(), box.maximum(), box.singleStep(), box.value()) == (50, 120, 5, 100)
    win.show()
    qtbot.waitExposed(win)
    win.activateWindow()
    qtbot.waitUntil(win.isActiveWindow, timeout=2000)
    qtbot.keyClick(win, QtCore.Qt.Key_Minus)
    assert box.value() == 95
    qtbot.keyClick(win, QtCore.Qt.Key_Plus)
    qtbot.keyClick(win, QtCore.Qt.Key_Plus)
    assert box.value() == 105
    for _ in range(30):
        win.speed_up()
    assert box.value() == 120
    for _ in range(30):
        win.speed_down()
    assert box.value() == 50
    win.reference_box.setCurrentText("Key")
    assert not box.isEnabled()


def test_a_new_speed_shows_preparing_at_once_without_blocking_and_start_waits_for_it(qtbot, tmp_path):
    stretcher = GatedStretcher()
    backings = []
    win, song = melody_window(qtbot, tmp_path, stretcher, backings)
    win.speed_box.setValue(70)
    qtbot.waitUntil(lambda: "preparing 70 %" in win.speed_label.text(), timeout=3000)     # the UI thread stayed free
    win.toggle()
    assert win._runner is None and "preparing 70 %" in win.summary_label.text()           # Start waits, says why
    stretcher.gate.set()
    qtbot.waitUntil(lambda: win.speed_label.text() == "", timeout=5000)
    assert (song.dir / "speed" / "70" / "backing.wav").exists()
    win.toggle()
    assert win._runner is not None
    speed, backing, vocal = backings[0].speeds[0]
    assert speed == 0.7 and abs(len(backing) - SR * 2.0 / 0.7) <= 1 and vocal is not None
    win.toggle()


def test_a_cached_speed_is_not_stretched_again_and_a_change_mid_take_reaches_the_player(qtbot, tmp_path):
    stretcher = CountingStretcher()
    backings = []
    win, song = melody_window(qtbot, tmp_path, stretcher, backings)
    win.speed_box.setValue(70)
    qtbot.waitUntil(lambda: (song.dir / "speed" / "70" / "backing.wav").exists() and win.speed_label.text() == "",
                    timeout=5000)
    assert len(stretcher.calls) == 2
    win.speed_box.setValue(100)
    win.toggle()                                        # Take starts at 100 %, nothing to hand the player
    assert backings[0].speeds == []
    win.speed_box.setValue(70)                          # mid-Take, cached: no preparing message
    qtbot.waitUntil(lambda: len(backings[0].speeds) == 1, timeout=5000)
    assert backings[0].speeds[0][0] == 0.7 and len(stretcher.calls) == 2 and win.speed_label.text() == ""
    win.speed_box.setValue(55)                          # mid-Take, not cached: old speed plays until it is ready
    qtbot.waitUntil(lambda: "preparing 55 %" in win.speed_label.text() or len(backings[0].speeds) == 2, timeout=3000)
    qtbot.waitUntil(lambda: len(backings[0].speeds) == 2, timeout=5000)
    assert backings[0].speeds[1][0] == 0.55
    win.toggle()


def test_a_failed_stretch_says_so_and_goes_back_to_the_speed_that_plays(qtbot, tmp_path):
    def broken(audio, sample_rate, speed):
        raise RuntimeError("out of memory")

    win, _ = melody_window(qtbot, tmp_path, broken)
    win.speed_box.setValue(60)
    qtbot.waitUntil(lambda: "failed" in win.speed_label.text(), timeout=5000)
    assert "out of memory" in win.speed_label.text() and win.speed_box.value() == 100


def test_the_reference_line_is_drawn_stretched_with_the_speed(qtbot, tmp_path):
    backings = []
    win, song = melody_window(qtbot, tmp_path, CountingStretcher(), backings)
    win.speed_box.setValue(50)
    qtbot.waitUntil(lambda: (song.dir / "speed" / "50" / "backing.wav").exists() and win.speed_label.text() == "",
                    timeout=5000)
    win.toggle()
    win._melody_latency = 0.0
    win._draw_melody(10.0)
    x, _ = win._ref_curve.getData()
    assert abs(x[-1] - 2.0 * 2.0) < 0.05 + 0.02 * 2 and x[0] >= 0.0         # a 2 s melody lasts 4 s of output at 50 %
    win.toggle()

import os
import subprocess
import sys

import numpy as np
from audio import SR, harmonic, quiet
from frames import hz_at

from sing_pitch.audio_input import ArrayInput
from sing_pitch.pitch.swiftf0 import SwiftF0Tracker


def test_smoke_launches_headless_and_quits():
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    done = subprocess.run([sys.executable, "-m", "sing_pitch", "--smoke"], env=env, capture_output=True,
                          text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert done.returncode == 0 and "smoke ok" in done.stdout, done.stderr


def test_window_start_sing_stop_shows_the_summary_and_saves(qtbot, tmp_path):
    from sing_pitch.ui import MainWindow

    audio = np.concatenate([quiet(0.3), harmonic(hz_at(10), 1.5), quiet(0.3)])
    win = MainWindow(SwiftF0Tracker(), home=tmp_path, input_factory=lambda _i: ArrayInput(audio, SR, 256))
    qtbot.addWidget(win)
    win.toggle()
    qtbot.wait(100)
    win.toggle()
    assert "In tune" in win.summary_label.text()
    assert (tmp_path / "takes.csv").exists()
    assert not win.grab().isNull()


class FakeDrone:
    def __init__(self):
        self.started_with, self.stopped = None, False

    def start(self, key):
        self.started_with = key

    def stop(self):
        self.stopped = True


def test_key_take_in_the_window_flags_out_of_key_plays_the_drone_and_saves_the_key(qtbot, tmp_path):
    import csv

    from sing_pitch.ui import MainWindow

    audio = np.concatenate([quiet(0.3), harmonic(hz_at(-515), 1.2), quiet(0.3), harmonic(hz_at(-300), 1.2), quiet(0.3)])
    drone = FakeDrone()
    win = MainWindow(SwiftF0Tracker(), home=tmp_path, input_factory=lambda _i: ArrayInput(audio, SR, 256),
                     drone_factory=lambda: drone)
    qtbot.addWidget(win)
    assert win.key_box.count() == 24 and not win.key_box.isEnabled()      # picker only for the Key Reference
    win.reference_box.setCurrentText("Key")
    assert win.key_box.isEnabled() and win.drone_box.isEnabled()
    win.key_box.setCurrentText("C major")
    win.drone_box.setChecked(True)
    win.toggle()
    assert drone.started_with.name == "C major"
    seen = []
    for _ in range(200):
        qtbot.wait(20)
        win.tick()
        seen.append(win.note_label.text())
        if any("out of key" in t for t in seen):
            break
    assert any("out of key" in t for t in seen)
    win.toggle()
    assert drone.stopped
    assert "1 out of key" in win.summary_label.text() and "C major" in win.summary_label.text()
    row = next(csv.DictReader(open(tmp_path / "takes.csv")))
    assert (row["reference"], row["key"], row["out_of_key_count"]) == ("key", "C major", "1")


def test_nearest_note_take_never_touches_the_drone(qtbot, tmp_path):
    from sing_pitch.ui import MainWindow

    drone = FakeDrone()
    audio = np.concatenate([quiet(0.3), harmonic(hz_at(10), 0.8), quiet(0.3)])
    win = MainWindow(SwiftF0Tracker(), home=tmp_path, input_factory=lambda _i: ArrayInput(audio, SR, 256),
                     drone_factory=lambda: drone)
    qtbot.addWidget(win)
    win.drone_box.setChecked(True)          # ticked, but the Reference is Nearest note
    win.toggle(), qtbot.wait(100), win.toggle()
    assert drone.started_with is None
    assert "out of key" not in win.summary_label.text()


def test_a_missing_output_device_does_not_stop_the_take(qtbot, tmp_path):
    from sing_pitch.ui import MainWindow

    class Broken:
        def start(self, key):
            raise OSError("no output device")

        def stop(self):
            pass

    audio = np.concatenate([quiet(0.3), harmonic(hz_at(-500), 0.8), quiet(0.3)])
    win = MainWindow(SwiftF0Tracker(), home=tmp_path, input_factory=lambda _i: ArrayInput(audio, SR, 256),
                     drone_factory=Broken)
    qtbot.addWidget(win)
    win.reference_box.setCurrentText("Key")
    win.drone_box.setChecked(True)
    win.toggle()
    assert win.button.text() == "Stop"
    assert "Drone" in win.summary_label.text()
    qtbot.wait(100)
    win.toggle()
    assert (tmp_path / "takes.csv").exists()


def test_replay_plays_the_last_saved_take_back_through_the_live_view(qtbot, tmp_path):
    from sing_pitch.ui import MainWindow

    audio = np.concatenate([quiet(0.3), harmonic(hz_at(10), 1.0), quiet(0.3)])
    win = MainWindow(SwiftF0Tracker(), home=tmp_path, input_factory=lambda _i: ArrayInput(audio, SR, 256))
    qtbot.addWidget(win)
    win.toggle()
    qtbot.wait(100)
    win.toggle()
    assert "steadiness" in win.summary_label.text()

    win.toggle_replay()
    qtbot.waitUntil(lambda: len(win._t) > 20, timeout=3000)
    assert win._midi and not np.all(np.isnan(np.fromiter(win._midi, float)))
    qtbot.waitUntil(lambda: win.summary_label.text() == "Replay finished.", timeout=6000)
    assert win.button.isEnabled() and win.replay_button.text() == "Replay last Take"


def test_replay_with_no_saved_take_says_so(qtbot, tmp_path):
    from sing_pitch.ui import MainWindow

    win = MainWindow(SwiftF0Tracker(), home=tmp_path)
    qtbot.addWidget(win)
    win.toggle_replay()
    assert "No saved Take" in win.summary_label.text()


def test_replaying_a_key_take_flags_the_out_of_key_note_again(qtbot, tmp_path):
    from sing_pitch.ui import MainWindow

    audio = np.concatenate([quiet(0.3), harmonic(hz_at(-300), 1.2), quiet(0.3)])
    win = MainWindow(SwiftF0Tracker(), home=tmp_path, input_factory=lambda _i: ArrayInput(audio, SR, 256),
                     drone_factory=FakeDrone)
    qtbot.addWidget(win)
    win.reference_box.setCurrentText("Key")
    win.key_box.setCurrentText("C major")
    win.toggle(), qtbot.wait(100), win.toggle()
    win.toggle_replay()
    qtbot.waitUntil(lambda: any(win._out), timeout=3000)

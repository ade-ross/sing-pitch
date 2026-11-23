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

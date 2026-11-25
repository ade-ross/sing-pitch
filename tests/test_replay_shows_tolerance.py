"""#67: a replay shows its Take's settings, the Tolerance included: the slider and its label show the Take's value,
the slider is locked, nothing is remembered, and his own value is back after the replay."""
import numpy as np
from audio import quiet, harmonic
from frames import hz_at
from test_tolerance_window import nearest_window, sing

from sing_pitch.settings import Settings


def test_the_slider_shows_the_takes_tolerance_locked_during_a_replay_and_his_own_after(qtbot, tmp_path):
    audio = np.concatenate([quiet(0.2), harmonic(hz_at(40), 1.5), quiet(0.3)])
    win = nearest_window(qtbot, tmp_path, audio)
    win.tolerance_slider.setValue(50)
    sing(qtbot, win)
    win.tolerance_slider.setValue(100)                      # his own value now
    win.toggle_replay()
    assert win._replay is not None
    assert win.tolerance_slider.value() == 50 and win.tolerance_label.text() == "±50 c"
    assert not win.tolerance_slider.isEnabled()
    assert Settings.load(tmp_path).tolerance_cents == 100   # showing the Take's value remembered nothing
    win.toggle_replay()
    assert win.tolerance_slider.value() == 100 and win.tolerance_label.text() == "±100 c"
    assert win.tolerance_slider.isEnabled()
    assert Settings.load(tmp_path).tolerance_cents == 100

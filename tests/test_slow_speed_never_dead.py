"""A slow Speed preparation (machine under load, a long song) never leaves Start, Space or Stop doing nothing."""
from test_speed_window import GatedStretcher, melody_window


def test_a_start_waiting_for_a_slow_speed_shows_cancel_and_the_button_cancels_it(qtbot, tmp_path):
    stretcher = GatedStretcher()
    win, _ = melody_window(qtbot, tmp_path, stretcher)
    win.speed_box.setValue(70)
    win.toggle()
    assert win._runner is None and win.button.text() == "Cancel start" and win.button.isEnabled()
    win.toggle()
    assert win.button.text() == "Start" and not win._start_waiting
    win.toggle()                                         # and Start works again, waiting again for the same Speed
    assert win.button.text() == "Cancel start"
    stretcher.gate.set()
    qtbot.waitUntil(lambda: win._runner is not None, timeout=5000)
    assert win.button.text() == "Stop"
    win.toggle()
    assert win.button.text() == "Start"


def test_a_start_waiting_is_cancelled_by_going_back_to_100_percent_which_starts_at_once(qtbot, tmp_path):
    stretcher = GatedStretcher()
    backings = []
    win, _ = melody_window(qtbot, tmp_path, stretcher, backings)
    win.speed_box.setValue(70)
    win.toggle()
    win.speed_box.setValue(100)
    assert win._runner is not None and win.button.text() == "Stop" and backings[0].speeds == []
    win.toggle()
    stretcher.gate.set()                                 # the abandoned preparation finishes into the cache, quietly
    qtbot.waitUntil(lambda: win.speed_label.text() == "", timeout=5000)
    assert win._runner is None and win.button.text() == "Start"


def test_a_speed_still_preparing_mid_take_leaves_pause_and_stop_working_and_its_end_after_stop_is_harmless(qtbot, tmp_path):
    stretcher = GatedStretcher()
    backings = []
    win, _ = melody_window(qtbot, tmp_path, stretcher, backings)
    win.toggle()
    win.speed_box.setValue(70)                           # asked mid-Take: the stretch is blocked
    qtbot.waitUntil(lambda: "preparing 70 %" in win.speed_label.text(), timeout=3000)
    win.toggle_pause()
    assert win._paused and win.pause_button.text() == "Play"
    win.toggle_pause()
    assert not win._paused
    win.toggle()                                         # Stop works with a stretch in flight
    assert win._runner is None and win.button.text() == "Start"
    stretcher.gate.set()
    qtbot.waitUntil(lambda: win.speed_label.text() == "", timeout=5000)
    assert backings[0].speeds == [] and win._runner is None          # nothing handed to a stopped player
    win.toggle()                                         # Start still works, now at the prepared Speed
    qtbot.waitUntil(lambda: win._runner is not None, timeout=3000)
    win.toggle()

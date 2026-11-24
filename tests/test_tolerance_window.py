"""The Tolerance slider in the window (#41): ±10 to ±100 cents in 5-cent steps, default ±25, remembered as an app
setting, used by every Reference, locked during a Take, recorded with the Take; a replay is coloured with the
Tolerance the Take was saved with. Notes mode: a bar takes its note's verdict colour once the note has ended."""
import csv

import numpy as np
from audio import SR, harmonic, quiet
from frames import hz_at
from melody_helpers import library_song, reference_line
from test_melody_window import window

from sing_pitch.reference import Band
from sing_pitch.settings import Settings


def nearest_window(qtbot, home, audio):
    win = window(qtbot, home, audio)
    win.reference_box.setCurrentText("Nearest note")
    return win


def sing(qtbot, win, seconds=0.25):
    win.toggle()
    qtbot.wait(int(seconds * 1000))
    win.tick()
    win.toggle()


def rows(home):
    return list(csv.DictReader(open(home / "takes.csv")))


def test_the_slider_runs_from_10_to_100_cents_in_5_cent_steps_and_starts_at_25(qtbot, tmp_path):
    win = nearest_window(qtbot, tmp_path, quiet(1.0))
    s = win.tolerance_slider
    assert (s.minimum(), s.maximum(), s.singleStep(), s.pageStep(), s.value()) == (10, 100, 5, 5, 25)
    assert win.tolerance_label.text() == "±25 c"
    s.setValue(23)
    assert s.value() == 25                                    # snapped to a 5-cent step
    s.setValue(47)
    assert s.value() == 45 and win.tolerance_label.text() == "±45 c"
    assert s.toolTip() and win.tolerance_label.toolTip()


def test_the_tolerance_is_remembered_for_the_next_window(qtbot, tmp_path):
    win = nearest_window(qtbot, tmp_path, quiet(1.0))
    win.tolerance_slider.setValue(60)
    assert Settings.load(tmp_path).tolerance_cents == 60
    again = nearest_window(qtbot, tmp_path, quiet(1.0))
    assert again.tolerance_slider.value() == 60


def test_a_bad_saved_tolerance_reads_as_the_nearest_allowed_value(tmp_path):
    (tmp_path / "settings.json").write_text('{"tolerance_cents": 3}')
    assert Settings.load(tmp_path).tolerance_cents == 10
    (tmp_path / "settings.json").write_text('{"tolerance_cents": "x"}')
    assert Settings.load(tmp_path).tolerance_cents == 25


def test_a_take_is_scored_and_saved_with_the_tolerance_set_and_the_slider_is_locked_meanwhile(qtbot, tmp_path):
    audio = np.concatenate([quiet(0.2), harmonic(hz_at(40), 1.5), quiet(0.3)])     # 40 cents sharp of A4
    win = nearest_window(qtbot, tmp_path, audio)
    win.tolerance_slider.setValue(50)
    win.toggle()
    assert not win.tolerance_slider.isEnabled()
    qtbot.wait(250)
    win.tick()
    win.toggle()
    assert win.tolerance_slider.isEnabled()
    (row,) = rows(tmp_path)
    assert float(row["tolerance_cents"]) == 50.0 and float(row["pct_in_tune"]) >= 90.0
    assert "±50 c" in win.summary_label.text()


def test_a_replay_is_coloured_with_the_saved_tolerance_not_the_slider(qtbot, tmp_path):
    audio = np.concatenate([quiet(0.2), harmonic(hz_at(40), 1.5), quiet(0.3)])
    win = nearest_window(qtbot, tmp_path, audio)
    win.tolerance_slider.setValue(50)
    sing(qtbot, win)
    win.tolerance_slider.setValue(10)
    win.toggle_replay()
    qtbot.waitUntil(lambda: len([b for b in win._band if b is not None]) > 30, timeout=4000)
    bands = [b for b in win._band if b is not None]
    assert sum(b is Band.GREEN for b in bands) / len(bands) > 0.9
    win.toggle_replay()
    assert "±50 c" in win.summary_label.text()


def test_in_notes_mode_a_bar_turns_its_verdict_colour_once_the_note_has_ended(qtbot, tmp_path):
    notes = [(-900, 0.6), (-500, 0.6), (-200, 0.8)]
    library_song(tmp_path, reference_line(notes), seconds=2.0)
    s = Settings.load(tmp_path)
    s.set_latency("WF-1000XM5", 0.0)
    s.save()
    audio = np.concatenate([harmonic(hz_at(c), d) for c, d in notes] + [quiet(0.4)])
    win = window(qtbot, tmp_path, audio)
    win.reference_box.setCurrentText("Melody")
    win.notes_box.setChecked(True)
    win.toggle()
    qtbot.wait(300)
    win.tick()
    win.toggle()
    win.toggle_replay()
    qtbot.waitUntil(lambda: 0 in win._verdicts, timeout=4000)       # the first note (ends at 0.6 s) has its verdict
    assert win._verdicts[0] is Band.GREEN
    win.tick()
    _, green = win._verdict_curves[Band.GREEN].getData()
    assert green is not None and np.isfinite(green).any()           # drawn in green
    _, blue = win._ref_curve.getData()
    assert np.isfinite(blue).any()                                  # the notes still to come stay blue


def test_replaying_a_take_from_history_uses_its_saved_tolerance_not_the_slider(qtbot, tmp_path):
    """#41 + #43: History's Replay (replay_take(id)) colours the Take with the Tolerance it was scored with."""
    audio = np.concatenate([quiet(0.2), harmonic(hz_at(40), 1.5), quiet(0.3)])
    win = nearest_window(qtbot, tmp_path, audio)
    win.tolerance_slider.setValue(50)
    sing(qtbot, win)
    take_id = rows(tmp_path)[-1]["id"]
    win.tolerance_slider.setValue(10)
    win.replay_take(take_id)
    qtbot.waitUntil(lambda: len([b for b in win._band if b is not None]) > 30, timeout=4000)
    bands = [b for b in win._band if b is not None]
    assert sum(b is Band.GREEN for b in bands) / len(bands) > 0.9
    win.toggle_replay()
    assert "±50 c" in win.summary_label.text()

"""#43: Speed presets (70 / 85 / 100 beside the Speed box) and replaying any Take from History with its saved settings."""
import csv

import numpy as np
from audio import SR, harmonic, quiet
from frames import hz_at
from melody_helpers import library_song, reference_line
from PySide6 import QtCore
from test_melody_window import window
from test_speed_cache import CountingStretcher

from sing_pitch.library import Library
from sing_pitch.take_store import COLUMNS

NOTES = [(-900, 0.6), (-500, 0.6), (-200, 0.8)]


def melody_window(qtbot, tmp_path, stretcher=None, **kw):
    song = library_song(tmp_path, reference_line(NOTES), seconds=2.0)
    audio = np.concatenate([quiet(0.2)] + [harmonic(hz_at(c), d) for c, d in NOTES] + [quiet(0.4)])
    win = window(qtbot, tmp_path, audio, stretcher=stretcher or CountingStretcher(), **kw)
    win.reference_box.setCurrentText("Melody")
    return win, song


# --- Speed presets ------------------------------------------------------------------------------------------------

def test_three_preset_buttons_sit_beside_the_speed_box(qtbot, tmp_path):
    win, _ = melody_window(qtbot, tmp_path)
    buttons = (win.speed_70_button, win.speed_85_button, win.speed_100_button)
    assert [b.text() for b in buttons] == ["70 %", "85 %", "100 %"]
    assert all(b.toolTip() for b in buttons) and all(b.objectName() for b in buttons)


def test_a_preset_sets_the_speed_box_and_prepares_through_the_same_path_as_the_box(qtbot, tmp_path):
    stretcher = CountingStretcher()
    backings = []
    win, song = melody_window(qtbot, tmp_path, stretcher, backings=backings)
    win.speed_85_button.click()
    assert win.speed_box.value() == 85 and "preparing 85 %" in win.speed_label.text()
    qtbot.waitUntil(lambda: (song.dir / "speed" / "85" / "backing.wav").exists() and win.speed_label.text() == "",
                    timeout=5000)
    assert Library(tmp_path).player_settings(song.id).load().speed_pct == 85          # remembered like the box
    win.speed_85_button.click()                                                        # cached: nothing stretched again
    assert len(stretcher.calls) == 2
    win.speed_100_button.click()
    assert win.speed_box.value() == 100 and win.speed_label.text() == ""
    win.speed_70_button.click()
    assert win.speed_box.value() == 70


def test_the_preset_of_the_current_speed_is_shown_pressed(qtbot, tmp_path):
    win, _ = melody_window(qtbot, tmp_path)
    pressed = lambda: [b.text() for b in (win.speed_70_button, win.speed_85_button, win.speed_100_button)  # noqa: E731
                       if b.isChecked()]
    assert pressed() == ["100 %"]
    win.speed_box.setValue(70)
    assert pressed() == ["70 %"]
    win.speed_box.setValue(75)
    assert pressed() == []
    win.speed_box.setValue(85)
    assert pressed() == ["85 %"]


def test_the_presets_are_greyed_like_the_speed_box_outside_melody(qtbot, tmp_path):
    win, _ = melody_window(qtbot, tmp_path)
    buttons = (win.speed_70_button, win.speed_85_button, win.speed_100_button)
    assert all(b.isEnabled() for b in buttons)
    win.reference_box.setCurrentText("Nearest note")
    assert not any(b.isEnabled() for b in buttons)
    win.reference_box.setCurrentText("Melody")
    assert all(b.isEnabled() for b in buttons)


# --- the History view lists Takes with a Replay action ------------------------------------------------------------

def write_rows(home, rows):
    with open(home / "takes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def row(i, day, reference="nearest", song="", pct="50", **kw):
    return {"id": i, "started_at": f"2026-11-{day:02d}T10:00:00", "reference": reference, "song_id": song,
            "pct_in_tune": pct, "median_abs_cents": "20", **kw}


def history(qtbot, tmp_path):
    from sing_pitch.ui.history_view import HistoryView

    write_rows(tmp_path, [row("a", 10), row("b", 11, reference="melody", song="s1", pct="70", speed_pct="85"),
                          row("c", 12, pct="60")])
    view = HistoryView(tmp_path)
    qtbot.addWidget(view)
    return view


def test_history_lists_every_filtered_take_newest_first(qtbot, tmp_path):
    view = history(qtbot, tmp_path)
    ids = lambda: [view.takes_table.item(r, 0).data(QtCore.Qt.UserRole) for r in range(view.takes_table.rowCount())]  # noqa: E731
    assert ids() == ["c", "b", "a"]
    view.reference_box.setCurrentIndex(view.reference_box.findData("melody"))
    assert ids() == ["b"]
    row_b = [view.takes_table.item(0, c).text() for c in range(view.takes_table.columnCount())]
    assert "Melody" in row_b and "85 %" in row_b and "70 %" in row_b and "s1" in row_b


def test_the_replay_button_needs_a_chosen_row_and_asks_for_that_take(qtbot, tmp_path):
    view = history(qtbot, tmp_path)
    assert not view.replay_button.isEnabled() and view.replay_button.toolTip()
    got = []
    view.replay_requested.connect(got.append)
    view.takes_table.selectRow(1)                       # b
    assert view.replay_button.isEnabled()
    view.replay_button.click()
    assert got == ["b"]


def test_a_double_click_on_a_row_asks_for_a_replay_too(qtbot, tmp_path):
    view = history(qtbot, tmp_path)
    got = []
    view.replay_requested.connect(got.append)
    view.takes_table.cellDoubleClicked.emit(2, 0)       # a
    assert got == ["a"]


# --- the main window replays that Take with its settings ----------------------------------------------------------

def sing_a_take(qtbot, win):
    win.toggle()
    qtbot.wait(300)
    win.tick()
    win.toggle()
    return next(r for r in reversed(list(csv.DictReader(open(win._home / "takes.csv")))))["id"]


def test_replaying_an_older_take_shows_its_settings_and_the_old_ones_come_back_after(qtbot, tmp_path):
    win, song = melody_window(qtbot, tmp_path)
    win.octave_box.setCurrentIndex(win.octave_box.findData(-1))
    win.notes_box.setChecked(True)
    first = sing_a_take(qtbot, win)
    rows = list(csv.DictReader(open(tmp_path / "takes.csv")))
    for r in rows:
        r["speed_pct"] = "85"                                       # as if it had been sung at 85 %
    write_rows(tmp_path, rows)
    win.notes_box.setChecked(False)                                 # the controls now say something else
    win.octave_box.setCurrentIndex(0)
    win.speed_box.setValue(100)
    win.reference_box.setCurrentText("Nearest note")
    second = sing_a_take(qtbot, win)
    assert first != second
    win.replay_take(first)
    assert win._replay_take_id == first and f"Replaying Take {first}" in win.summary_label.text()
    assert win.reference_box.currentText() == "Melody" and win.song_box.currentData() == song.id
    assert win.octave_box.currentData() == -1 and win.notes_box.isChecked() and win.speed_box.value() == 85
    assert not (song.dir / "speed" / "85").exists()                 # looking back stretches nothing
    saved = Library(tmp_path).player_settings(song.id).load()
    assert (saved.octave, saved.notes, saved.speed_pct) == (None, False, 100)       # and changes nothing remembered
    win.toggle_replay()                                             # Stop replay
    assert win.reference_box.currentText() == "Nearest note"
    assert win.octave_box.currentData() is None and not win.notes_box.isChecked() and win.speed_box.value() == 100


def replay_locked_controls(win):
    return (win.reference_box, win.song_box, win.key_box, win.speed_box, win.speed_70_button, win.speed_85_button,
            win.speed_100_button, win.octave_box, win.between_box, win.bpm_box, win.beats_box)


def test_during_a_replay_the_controls_it_shows_are_locked_and_nothing_is_remembered_or_stretched(qtbot, tmp_path):
    """Review 5 #3: the replay put its Take's settings on live controls; touching one (Speed, a preset, BPM, the song)
    wrote them to player.json as the song's own and started a stretch."""
    win, song = melody_window(qtbot, tmp_path)
    take = sing_a_take(qtbot, win)
    remembered = song.dir / "player.json"
    before = remembered.read_text() if remembered.exists() else None
    enabled = [w.isEnabled() for w in replay_locked_controls(win)]
    win.replay_take(take)
    assert not any(w.isEnabled() for w in replay_locked_controls(win))
    win.speed_box.setValue(70)                                      # even a change made by code changes nothing
    win.speed_85_button.click()
    win.bpm_box.setValue(90.0)
    win.beats_box.setValue(3)
    win.between_box.setCurrentText("Count-in")
    win.octave_box.setCurrentIndex(win.octave_box.findData(1))
    qtbot.wait(700)                                                 # longer than the stretch's start delay
    assert win._speed_job is None and not win._speed_timer.isActive()
    assert not (song.dir / "speed").exists()
    assert (remembered.read_text() if remembered.exists() else None) == before
    win.toggle_replay()
    assert [w.isEnabled() for w in replay_locked_controls(win)] == enabled


def test_replaying_while_another_replay_runs_switches_to_the_new_take(qtbot, tmp_path):
    win, song = melody_window(qtbot, tmp_path)
    first = sing_a_take(qtbot, win)
    second = sing_a_take(qtbot, win)
    win.replay_take(first)
    win.replay_take(second)
    assert win._replay_take_id == second
    win.toggle_replay()
    assert win._replay is None


def test_a_take_whose_song_is_gone_says_so_and_changes_nothing(qtbot, tmp_path):
    import shutil

    win, song = melody_window(qtbot, tmp_path)
    take = sing_a_take(qtbot, win)
    win.reference_box.setCurrentText("Nearest note")
    shutil.rmtree(song.dir)
    win.replay_take(take)
    text = win.summary_label.text()
    assert "Cannot replay" in text and "no longer in the library" in text
    assert win._replay is None and win.reference_box.currentText() == "Nearest note"


def test_no_take_can_be_replayed_while_one_is_being_sung(qtbot, tmp_path):
    win, song = melody_window(qtbot, tmp_path)
    take = sing_a_take(qtbot, win)
    win.toggle()
    win.replay_take(take)
    assert win._replay is None and "Stop" in win.summary_label.text()
    win.toggle()


def test_the_history_view_asks_the_main_window_to_replay(qtbot, tmp_path):
    win, song = melody_window(qtbot, tmp_path)
    take = sing_a_take(qtbot, win)
    win.open_history()
    qtbot.addWidget(win.history_window)
    table = win.history_window.takes_table
    assert table.rowCount() == 1
    table.selectRow(0)
    win.history_window.replay_button.click()
    assert win._replay_take_id == take
    win.toggle_replay()


def test_a_failed_replay_is_also_said_in_the_history_window(qtbot, tmp_path):
    import shutil

    win, song = melody_window(qtbot, tmp_path)
    sing_a_take(qtbot, win)
    win.open_history()
    qtbot.addWidget(win.history_window)
    shutil.rmtree(song.dir)
    win.history_window.takes_table.selectRow(0)
    win.history_window.replay_button.click()
    assert "no longer in the library" in win.history_window.status_label.text()


def test_replaying_a_key_take_shows_its_key_and_the_old_one_comes_back_after(qtbot, tmp_path):
    """QA #47: the Key box stayed on the current key (findData compared the Key objects by identity)."""
    win, _ = melody_window(qtbot, tmp_path)
    win.reference_box.setCurrentText("Key")
    win.key_box.setCurrentText("D major")
    take = sing_a_take(qtbot, win)
    win.reference_box.setCurrentText("Melody")
    win.key_box.setCurrentText("C major")
    win.replay_take(take)
    assert win.reference_box.currentText() == "Key" and win.key_box.currentText() == "D major"
    win.toggle_replay()
    assert win.key_box.currentText() == "C major"

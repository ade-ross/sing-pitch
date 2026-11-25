"""#52, #53, #58: no text ever changes the window's size or minimum size (a resize under a running Take breaks the
plot and the Timeline, #49), the Section name field keeps a usable width, and the first-launch size gives the plot room."""
import numpy as np
import pytest
from melody_helpers import library_song, reference_line
from PySide6 import QtWidgets
from test_melody_window import window
from test_navigate_and_loops import echo_singer
from audio import quiet

NOTES = [(-900, 0.6), (-500, 0.6), (-200, 0.8)]
LONG = "A very long message that goes on and on and on and on, much wider than any window could ever be " * 3


def sizes(win):
    QtWidgets.QApplication.processEvents()
    return (win.width(), win.height(), win.minimumSize().width(), win.minimumSize().height(),
            win.minimumSizeHint().width(), win.minimumSizeHint().height())


def melody_window(qtbot, tmp_path):
    library_song(tmp_path, reference_line(NOTES), title="Do Re Mi", seconds=2.0)
    win = window(qtbot, tmp_path, quiet(3.0))
    win.reference_box.setCurrentText("Melody")
    return win


def shown_at(qtbot, win, width, height):
    win.resize(width, height)
    win.show()
    qtbot.waitExposed(win)
    QtWidgets.QApplication.processEvents()


@pytest.mark.parametrize("small", [False, True], ids=["default size", "smallest size"])
def test_no_label_text_changes_the_window_size_or_minimum_size(qtbot, tmp_path, small):
    win = melody_window(qtbot, tmp_path)
    shown_at(qtbot, win, 1, 1) if small else shown_at(qtbot, win, 1000, 800)
    win.timeline.set_section(0.5, 1.5)
    before = sizes(win)
    for label in (win.loop_label, win.tempo_label, win.speed_label, win.summary_label, win.output_label):
        for text in ("Loop · Pass 1", "Loop · Pass 12 · last 100 %", LONG, "", "x"):
            label.setText(text)
            assert sizes(win) == before, (label.objectName(), text[:30])
    assert before[2] <= 1011       # the minimum width was 1011 on the real macOS style before this fix


def test_a_loop_through_several_passes_never_resizes_the_window(tmp_path):
    with echo_singer(tmp_path) as qa:
        w = qa.window
        w.show()
        qa.set("between_box", "Count-in")
        qa.set("bpm_box", 120)
        qa.mark_section(4.0, 8.0)
        qa.set("loop_box", True)
        before = sizes(w)
        qa.start_take()
        seen = {w.loop_label.text()}
        for n in (1, 2, 3):
            qa.wait_until(lambda: w.loop_label.text().startswith(f"Loop · Pass {n}"), timeout_s=30)
            seen.add(w.loop_label.text())
            assert sizes(w) == before, w.loop_label.text()
        qa.wait_until(lambda: "last" in w.loop_label.text(), timeout_s=30)
        assert sizes(w) == before
        # every status message the controls can put on the window while the loop goes on or after it
        for text in ("Jumped. Take saved: " + LONG, "Pass 3 of Chorus: In tune 77% " + LONG, LONG):
            w.summary_label.setText(text)
            assert sizes(w) == before
        for typed in (10, 0, 500, 143.6):                # BPM notes: "BPM is 30-300: 10 became 30", "0 or empty = ..."
            qa.set("bpm_box", typed)
            assert sizes(w) == before, w.tempo_label.text()
        qa.pause()
        assert w.pause_button.text() == "Play" and sizes(w) == before
        qa.pause()
        qa.stop_take()
        assert sizes(w) == before
        qa.click("replay_button")                     # "Stop replay" and the replayed Take's numbers
        qa.wait(0.5)
        assert w.replay_button.text() == "Stop replay" and sizes(w) == before
        qa.click("replay_button")
        assert sizes(w) == before


def test_the_full_text_of_an_elided_label_is_in_its_tooltip(qtbot, tmp_path):
    win = melody_window(qtbot, tmp_path)
    shown_at(qtbot, win, 1000, 800)
    win.tempo_label.setText(LONG)
    QtWidgets.QApplication.processEvents()
    assert win.tempo_label.text() == LONG                 # the full text is what the label holds
    assert LONG in win.tempo_label.toolTip()
    win.tempo_label.setText("short")
    QtWidgets.QApplication.processEvents()
    assert win.tempo_label.text() == "short" and LONG not in win.tempo_label.toolTip()


def test_the_section_name_field_keeps_a_usable_width_at_every_window_width(qtbot, tmp_path):
    win = melody_window(qtbot, tmp_path)
    shown_at(qtbot, win, 1, 1)                            # the smallest the window goes
    assert win.section_name_edit.width() >= 90
    win.timeline.set_section(0.5, 1.5)
    win.loop_box.setEnabled(True)
    win.loop_box.setChecked(True)
    win.loop_label.setText("Loop · Pass 12 · last 100 %")
    QtWidgets.QApplication.processEvents()
    assert win.section_name_edit.width() >= 90


def test_the_first_launch_size_gives_the_plot_room(qtbot, tmp_path):
    """The tester measured the plot at about 150 px at 1000x620 on the macOS style: it needs at least about 300 px."""
    win = melody_window(qtbot, tmp_path)
    win.show()
    qtbot.waitExposed(win)
    QtWidgets.QApplication.processEvents()
    assert win.height() >= 780
    assert win.plot_widget.height() >= 300
    assert win.minimumSizeHint().height() <= 560            # the minimum is not raised for it

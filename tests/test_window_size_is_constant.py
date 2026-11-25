"""#52 for every state, not case by case: the main window walks through every reachable state (all References, long
song, Section and device names in every combo box, every widget that is shown or hidden, the note read-out, Speed
preparation, calibration, Take, Pause, History) and its size(), minimumSize() and minimumSizeHint() never change after
first show, at the default size and at the smallest."""
import pytest
from PySide6 import QtWidgets
from size_walk import Watch, build, walk
from test_replay import saved_take


@pytest.mark.parametrize("small", [False, True], ids=["default size", "smallest size"])
def test_the_main_window_never_changes_size_or_minimum_size_in_any_state(qtbot, tmp_path, small):
    take = saved_take(tmp_path)
    win = build(tmp_path)
    qtbot.addWidget(win)
    if small:
        win.resize(1, 1)
    win.show()
    qtbot.waitExposed(win)
    QtWidgets.QApplication.processEvents()
    watch = Watch(win)
    walk(win, watch, take.id)
    assert watch.steps > 100
    assert not watch.broken, "\n".join(watch.broken)


def test_a_long_word_in_the_status_line_or_the_output_note_leaves_the_size_alone(qtbot, tmp_path):
    """#66: word-wrapped labels asked for their longest word's width; 400 characters made the window 3400 px wide."""
    win = build(tmp_path)
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    watch = Watch(win)
    for label in (win.summary_label, win.output_label):
        label.setText("Y" * 400)
        watch(f"{label.objectName()} 400 chars")
        label.setText("a b c " * 200)
        watch(f"{label.objectName()} 200 words")
    assert not watch.broken, "\n".join(watch.broken)
    assert win.summary_label.text() == "a b c " * 200 and win.summary_label.toolTip()    # whole text in the tooltip


def test_the_plot_keeps_a_usable_height_at_the_smallest_window_and_every_button_shows_its_text(qtbot, tmp_path):
    """#68 and the 72 px plot of the minimum window."""
    win = build(tmp_path)
    qtbot.addWidget(win)
    win.resize(1, 1)
    win.show()
    qtbot.waitExposed(win)
    QtWidgets.QApplication.processEvents()
    assert win.plot_widget.height() >= 200
    from PySide6 import QtWidgets as W
    for b in win.findChildren(W.QPushButton):
        if b.isVisible() and b.text():
            assert b.fontMetrics().horizontalAdvance(b.text()) + 6 <= b.width(), (b.text(), b.width())


def test_the_first_launch_size_fits_the_screen(qtbot, tmp_path, monkeypatch):
    from PySide6 import QtCore
    from sing_pitch.ui import main_window as mw

    class Screen:
        def availableGeometry(self):
            return QtCore.QRect(0, 25, 1280, 695)
    monkeypatch.setattr(mw.QtGui.QGuiApplication, "primaryScreen", staticmethod(lambda: Screen()))
    assert mw.MainWindow._first_size() == (1000, 695 - mw.SCREEN_MARGIN[1])
    monkeypatch.setattr(mw.QtGui.QGuiApplication, "primaryScreen", staticmethod(lambda: None))
    assert mw.MainWindow._first_size() == mw.DEFAULT_SIZE

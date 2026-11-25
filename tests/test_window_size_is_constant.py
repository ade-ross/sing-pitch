"""#52 for every state, not case by case: the main window walks through every reachable state (all References, long
song, Section and device names in every combo box, every widget that is shown or hidden, the note read-out, Speed
preparation, calibration, Take, Pause, History) and its size(), minimumSize() and minimumSizeHint() never change after
first show, at the default size and at the smallest."""
import pytest
from PySide6 import QtWidgets
from size_walk import Watch, build, walk


@pytest.mark.parametrize("small", [False, True], ids=["default size", "smallest size"])
def test_the_main_window_never_changes_size_or_minimum_size_in_any_state(qtbot, tmp_path, small):
    win = build(tmp_path)
    qtbot.addWidget(win)
    if small:
        win.resize(1, 1)
    win.show()
    qtbot.waitExposed(win)
    QtWidgets.QApplication.processEvents()
    watch = Watch(win)
    walk(win, watch)
    assert watch.steps > 100
    assert not watch.broken, "\n".join(watch.broken)

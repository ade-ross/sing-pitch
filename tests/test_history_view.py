import csv
from datetime import datetime

from test_history import row, write_table

from sing_pitch.history import WHOLE_SONG


def test_empty_table_shows_empty_state_and_does_not_crash(qtbot, tmp_path):
    from sing_pitch.ui.history_view import HistoryView

    v = HistoryView(tmp_path)
    qtbot.addWidget(v)
    assert "No Takes yet" in v.empty_label.text()
    assert not v.empty_label.isHidden() and v.plot_widget.isHidden()
    assert not v.grab().isNull()


def test_view_plots_both_metrics_and_filters(qtbot, tmp_path):
    from sing_pitch.ui.history_view import HistoryView

    write_table(tmp_path, [
        row("a", 10, pct="50", med="20"),
        row("b", 11, reference="melody", song="s1", pct="70", med="10"),
    ])
    v = HistoryView(tmp_path)
    qtbot.addWidget(v)
    assert v.empty_label.isHidden() and not v.plot_widget.isHidden()
    assert list(v.pct_curve.getData()[1]) == [50.0, 70.0]
    assert list(v.cents_curve.getData()[1]) == [20.0, 10.0]
    v.reference_box.setCurrentIndex(v.reference_box.findData("melody"))
    assert list(v.pct_curve.getData()[1]) == [70.0]
    assert v.song_box.isEnabled()
    v.song_box.setCurrentIndex(v.song_box.findData("s1"))
    assert list(v.cents_curve.getData()[1]) == [10.0]
    assert not v.grab().isNull()


def test_song_filter_disabled_without_melody_takes(qtbot, tmp_path):
    from sing_pitch.ui.history_view import HistoryView

    write_table(tmp_path, [row("a", 10)])
    v = HistoryView(tmp_path)
    qtbot.addWidget(v)
    assert not v.song_box.isEnabled()


def test_filter_matching_nothing_shows_empty_state(qtbot, tmp_path):
    from sing_pitch.ui.history_view import HistoryView

    write_table(tmp_path, [row("a", 10), row("b", 11, reference="key")])
    v = HistoryView(tmp_path)
    qtbot.addWidget(v)
    v.reference_box.setCurrentIndex(v.reference_box.findData("key"))
    v.song_box.setEnabled(True)  # not reachable normally; filter via the model instead
    v.show_series(v.history.series(reference="melody"))
    assert not v.empty_label.isHidden()


def test_main_window_opens_history(qtbot, tmp_path):
    from sing_pitch.pitch.swiftf0 import SwiftF0Tracker
    from sing_pitch.ui import MainWindow

    write_table(tmp_path, [row("a", 10)])
    win = MainWindow(SwiftF0Tracker(), home=tmp_path)
    qtbot.addWidget(win)
    win.history_button.click()
    assert win.history_window is not None and win.history_window.isVisible()
    win.history_window.close()


def test_guide_vocal_speed_and_section_filters_narrow_the_plot(qtbot, tmp_path):
    from sing_pitch.ui.history_view import HistoryView
    from test_history import practice_table

    practice_table(tmp_path)
    v = HistoryView(tmp_path)
    qtbot.addWidget(v)
    assert list(v.pct_curve.getData()[1]) == [40.0, 50.0, 60.0, 65.0, 30.0]
    v.guide_box.setCurrentIndex(v.guide_box.findData("off"))
    assert list(v.pct_curve.getData()[1]) == [50.0, 60.0, 30.0]
    v.speed_box.setCurrentIndex(v.speed_box.findData(70))
    assert list(v.pct_curve.getData()[1]) == [60.0]
    v.guide_box.setCurrentIndex(0)
    v.speed_box.setCurrentIndex(0)
    v.section_box.setCurrentIndex(v.section_box.findData("Chorus 1"))
    assert list(v.pct_curve.getData()[1]) == [50.0, 60.0]
    v.section_box.setCurrentIndex(v.section_box.findData(WHOLE_SONG))
    assert list(v.pct_curve.getData()[1]) == [40.0, 30.0]
    assert not v.grab().isNull()


def test_filter_boxes_keep_their_choice_over_a_refresh(qtbot, tmp_path):
    from sing_pitch.ui.history_view import HistoryView
    from test_history import practice_table

    practice_table(tmp_path)
    v = HistoryView(tmp_path)
    qtbot.addWidget(v)
    v.speed_box.setCurrentIndex(v.speed_box.findData(70))
    v.reload()
    assert v.speed_box.currentData() == 70 and list(v.pct_curve.getData()[1]) == [60.0, 65.0]


def test_the_cents_chart_shows_every_take_however_bad(qtbot, tmp_path):
    from sing_pitch.ui.history_view import HistoryView

    write_table(tmp_path, [row("a", 10, med="116"), row("b", 11, med="20"), row("c", 12, med="148")])
    v = HistoryView(tmp_path)
    qtbot.addWidget(v)
    top = v._cents_plot.viewRange()[1][1]
    assert top >= 148 and top <= 148 * 1.5          # the worst Take is inside, and the chart is not mostly empty


def test_the_cents_chart_never_zooms_in_below_fifty_cents(qtbot, tmp_path):
    from sing_pitch.ui.history_view import HistoryView

    write_table(tmp_path, [row("a", 10, med="4"), row("b", 11, med="6")])
    v = HistoryView(tmp_path)
    qtbot.addWidget(v)
    assert v._cents_plot.viewRange()[1] == [0, 50]


def test_the_cents_chart_follows_the_filter(qtbot, tmp_path):
    from sing_pitch.ui.history_view import HistoryView

    write_table(tmp_path, [row("a", 10, med="200"), row("b", 11, reference="melody", song="s1", med="30")])
    v = HistoryView(tmp_path)
    qtbot.addWidget(v)
    assert v._cents_plot.viewRange()[1][1] >= 200
    v.reference_box.setCurrentIndex(v.reference_box.findData("melody"))
    assert v._cents_plot.viewRange()[1] == [0, 50]

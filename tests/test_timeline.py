"""The Timeline widget alone: the melody drawn over the whole song, a playhead, a click gives the song time."""
import numpy as np
import pytest
from melody_helpers import reference_line, with_confidence
from PySide6 import QtCore

from sing_pitch.reference import Melody
from sing_pitch.take_engine import SampledClock

NOTES = [(-900, 0.6), (-500, 0.6), (-200, 0.8)]     # 2.0 s


def pixel_of(timeline, song_s):
    """Where in the widget the strip shows song time song_s (the plot area's mapping, not the widget's own)."""
    vb = timeline.plot.vb
    scene = vb.mapViewToScene(QtCore.QPointF(song_s, sum(vb.viewRange()[1]) / 2))
    return timeline.mapFromScene(scene)


@pytest.fixture
def timeline(qtbot):
    from sing_pitch.ui.timeline import Timeline

    t = Timeline()
    qtbot.addWidget(t)
    t.resize(800, 90)
    t.show()
    qtbot.waitExposed(t)
    return t


def test_it_draws_the_whole_melody_and_a_playhead_that_moves(timeline):
    timeline.set_melody(Melody(reference_line(NOTES), "x"), duration_s=4.0)
    x, y = timeline._line.getData()
    assert timeline.duration_s == 4.0 and x[0] == 0.0 and abs(x[-1] - 1.95) < 0.06      # thinned, whole song
    assert abs(np.nanmin(y) - 60) < 0.01 and abs(np.nanmax(y) - 67) < 0.01
    timeline.set_playhead(1.5)
    assert timeline.playhead_s == 1.5


def test_a_click_asks_for_the_song_time_under_the_mouse(qtbot, timeline):
    timeline.set_melody(Melody(reference_line(NOTES), "x"), duration_s=4.0)
    with qtbot.waitSignal(timeline.seek_requested, timeout=1000) as got:
        qtbot.mouseClick(timeline.viewport(), QtCore.Qt.LeftButton, pos=pixel_of(timeline, 3.0))
    assert abs(got.args[0] - 3.0) < 0.1
    with qtbot.waitSignal(timeline.seek_requested, timeout=1000) as got:
        qtbot.mouseClick(timeline.viewport(), QtCore.Qt.LeftButton, pos=pixel_of(timeline, 0.5))
    assert abs(got.args[0] - 0.5) < 0.1


def test_a_region_item_can_sit_on_the_strip_without_stealing_clicks_elsewhere(qtbot, timeline):
    """The hook for Sections (#13): a draggable region added to timeline.plot, a click outside it still seeks."""
    import pyqtgraph as pg

    timeline.set_melody(Melody(reference_line(NOTES), "x"), duration_s=4.0)
    region = pg.LinearRegionItem((1.0, 1.5))
    timeline.plot.addItem(region)
    with qtbot.waitSignal(timeline.seek_requested, timeout=1000) as got:
        qtbot.mouseClick(timeline.viewport(), QtCore.Qt.LeftButton, pos=pixel_of(timeline, 3.5))
    assert abs(got.args[0] - 3.5) < 0.1
    assert region.getRegion() == (1.0, 1.5)


def test_the_melody_can_be_sampled_at_any_song_times_for_drawing_on_a_clock():
    line = with_confidence(reference_line([(0, 0.6), (100, 0.6), (200, 0.6)]), 0.6, 1.2, 0.05)
    midi, unscored = Melody(line, "x").sample(np.array([-1.0, 0.3, 0.9, 1.5, 2.5, np.nan]))
    assert np.isnan(midi[[0, 2, 4, 5]]).all() and abs(midi[1] - 69.0) < 1e-6 and abs(midi[3] - 71.0) < 0.01
    assert list(unscored) == [False, False, True, False, False, False]      # only inside the unreliable stretch


def test_a_saved_takes_clock_follows_its_song_times_across_a_jump_and_a_pause():
    t = [i * 0.1 for i in range(40)]
    song = [10.0 + x if x < 1.0 else None if x < 2.0 else 3.0 + (x - 2.0) for x in t]
    clock = SampledClock(t, song)
    assert abs(clock(0.5) - 10.5) < 1e-9 and clock(1.5) is None and abs(clock(3.0) - 4.0) < 1e-9
    assert abs(clock(-0.2) - 9.8) < 1e-9 and abs(clock(4.5) - (3.0 + 1.9 + 0.6)) < 1e-9      # beyond the ends: slope 1


def test_a_saved_takes_clock_goes_on_at_its_last_rate_and_ends_with_the_song():
    t = [i * 0.1 for i in range(40)]                        # a Take at Speed 50 %: the song advances 0.5 s per second
    clock = SampledClock(t, [0.5 * x + 10.0 for x in t], song_end_s=13.0)
    assert abs(clock(3.9) - 11.95) < 1e-9
    assert abs(clock(5.9) - 12.95) < 1e-9                   # beyond the last frame at 0.5, not 1.0
    assert clock(6.1) is None                               # past the song end: no song
    assert abs(SampledClock(t, [0.5 * x + 10.0 for x in t])(5.9) - 12.95) < 1e-9      # rate also read without an end


def test_a_projection_from_past_the_loop_end_restarts_at_the_pass_start():
    from sing_pitch.take_engine import project_song_time

    # the song stands at 10.2 s, just past the Section end (10 s): the wrap is due, so it goes on from the Lead-in (6 s)
    got = project_song_time(np.array([0.0, 1.0, 5.0]), 10.2, 1.0, loop_end_s=10.0, loop_pass_start_s=6.0)
    assert np.allclose(got, [6.0, 7.0, 7.0])           # 5 s on: one 4 s Pass later, 1 s into the next


def test_a_melody_knows_where_its_line_ends():
    line = reference_line([(0, 1.0)])
    assert abs(Melody(line, "x").end_s - 1.0) < 0.011


def drag(timeline, from_s, to_s, steps=5):
    """A real left-button drag across the strip (press, moves with the button held, release)."""
    from PySide6 import QtGui, QtWidgets

    vp = timeline.viewport()
    a, b = pixel_of(timeline, from_s), pixel_of(timeline, to_s)

    def send(kind, pos, buttons):
        ev = QtGui.QMouseEvent(kind, QtCore.QPointF(pos), QtCore.QPointF(vp.mapToGlobal(pos)), QtCore.Qt.LeftButton,
                               buttons, QtCore.Qt.NoModifier)
        QtWidgets.QApplication.sendEvent(vp, ev)

    send(QtCore.QEvent.MouseButtonPress, a, QtCore.Qt.LeftButton)
    for i in range(1, steps + 1):
        p = QtCore.QPoint(round(a.x() + (b.x() - a.x()) * i / steps), a.y())
        send(QtCore.QEvent.MouseMove, p, QtCore.Qt.LeftButton)
    send(QtCore.QEvent.MouseButtonRelease, b, QtCore.Qt.NoButton)


def test_dragging_across_the_strip_marks_a_section_and_does_not_seek(qtbot, timeline):
    timeline.set_melody(Melody(reference_line(NOTES), "x"), duration_s=4.0)
    seeks = []
    timeline.seek_requested.connect(seeks.append)
    with qtbot.waitSignal(timeline.section_marked, timeout=1000) as got:
        drag(timeline, 2.5, 1.0)                           # right to left works too
    start, end = got.args
    assert abs(start - 1.0) < 0.1 and abs(end - 2.5) < 0.1 and seeks == []
    assert timeline.section == (start, end)


def test_a_section_set_from_outside_is_shown_with_edges_that_can_be_moved(qtbot, timeline):
    timeline.set_melody(Melody(reference_line(NOTES), "x"), duration_s=4.0)
    timeline.set_section(1.0, 2.0)
    assert timeline.section == (1.0, 2.0) and timeline.region.isVisible()
    with qtbot.waitSignal(timeline.section_changed, timeout=1000) as got:
        timeline.region.lines[1].setValue(3.0)             # what dragging the right handle does
        timeline.region.lines[1].sigPositionChangeFinished.emit(timeline.region.lines[1])
    assert got.args == [1.0, 3.0]
    timeline.set_section(None)
    assert timeline.section is None and not timeline.region.isVisible()


def test_an_edge_dragged_onto_the_other_keeps_the_section_at_least_the_minimum_long(qtbot, timeline):
    from sing_pitch.ui.timeline import MIN_SECTION_S

    timeline.set_melody(Melody(reference_line(NOTES), "x"), duration_s=4.0)
    timeline.set_section(1.0, 2.0)
    with qtbot.waitSignal(timeline.section_changed, timeout=1000) as got:
        timeline.region.lines[1].setValue(1.0)             # right handle dropped on the left one
        timeline.region.lines[1].sigPositionChangeFinished.emit(timeline.region.lines[1])
    start, end = got.args
    assert end - start >= MIN_SECTION_S - 1e-9 and 0.0 <= start and end <= 4.0
    assert timeline.section == (start, end)


def test_a_minimum_section_at_the_song_end_stays_inside_the_song(qtbot, timeline):
    from sing_pitch.ui.timeline import MIN_SECTION_S

    timeline.set_melody(Melody(reference_line(NOTES), "x"), duration_s=4.0)
    timeline.set_section(3.0, 4.0)
    with qtbot.waitSignal(timeline.section_changed, timeout=1000) as got:
        timeline.region.lines[0].setValue(4.0)
        timeline.region.lines[0].sigPositionChangeFinished.emit(timeline.region.lines[0])
    start, end = got.args
    assert end - start >= MIN_SECTION_S - 1e-9 and end <= 4.0

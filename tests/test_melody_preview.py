"""The Melody preview (#51 follow-up): with Reference = Melody and a song chosen, and no Take or replay running, the plot
shows the reference line, Guide lines, Unscored stretches and (Notes mode) bars around where Start would begin, with the
Tolerance band on it, and follows the Tolerance, Octave, Notes, song, Reference and Section controls at once."""
import numpy as np
from audio import harmonic, quiet
from frames import hz_at
from melody_helpers import library_song, reference_line, with_confidence
from test_melody_window import NOTES, window
from test_tolerance_band import assert_band_is, band_runs

from sing_pitch.settings import Settings


def song_window(qtbot, tmp_path, line=None, seconds=2.0, **kw):
    library_song(tmp_path, line if line is not None else reference_line(NOTES), seconds=seconds, **kw)
    s = Settings.load(tmp_path)
    s.set_latency("WF-1000XM5", 0.0)
    s.save()
    audio = np.concatenate([harmonic(hz_at(c), d) for c, d in NOTES] + [quiet(0.4)])
    win = window(qtbot, tmp_path, audio)
    win.reference_box.setCurrentText("Melody")
    return win


def line_of(win):
    x, y = win._ref_curve.getData()
    return (np.array([]), np.array([])) if x is None else (np.asarray(x), np.asarray(y, float))


def test_before_start_the_plot_shows_the_line_the_guide_lines_and_the_band_and_nothing_runs(qtbot, tmp_path):
    line = with_confidence(reference_line(NOTES), 0.6, 1.2, 0.05)
    win = song_window(qtbot, tmp_path, line)
    x, y = line_of(win)
    assert np.isfinite(y).sum() > 50
    for curve in win._guide_curves:
        assert np.isfinite(curve.getData()[1]).any()
    assert len(win._unscored.opts["x0"]) == 1                      # the Unscored stretch is greyed
    assert band_runs(win)
    assert_band_is(win, 25)
    t, midi = win._curve.getData()
    assert t is None or len(t) == 0                                # no voice, nothing scored
    assert len(win._dots.getData()[0]) == 0
    assert not win._timer.isActive() and win._runner is None
    lo, hi = win._plot.viewRange()[0]
    assert lo < win._now_line.value() < hi and abs(win._now_line.value()) < 1e-9     # centred on where Start begins


def test_the_band_follows_the_slider_with_no_take_ever_run_and_adds_no_items(qtbot, tmp_path):
    win = song_window(qtbot, tmp_path)
    items = len(win._plot.items)
    for cents in (10, 25, 100, 55, 10):
        win.tolerance_slider.setValue(cents)
        assert_band_is(win, cents)
    assert len(win._plot.items) == items and not win._timer.isActive()


def test_the_vertical_range_shows_the_previewed_stretch(qtbot, tmp_path):
    win = song_window(qtbot, tmp_path)
    _, y = line_of(win)
    lo, hi = win._plot.viewRange()[1]
    finite = y[np.isfinite(y)]
    assert lo < finite.min() and finite.max() < hi


def test_the_preview_is_gone_outside_melody(qtbot, tmp_path):
    win = song_window(qtbot, tmp_path)
    assert np.isfinite(line_of(win)[1]).any()
    win.reference_box.setCurrentText("Nearest note")
    assert not np.isfinite(line_of(win)[1]).any() and not band_runs(win)
    win.reference_box.setCurrentText("Melody")
    assert np.isfinite(line_of(win)[1]).any() and band_runs(win)


def test_a_pinned_octave_moves_the_preview_and_auto_previews_at_zero(qtbot, tmp_path):
    win = song_window(qtbot, tmp_path)
    _, auto = line_of(win)
    win.octave_box.setCurrentIndex(win.octave_box.findData(1))
    _, up = line_of(win)
    assert np.allclose(up[np.isfinite(up)], auto[np.isfinite(auto)] + 12)
    assert_band_is(win, 25)
    win.octave_box.setCurrentIndex(win.octave_box.findData(None))
    assert np.allclose(np.nan_to_num(line_of(win)[1], nan=-1), np.nan_to_num(auto, nan=-1))


def test_notes_mode_and_the_minimum_note_length_redraw_the_preview_at_once(qtbot, tmp_path):
    win = song_window(qtbot, tmp_path)
    _, plain = line_of(win)
    win.notes_box.setChecked(True)
    _, bars = line_of(win)
    assert np.isfinite(bars).any() and not np.allclose(np.nan_to_num(bars), np.nan_to_num(plain))
    assert_band_is(win, 25)
    runs = len(band_runs(win))
    assert runs >= 2 and all(np.ptp(c) < 1e-9 for _, c, _ in band_runs(win))        # flat bars, flat band
    win.tolerance_slider.setValue(100)
    assert_band_is(win, 100)
    win.note_length_slider.setValue(win.note_length_slider.maximum())
    win.note_length_slider.sliderReleased.emit()
    assert np.isfinite(line_of(win)[1]).any() and band_runs(win)


def test_a_chosen_or_marked_section_puts_the_preview_at_its_start(qtbot, tmp_path):
    win = song_window(qtbot, tmp_path)
    assert abs(win._song_times(np.array([0.0]))[0] - 0.0) < 1e-9
    win.timeline.set_section(1.0, 1.8)
    win._section_marked(1.0, 1.8)
    assert abs(win._song_times(np.array([0.0]))[0] - 1.0) < 1e-9
    assert [t for t in win._axis_seconds(np.array([0.0, 1.0]))] == [1.0, 2.0]       # and the axis says so
    assert_band_is(win, 25)
    win.clear_section()
    assert abs(win._song_times(np.array([0.0]))[0] - 0.0) < 1e-9


def test_a_song_change_redraws_the_preview(qtbot, tmp_path):
    library_song(tmp_path, reference_line([(-900, 1.0)]), title="Low", seconds=2.0)
    library_song(tmp_path, reference_line([(-100, 1.0)]), title="High", seconds=2.0)
    win = window(qtbot, tmp_path, quiet(1.0))
    win.reference_box.setCurrentText("Melody")
    win.song_box.setCurrentIndex(win.song_box.findText("Low"))
    low = np.nanmedian(line_of(win)[1])
    win.song_box.setCurrentIndex(win.song_box.findText("High"))
    assert np.nanmedian(line_of(win)[1]) > low + 5
    assert_band_is(win, 25)


def test_after_stop_the_take_stays_until_a_control_is_touched_and_the_tolerance_stays_locked_in_a_take(qtbot, tmp_path):
    win = song_window(qtbot, tmp_path)
    win.toggle()
    qtbot.waitUntil(lambda: len(win._t) > 20, timeout=3000)
    win.tick()
    assert not win.tolerance_slider.isEnabled()
    win.toggle()
    t, _ = win._curve.getData()
    assert t is not None and len(t) > 10                            # the Take just sung is still shown
    n = len(t)
    win.tick() if win._timer.isActive() else None
    assert len(win._curve.getData()[0]) == n
    win.tolerance_slider.setValue(60)                               # now the preview replaces it
    t, _ = win._curve.getData()
    assert t is None or len(t) == 0
    assert_band_is(win, 60)
    assert not win._timer.isActive()


def test_the_preview_costs_no_timer_and_a_redraw_is_cheap(qtbot, tmp_path):
    import time

    win = song_window(qtbot, tmp_path)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        win._draw_melody(0.0)
        times.append(time.perf_counter() - t0)
    assert np.median(times) < 0.05
    assert not win._timer.isActive()

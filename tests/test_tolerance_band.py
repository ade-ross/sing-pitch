"""The Tolerance as a soft band around the blue Melody line (#51): exactly ± the Tolerance around the line at the
active Octave shift (around the bars in Notes mode), broken where the line breaks, none on the Guide lines, and
following the slider at once (a replay shows the Tolerance its Take was saved with)."""
import numpy as np
from audio import harmonic, quiet
from frames import hz_at
from melody_helpers import library_song, reference_line, with_confidence
from test_melody_window import NOTES, window

from sing_pitch.settings import Settings


def band_runs(win):
    """The band as [(x, centre, half_width)] per drawn piece, read from the path the plot paints: each piece is the
    upper edge left to right, then the lower edge right to left."""
    out = []
    for poly in win._tolerance_band.path().toSubpathPolygons():
        pts = np.array([(p.x(), p.y()) for p in poly])
        if len(pts) % 2:
            pts = pts[:-1]                                     # Qt ends a closed polygon with one repeated point
        n = len(pts) // 2
        upper, lower = pts[:n], pts[n:][::-1]
        assert np.allclose(upper[:, 0], lower[:, 0])
        out.append((upper[:, 0], (upper[:, 1] + lower[:, 1]) / 2, (upper[:, 1] - lower[:, 1]) / 2))
    return out


def blue_line(win):
    """The Melody line as drawn: the blue curve, and the verdict-coloured bars (Notes mode) that left it."""
    x, y = win._ref_curve.getData()
    y = np.array(y, float)
    for curve in win._verdict_curves.values():
        v = curve.getData()[1]
        y = np.where(np.isnan(y), v, y)
    return np.asarray(x), y


def assert_band_is(win, cents):
    """Every piece of the band is exactly ± cents around the blue line, and covers it wherever it has two samples."""
    x, y = blue_line(win)
    runs = band_runs(win)
    assert runs
    covered = np.zeros(len(x), bool)
    for bx, centre, half in runs:
        idx = np.searchsorted(x, bx)
        assert np.array_equal(x[idx], bx)
        assert np.allclose(centre, y[idx]) and np.allclose(half, cents / 100)
        covered[idx] = True
    finite = np.isfinite(y)
    alone = finite & ~np.r_[False, finite[:-1]] & ~np.r_[finite[1:], False]      # a single sample has no width
    assert np.array_equal(covered, finite & ~alone)


def melody_take(qtbot, tmp_path, line=None, notes=False, stop=True):
    library_song(tmp_path, line if line is not None else reference_line(NOTES), seconds=2.0)
    s = Settings.load(tmp_path)
    s.set_latency("WF-1000XM5", 0.0)
    s.save()
    audio = np.concatenate([harmonic(hz_at(c), d) for c, d in NOTES] + [quiet(0.4)])
    win = window(qtbot, tmp_path, audio)
    win.reference_box.setCurrentText("Melody")
    win.notes_box.setChecked(notes)
    win.toggle()
    qtbot.waitUntil(lambda: len(win._t) > 20, timeout=3000)
    win.tick()
    if stop:
        win.toggle()
    return win


def test_the_band_is_exactly_the_tolerance_around_the_line_at_10_25_and_100_cents(qtbot, tmp_path):
    win = melody_take(qtbot, tmp_path)
    assert_band_is(win, 25)                                  # the default, as the Take was run
    for cents in (10, 25, 100):
        win.tolerance_slider.setValue(cents)                 # no Take running, no redraw tick: the band follows at once
        assert_band_is(win, cents)


def test_the_band_follows_the_line_at_a_pinned_octave_shift(qtbot, tmp_path):
    library_song(tmp_path, reference_line(NOTES), seconds=2.0)
    s = Settings.load(tmp_path)
    s.set_latency("WF-1000XM5", 0.0)
    s.save()
    audio = np.concatenate([harmonic(hz_at(c - 1200), d) for c, d in NOTES] + [quiet(0.4)])
    win = window(qtbot, tmp_path, audio)
    win.reference_box.setCurrentText("Melody")
    win.octave_box.setCurrentIndex(win.octave_box.findData(-1))
    win.toggle()
    qtbot.waitUntil(lambda: len(win._t) > 20, timeout=3000)
    win.tick()
    x, y = win._ref_curve.getData()
    assert np.nanmin(y) < 69 - 12 + 1                       # drawn an octave down, where the Take is scored
    assert_band_is(win, 25)
    win.toggle()


def test_the_band_follows_the_slider_during_a_take_too_it_is_the_tolerance_the_take_runs_at(qtbot, tmp_path):
    win = melody_take(qtbot, tmp_path, stop=False)
    assert not win.tolerance_slider.isEnabled()
    assert_band_is(win, 25)
    win.toggle()


def test_the_band_breaks_where_the_line_is_unscored(qtbot, tmp_path):
    line = with_confidence(reference_line(NOTES), 0.6, 1.2, 0.05)
    win = melody_take(qtbot, tmp_path, line)
    x, y = blue_line(win)
    gap = int(np.count_nonzero(np.isnan(y)))
    assert gap > 20
    for cents in (10, 100):
        win.tolerance_slider.setValue(cents)
        assert_band_is(win, cents)
        (x0,) = win._unscored.opts["x0"]
        (x1,) = win._unscored.opts["x1"]
        for bx, _, _ in band_runs(win):
            assert not ((bx > x0 + 0.02) & (bx < x1 - 0.02)).any()          # nothing over the Unscored stretch
        assert len(band_runs(win)) >= 2


def test_in_notes_mode_the_band_sits_around_the_bars_and_breaks_between_them(qtbot, tmp_path):
    win = melody_take(qtbot, tmp_path, notes=True)
    x, y = blue_line(win)
    assert 1 <= len(np.flatnonzero(np.diff(np.isfinite(y).astype(int)) == 1)) + int(np.isfinite(y[0])) <= 6
    for cents in (10, 25, 100):
        win.tolerance_slider.setValue(cents)
        assert_band_is(win, cents)
        for _, centre, _ in band_runs(win):
            assert np.ptp(centre) < 1e-9                      # a bar is flat, so is the band around it
    assert len(band_runs(win)) >= 2                            # one piece per bar on screen


def test_the_guide_lines_have_no_band(qtbot, tmp_path):
    win = melody_take(qtbot, tmp_path)
    x, y = blue_line(win)
    for bx, centre, half in band_runs(win):
        for curve, d in zip(win._guide_curves, (12, -12)):
            gy = curve.getData()[1][np.searchsorted(x, bx)]
            assert not np.allclose(centre, gy)                  # centred on the blue line, not on an octave copy
    top = max(r[1].max() + r[2].max() for r in band_runs(win))
    bottom = min(r[1].min() - r[2].min() for r in band_runs(win))
    assert top < np.nanmax(y) + 0.5 and bottom > np.nanmin(y) - 0.5      # nowhere near the copies 12 semitones off


def test_a_replay_shows_the_tolerance_its_take_was_saved_with_whatever_the_slider_says(qtbot, tmp_path):
    library_song(tmp_path, reference_line(NOTES), seconds=2.0)
    s = Settings.load(tmp_path)
    s.set_latency("WF-1000XM5", 0.0)
    s.save()
    audio = np.concatenate([harmonic(hz_at(c), d) for c, d in NOTES] + [quiet(0.4)])
    win = window(qtbot, tmp_path, audio)
    win.reference_box.setCurrentText("Melody")
    win.tolerance_slider.setValue(50)
    win.toggle()
    qtbot.waitUntil(lambda: len(win._t) > 20, timeout=3000)
    win.toggle()                                                # a Take saved at +-50
    win.tolerance_slider.setValue(10)
    win.toggle_replay()
    qtbot.waitUntil(lambda: len(win._t) > 20, timeout=4000)
    win.tick()
    assert_band_is(win, 50)
    win.tolerance_slider.setValue(80)
    assert_band_is(win, 50)                                     # the slider does not touch a replay's band
    win.toggle_replay()
    win.tolerance_slider.setValue(80)
    assert_band_is(win, 80)                                     # the replay is over: the slider's again


def test_the_band_left_on_the_plot_after_a_replay_is_his_own_tolerance_not_the_takes(qtbot, tmp_path):
    """#70 shows the Take's value on the slider during a replay and gives his back after; the band that stays on the
    plot when the replay stops (#71) must be his value too, with no slider touch needed."""
    library_song(tmp_path, reference_line(NOTES), seconds=2.0)
    s = Settings.load(tmp_path)
    s.set_latency("WF-1000XM5", 0.0)
    s.save()
    audio = np.concatenate([harmonic(hz_at(c), d) for c, d in NOTES] + [quiet(0.4)])
    win = window(qtbot, tmp_path, audio)
    win.reference_box.setCurrentText("Melody")
    win.tolerance_slider.setValue(50)
    win.toggle()
    qtbot.waitUntil(lambda: len(win._t) > 20, timeout=3000)
    win.toggle()                                                # a Take saved at +-50
    win.tolerance_slider.setValue(10)                           # his own value now
    win.toggle_replay()
    qtbot.waitUntil(lambda: len(win._t) > 20, timeout=4000)
    win.tick()
    win.toggle_replay()                                         # replay stopped, the plot stays
    assert win.tolerance_slider.value() == 10
    assert_band_is(win, 10)

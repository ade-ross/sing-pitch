"""The plot's vertical range (#34): it follows a smoothed centre of the voice AND the coming Reference at the active
Octave, and moves only when that content would leave the view (hysteresis). Pure rule first, then through the window."""
import numpy as np

from sing_pitch.ui.y_window import YWindow

HALF = 8


def line(lo, hi, n=100):
    return np.linspace(lo, hi, n)


def test_a_voice_sitting_near_the_centre_never_moves_the_view():
    y = YWindow(60, HALF)
    for med in (60, 61.5, 58.5, 62, 57):
        assert y.follow(np.full(50, med), np.array([])) is False
    assert y.centre == 60


def test_a_single_blip_in_the_voice_does_not_move_the_view():
    y = YWindow(60, HALF)
    voice = np.full(50, 60.0)
    voice[-1] = 78.0                              # the old rule jumped 18 semitones on the last sample
    assert y.follow(voice, np.array([])) is False and y.centre == 60


def test_a_voice_that_stays_away_moves_the_view_to_its_median():
    y = YWindow(60, HALF)
    assert y.follow(np.full(50, 68.0), np.array([])) is True
    assert y.centre == 68


def test_a_voice_with_a_few_notes_is_not_enough_to_decide():
    y = YWindow(60, HALF)
    assert y.follow(np.full(3, 80.0), np.array([])) is False and y.centre == 60


def test_the_coming_reference_inside_the_view_leaves_it_alone():
    y = YWindow(60, HALF)
    assert y.follow(np.full(50, 60.0), line(55, 65)) is False and y.centre == 60


def test_the_coming_reference_about_to_leave_the_view_moves_it_so_the_whole_of_it_is_in():
    y = YWindow(60, HALF)
    ahead = line(62, 69.4)                        # 69.4 is inside 52..68 only if the margin is ignored: it is not
    assert y.follow(np.full(50, 62.0), ahead) is True
    lo, hi = y.centre - HALF, y.centre + HALF
    assert lo < ahead.min() < ahead.max() < hi


def test_after_a_move_the_same_reference_does_not_move_it_again():
    y = YWindow(60, HALF)
    ahead = line(66, 74)
    assert y.follow(np.full(50, 66.0), ahead) is True
    centre = y.centre
    for _ in range(20):
        assert y.follow(np.full(50, 70.0), ahead) is False
    assert y.centre == centre


def test_the_voice_is_included_while_it_fits_with_the_reference():
    y = YWindow(60, HALF)
    ahead = line(66, 70)                          # alone it would centre on 68
    y.follow(np.full(50, 62.0), ahead)            # the voice at 62 fits with it: centre on 62..70
    assert y.centre == 66


def test_a_voice_far_from_the_reference_does_not_pull_the_reference_out_of_view():
    y = YWindow(60, HALF)
    ahead = line(70, 76)
    y.follow(np.full(50, 50.0), ahead)            # sings an octave off the shown line: the Reference wins
    assert y.centre - HALF < ahead.min() and ahead.max() < y.centre + HALF


def test_a_reference_wider_than_the_view_does_not_make_it_jump_around():
    y = YWindow(60, HALF)
    wide = line(50, 72)                           # 22 semitones in a 16 semitone view
    y.follow(np.full(50, 60.0), wide)
    centre = y.centre
    for shift in (0.3, 0.6, 0.0, -0.4):
        assert y.follow(np.full(50, 60.0), wide + shift) is False
    assert y.centre == centre and abs(centre - 61) <= 3


def test_a_stray_point_in_the_reference_is_ignored_but_a_phrase_is_not():
    y = YWindow(60, HALF)
    ahead = line(58, 64, 1000)
    ahead[10] = 90.0                              # one wild frame in 1000
    assert y.follow(np.full(50, 60.0), ahead) is False
    assert y.follow(np.full(50, 60.0), np.concatenate([ahead, np.full(60, 90.0)])) is True


def test_nan_and_empty_inputs_are_tolerated():
    y = YWindow(60, HALF)
    assert y.follow(np.array([np.nan] * 10), np.array([np.nan, np.nan])) is False
    assert y.follow(np.array([]), np.array([])) is False
    assert y.centre == 60


def test_a_wandering_voice_with_a_slowly_climbing_reference_recentres_rarely():
    y = YWindow(60, HALF)
    moves = 0
    rng = np.random.default_rng(0)
    for k in range(600):                          # 20 s at 30 ticks/s, the melody climbing 12 semitones
        base = 60 + 12 * k / 600
        ahead = line(base - 3, base + 3)
        voice = base + rng.normal(0, 0.7, 60)
        moves += y.follow(voice, ahead)
    assert 1 <= moves <= 4


# --- through the window ------------------------------------------------------------------------------------------

from test_centred_playhead import LATENCY, SONG_S, running, scores  # noqa: E402
from audio import SR  # noqa: E402


def settle(win):
    for _ in range(100):
        win._ease_y()


def view_y(win):
    return win._plot.getViewBox().viewRange()[1]


def test_a_climbing_melody_stays_in_view_with_a_few_moves_of_the_range(qtbot, tmp_path):
    win, player, log, offset = running(qtbot, tmp_path)
    log.append(round(SONG_S * SR), 0.0)
    win._set_y(60)
    moves, off_screen = 0, 0
    for now in np.arange(4.0, 20.0, 0.05):             # the song climbs a semitone a second: 57 + k at second k
        win._set_now(now)
        win._draw_melody(now)
        before = win._centre
        win._follow_y()
        moves += win._centre != before
        settle(win)
        lo, hi = view_y(win)
        ahead = win._ahead[np.isfinite(win._ahead)]
        assert len(ahead) > 100
        off_screen += int(((ahead < lo) | (ahead > hi)).any())
    assert off_screen == 0
    assert 1 <= moves <= 4


def test_the_range_pans_over_several_redraws_instead_of_jumping(qtbot, tmp_path):
    win, player, log, offset = running(qtbot, tmp_path)
    log.append(round(SONG_S * SR), 0.0)
    win._set_y(60)
    win._set_now(12.0)
    win._draw_melody(12.0)                             # the coming melody sits near MIDI 69-77, above the view
    win._follow_y()
    target = win._centre
    assert target > 62
    first = view_y(win)
    assert 60 - 12 < first[0] < target - 12            # started to move, has not arrived
    steps = 0
    while abs(win._y_drawn - target) > 1e-9 and steps < 100:
        win._ease_y()
        steps += 1
    assert 3 <= steps <= 30 and view_y(win)[0] == target - 12


def test_one_stray_voice_frame_does_not_move_the_range_of_a_nearest_note_window(qtbot, tmp_path):
    from audio import harmonic
    from frames import hz_at
    from sing_pitch.audio_input import ArrayInput
    from sing_pitch.pitch.swiftf0 import SwiftF0Tracker
    from sing_pitch.take_engine import FrameScore
    from sing_pitch.reference import Band
    from sing_pitch.ui import MainWindow

    win = MainWindow(SwiftF0Tracker(), home=tmp_path, input_factory=lambda _i: ArrayInput(harmonic(hz_at(0), 0.5), SR, 256))
    qtbot.addWidget(win)
    win.ingest(scores(np.arange(0.0, 3.0, 0.01)))      # steady at MIDI 60
    win.tick()
    before = view_y(win)
    blip = FrameScore(t=3.0, midi=78.0, cents=0.0, band=Band.GREEN, note_name="F#5")
    win.ingest([blip])
    win.tick()
    assert view_y(win) == before
    win.ingest([FrameScore(t=3.0 + 0.01 * k, midi=78.0, cents=0.0, band=Band.GREEN, note_name="F#5")
                for k in range(1, 300)])               # sung there for 3 s: now the view goes
    for _ in range(40):
        win.tick()
    assert view_y(win)[0] > before[0] + 8

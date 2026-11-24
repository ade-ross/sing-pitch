"""The centred playhead (#18): the plot shows [now - 4 s, now + 4 s] with a now-line in the middle; the voice only in the
left half, the Reference, Guide lines and Unscored bands in both halves, the right half drawn through the song clock
(extrapolated from what the Transport has rendered: it knows Speed, a pending loop wrap and the song end).

The window runs for real (array mic, fake player) but the song-time log is the test's own, so every scenario states
which song time plays when. Expected song times are worked out here from the scenario, never from the window's code."""
import numpy as np
from pytest import approx
from audio import SR, quiet
from melody_helpers import library_song, reference_line, with_confidence
from test_melody_window import FakeBacking

from sing_pitch.audio_input import ArrayInput
from sing_pitch.pitch.swiftf0 import SwiftF0Tracker
from sing_pitch.playback import SongTimeLog
from sing_pitch.playback.transport import Loop
from sing_pitch.reference import Band
from sing_pitch.settings import Settings
from sing_pitch.take_engine import FrameScore, SampledClock
from sing_pitch.ui import MainWindow

SONG_S = 24
NOTES = [(-1200 + 100 * k, 1.0) for k in range(SONG_S)]    # note k sounds from k to k + 1 s and is MIDI 57 + k
LATENCY = 0.2


def note_midi(song_s):   # (float32 reference line: compared with approx)
    return 57.0 + np.floor(song_s)


class PlayerStub(FakeBacking):
    duration_s = float(SONG_S)

    def __init__(self, path, vocals, log):
        super().__init__(path, vocals)
        self.song_log, self.loop, self.position = log, None, 0.0

    def position_s(self):
        return self.position


def running(qtbot, tmp_path, line=None):
    """A window mid-Take on a Melody. Returns (window, player, log, offset): voice time t reads the log at t - offset.
    The log starts empty (nothing rendered): each test appends what the player "rendered"."""
    library_song(tmp_path, line or reference_line(NOTES), seconds=float(SONG_S))
    settings = Settings.load(tmp_path)
    settings.set_latency("WF-1000XM5", LATENCY)
    settings.save()
    log, players = SongTimeLog(SR), []

    def make(path, vocals=None):
        players.append(PlayerStub(path, vocals, log))
        return players[-1]

    win = MainWindow(SwiftF0Tracker(), home=tmp_path, input_factory=lambda _i: ArrayInput(quiet(1.0), SR, 256),
                     backing_factory=make, output_info=lambda: ("WF-1000XM5", "headphones"))
    qtbot.addWidget(win)
    win.reference_box.setCurrentText("Melody")
    win.toggle()
    qtbot.waitUntil(lambda: hasattr(win._runner.song_clock, "output_at_input_zero_s"), timeout=3000)   # the LogClock
    win._timer.stop()                      # the test draws; nothing redraws behind its back
    win._t.clear(), win._midi.clear(), win._band.clear(), win._out.clear(), win._oct.clear()
    return win, players[0], log, win._draw_clock().offset_s


def drawn(win, now):
    win._draw_melody(now)
    x, midi = win._ref_curve.getData()
    return x, midi


def midi_at(win, now, voice_t):
    x, midi = drawn(win, now)
    return midi[int(np.argmin(np.abs(x - voice_t)))]


def song_frames(seconds):
    return seconds * SR            # the log counts song frames


def away_from_a_note_change(song):
    return abs(song - round(song)) > 0.05


def scores(ts):
    return [FrameScore(t=t, midi=60.0, cents=0.0, band=Band.GREEN, note_name="C4") for t in ts]


def test_the_plot_shows_four_seconds_either_side_of_now_with_a_line_in_the_middle(qtbot, tmp_path):
    win, *_ = running(qtbot, tmp_path)
    win.toggle()                                           # end the run; the view stays
    win._t.clear(), win._midi.clear(), win._band.clear(), win._out.clear(), win._oct.clear()
    win.ingest(scores(np.arange(0.0, 9.0, 0.01)))
    win.tick()
    now = win._t[-1]
    lo, hi = win._plot.getViewBox().viewRange()[0]
    assert abs(lo - (now - 4.0)) < 1e-6 and abs(hi - (now + 4.0)) < 1e-6
    assert abs(win._now_line.value() - now) < 1e-6
    assert abs((lo + hi) / 2 - win._now_line.value()) < 1e-6


def test_the_voice_is_drawn_only_up_to_now_and_only_the_last_four_seconds_are_kept(qtbot, tmp_path):
    win, *_ = running(qtbot, tmp_path)
    win.toggle()
    win._t.clear(), win._midi.clear(), win._band.clear(), win._out.clear(), win._oct.clear()
    win.ingest(scores(np.arange(0.0, 9.0, 0.01)))
    win.tick()
    now = win._t[-1]
    line_x, _ = win._curve.getData()
    dots_x, _ = win._dots.getData()
    for xs in (line_x, dots_x):
        assert xs.max() <= now + 1e-9 and xs.min() >= now - 4.0 - 0.011


def test_nearest_note_and_key_get_the_same_window_with_an_empty_right_half(qtbot, tmp_path):
    from audio import harmonic
    from frames import hz_at

    win = MainWindow(SwiftF0Tracker(), home=tmp_path,
                     input_factory=lambda _i: ArrayInput(harmonic(hz_at(0), 0.5), SR, 256))
    qtbot.addWidget(win)
    win.ingest(scores(np.arange(0.0, 5.0, 0.01)))
    win.tick()
    lo, hi = win._plot.getViewBox().viewRange()[0]
    assert abs((lo + hi) / 2 - win._now_line.value()) < 1e-6 and abs(win._now_line.value() - 4.99) < 1e-6
    assert win._curve.getData()[0].max() <= win._now_line.value() + 1e-9


def test_the_reference_runs_on_into_the_right_half_at_the_pitch_of_the_song_that_is_about_to_play(qtbot, tmp_path):
    win, player, log, offset = running(qtbot, tmp_path)
    log.append(round(10 * SR), 0.0)                       # 10 s rendered, song 0 -> 10 s, one stretch at Speed 100
    now = 10.0 + offset - LATENCY
    x, midi = drawn(win, now)
    assert abs(x[0] - (now - 4.0)) < 1e-9 and x[-1] >= now + 4.0 - 1e-9
    for voice_t in (now - 3.0, now - 0.5, now + 0.5, now + 1.7, now + 3.5):     # past and future alike
        song = voice_t - offset                           # linear song: song time = output time = voice time - offset
        if away_from_a_note_change(song):
            assert midi_at(win, now, voice_t) == approx(note_midi(song))
    assert np.isfinite(midi[x > now]).mean() > 0.9
    ref, up, down = win._ref_curve.getData()[1], *(c.getData()[1] for c in win._guide_curves)
    ok = np.isfinite(ref)
    assert np.allclose((up - ref)[ok], 12) and np.allclose((ref - down)[ok], 12)
    win.toggle()


def test_the_future_follows_the_speed_of_the_last_stretch(qtbot, tmp_path):
    win, player, log, offset = running(qtbot, tmp_path)
    log.append(round(8 * SR), 0.0, 0.5)                   # 8 s of output at 50 %: song 0 -> 4 s
    now = 8.0 + offset
    for voice_t in (now + 1.0, now + 2.6, now + 3.9):     # 1 output second = half a song second
        song = 4.0 + (voice_t - now) * 0.5
        if away_from_a_note_change(song):
            assert midi_at(win, now, voice_t) == approx(note_midi(song))
    win.toggle()


def test_the_future_starts_at_the_new_place_after_a_seek_even_inside_the_latency(qtbot, tmp_path):
    win, player, log, offset = running(qtbot, tmp_path)
    log.append(round(5 * SR), 0.0)                        # song 0 -> 5 s over output 0 -> 5 s
    log.append(round(0.4 * SR), song_frames(20.0))        # seek at output 5.0: song 20.0 -> 20.4 by output 5.4
    now = 5.0 + offset - 0.1                              # the voice still answers song time 4.9 (before the seek)
    assert abs(win._draw_clock()(now) - 4.9) < 1e-6
    checked = 0
    for voice_t in (now + 0.05, now + 0.5, now + 2.0, now + 3.9):
        output = voice_t - offset
        song = output if output < 5.0 else 20.0 + (output - 5.0)
        if away_from_a_note_change(song):
            assert midi_at(win, now, voice_t) == approx(note_midi(song)), (voice_t, song)
            checked += 1
    assert checked >= 3
    win.toggle()


def test_a_loop_wrap_in_the_right_half_shows_the_next_passes_lead_in(qtbot, tmp_path):
    win, player, log, offset = running(qtbot, tmp_path)
    player.loop = Loop(start_s=8.0, end_s=10.0, name="Chorus", lead_in_s=2.0)    # a Pass plays song 6 -> 10 s
    log.append(round(3 * SR), song_frames(6.0))           # Pass 1: 3 s in, song 9.0
    now = 3.0 + offset                                    # the end of the Section (song 10) is 1 s ahead

    def expected(dt):
        return 9.0 + dt if dt < 1.0 else 6.0 + (dt - 1.0) % 4.0

    checked = 0
    for dt in (0.3, 0.9, 1.2, 2.7, 3.9):
        if away_from_a_note_change(expected(dt)):
            assert midi_at(win, now, now + dt) == approx(note_midi(expected(dt))), dt
            checked += 1
    assert checked >= 4
    assert midi_at(win, now, now + 1.5) == approx(note_midi(6.5))     # the Lead-in, not the song after the Section
    win.toggle()


def test_a_loop_wrap_already_in_the_log_is_drawn_as_it_happened_and_the_next_one_is_predicted(qtbot, tmp_path):
    win, player, log, offset = running(qtbot, tmp_path)
    player.loop = Loop(start_s=8.0, end_s=10.0, lead_in_s=2.0)
    log.append(round(4 * SR), song_frames(6.0))           # Pass 1: output 0 -> 4 s, song 6 -> 10
    log.append(round(1.5 * SR), song_frames(6.0))         # Pass 2: output 4 -> 5.5 s, song 6 -> 7.5
    now = 5.5 + offset
    assert midi_at(win, now, now - 1.0) == approx(note_midi(6.5))          # 1 s ago: Pass 2
    assert midi_at(win, now, now - 2.0) == approx(note_midi(9.5))          # 2 s ago: still Pass 1
    assert midi_at(win, now, now + 2.4) == approx(note_midi(9.9))          # 2.5 s of Pass 2 left
    assert midi_at(win, now, now + 2.9) == approx(note_midi(6.4))          # Pass 3's Lead-in
    win.toggle()


def test_after_the_end_of_the_song_the_right_half_is_empty(qtbot, tmp_path):
    win, player, log, offset = running(qtbot, tmp_path)
    log.append(round(22 * SR), song_frames(1.0))          # song 1 -> 23 s: one second from the end of a 24 s song
    now = 22.0 + offset
    x, midi = drawn(win, now)
    assert midi_at(win, now, now + 0.5) == approx(note_midi(23.5))
    assert np.isnan(midi[x > now + 1.1]).all()
    win.toggle()


def test_paused_the_plot_stands_at_the_pause_and_the_right_half_shows_what_plays_on_resume(qtbot, tmp_path):
    win, player, log, offset = running(qtbot, tmp_path)
    log.append(round(3 * SR), 0.0)                        # song 0 -> 3 s
    log.append(round(2 * SR), None)                       # paused for 2 s (output 3 -> 5): the song waits at 3.0
    player.position = 3.0
    assert win._draw_clock()(5.0 + offset) is None        # the voice answers nothing now
    now = win._fold().to_view(5.0 + offset)
    assert now == approx(3.0 + offset)                    # the pause is folded away: the view stands at it (#20)
    assert midi_at(win, now, now - 0.5) == approx(note_midi(2.5))     # no gap behind the now-line
    for dt in (0.5, 1.7, 3.2):
        if away_from_a_note_change(3.0 + dt):
            assert midi_at(win, now, now + dt) == approx(note_midi(3.0 + dt))    # what plays on resume, Speed 100
    win.toggle()


def test_unscored_stretches_are_greyed_in_the_future_half_too(qtbot, tmp_path):
    line = with_confidence(reference_line(NOTES), 12.0, 13.0, 0.0)      # song 12 -> 13 s unreliable
    win, player, log, offset = running(qtbot, tmp_path, line=line)
    log.append(round(10 * SR), 0.0)
    now = 10.0 + offset
    win._draw_melody(now)
    x0, x1 = np.asarray(win._unscored.opts["x0"]), np.asarray(win._unscored.opts["x1"])
    assert len(x0) == 1
    assert abs(x0[0] - (12.0 + offset)) < 0.03 and abs(x1[0] - (13.0 + offset)) < 0.03 and x0[0] > now
    win.toggle()


def test_a_replay_draws_its_future_from_the_saved_takes_own_clock(qtbot, tmp_path):
    win, *_ = running(qtbot, tmp_path)
    win.toggle()
    win._melody = win._reference()
    win._replay_clock = SampledClock([0.0, 2.0, 2.01, 6.0], [0.0, 2.0, 10.0, 14.0])    # jumped from 2 to 10 s at t = 2
    assert midi_at(win, 1.0, 1.5) == approx(note_midi(1.5))
    assert midi_at(win, 1.0, 3.5) == approx(note_midi(11.5))      # a Take's clock knows its own future: the replay shows the jump

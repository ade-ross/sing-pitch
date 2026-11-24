"""Timing window (#41, QA #37): each voice frame is compared with the reference within +-100 ms of the song time it
answers, and the reference pitch closest to the voice counts. It forgives timing, not pitch; Unscored stretches never
count. Seam: score_take / TakeEngine with a Melody."""
import pytest
from melody_helpers import reference_line, voice_following

from sing_pitch.config import ScoreConfig
from sing_pitch.reference import Band, Melody
from sing_pitch.take_engine import LinearClock, TakeEngine, score_take

# a fast line: 0.2 s notes three semitones apart, so a voice 50 ms off is on the wrong note a quarter of the time
FAST = [(-900, 0.2), (-600, 0.2), (-300, 0.2), (0, 0.2), (-300, 0.2), (-600, 0.2), (-900, 0.2), (-600, 0.2)] * 3
STRICT = ScoreConfig(timing_window_s=0.0)


@pytest.mark.parametrize("off_s", [-0.1, -0.05, -0.025, 0.025, 0.05, 0.1])
def test_a_singer_on_pitch_but_up_to_100_ms_early_or_late_still_scores_in_tune(off_s):
    frames = voice_following(FAST, max(off_s, 0.0))
    early = max(-off_s, 0.0)               # early: each frame answers a song time later than the note he sings
    song_times = [f.t + early for f in frames]
    melody = Melody(reference_line(FAST), "fast")
    strict = score_take(frames, melody, STRICT, song_times).pct_in_tune
    windowed = score_take(frames, melody, song_times=song_times).pct_in_tune
    assert strict < 90.0                    # the old rule: timing scored as pitch
    assert windowed >= (98.0 if abs(off_s) < 0.1 else 94.0)    # at the window's edge the 10 ms grid loses a little


def test_the_window_is_100_ms_by_default():
    assert ScoreConfig().timing_window_s == pytest.approx(0.1)


def test_a_singer_250_ms_late_is_not_forgiven():
    scale = [(-1500 + 300 * k, 0.2) for k in range(8)]       # rising: no note comes back
    frames = voice_following(scale, 0.25)
    assert score_take(frames, Melody(reference_line(scale), "scale")).pct_in_tune < 40.0


def test_the_window_forgives_timing_not_pitch_a_voice_40_cents_sharp_stays_out_of_green():
    frames = voice_following(FAST, 0.05, cents_off=40)
    s = score_take(frames, Melody(reference_line(FAST), "fast"))
    assert s.pct_in_tune == 0.0
    assert 39.0 < s.median_abs_cents < 41.0


def test_live_frames_use_the_window_too():
    engine = TakeEngine(Melody(reference_line(FAST), "fast"), song_clock=LinearClock(0.0))
    scores = engine.push(voice_following(FAST, 0.05))
    scored = [s for s in scores if s.band is not None]
    assert len(scored) > 200 and all(s.band is Band.GREEN for s in scored)


def test_unscored_stretches_never_count_and_the_window_scores_no_extra_frame():
    from melody_helpers import with_confidence

    line = with_confidence(reference_line(FAST), 1.0, 2.0, 0.05)       # an Unscored stretch from 1 s to 2 s
    frames = voice_following(FAST, 0.05)
    strict = score_take(frames, Melody(line, "fast"), STRICT)
    windowed = score_take(frames, Melody(line, "fast"))
    assert windowed.n_scored == strict.n_scored            # a frame answering an Unscored stretch stays unscored
    engine = TakeEngine(Melody(line, "fast"))
    scores = engine.push(frames)
    assert all(s.band is None for s in scores if 1.02 <= s.t < 1.98)


@pytest.mark.parametrize("window_s", [0.001, 0.004])
def test_a_timing_window_smaller_than_the_frame_spacing_still_finds_the_nearest_frame(window_s):
    """Review 5 #7: a window narrower than the gap to the nearest frame used to raise All-NaN (nanargmin)."""
    melody = Melody(reference_line([(0, 0.5), (-300, 0.5)]), "two")
    target = melody.target(0.205, 69.0, window_s=window_s)          # halfway between frames 0.20 and 0.21
    assert target.midi == pytest.approx(69.0, abs=0.01)

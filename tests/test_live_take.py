"""Audio in -> real SwiftF0 tracker -> Take engine, no microphone: the whole non-UI path."""
import numpy as np
import pytest
from audio import SR, harmonic, noise, quiet
from frames import hz_at

from sing_pitch.live import LiveTake
from sing_pitch.pitch.swiftf0 import SwiftF0Tracker
from sing_pitch.reference import Band, NearestNote


@pytest.fixture(scope="module")
def tracker():
    return SwiftF0Tracker()


def sing(tracker, audio, chunk=512):
    take = LiveTake(tracker, NearestNote(), SR)
    scores = []
    for i in range(0, len(audio), chunk):
        scores += take.feed(audio[i : i + chunk])
    return scores, take.finish()


def bands_between(scores, t0, t1):
    return {s.band for s in scores if t0 <= s.t <= t1 and s.band is not None}


def test_tones_30_and_10_cents_sharp_with_noise_between_them(tracker):
    audio = np.concatenate([
        noise(0.6), harmonic(hz_at(30), 1.5), noise(0.8), harmonic(hz_at(10), 1.5), noise(0.6),
    ])
    scores, record = sing(tracker, audio)
    assert bands_between(scores, 0.9, 2.0) == {Band.YELLOW}        # +30 cents
    assert bands_between(scores, 3.2, 4.3) == {Band.GREEN}         # +10 cents
    assert bands_between(scores, 0.0, 0.4) == set()                # noise: not drawn
    assert bands_between(scores, 2.3, 2.8) == set()                # noise between notes: not drawn
    assert 3.0 < record.summary.n_voiced * 0.016 < 3.2             # 3.0 s of tone is the only voiced time
    assert record.summary.tendency == "sharp"
    assert abs(record.summary.median_abs_cents - 20) < 12          # |30| and |10| halves


def test_digital_silence_gives_an_empty_summary(tracker):
    scores, record = sing(tracker, quiet(2.0))
    assert all(s.band is None for s in scores)
    assert record.summary.n_scored == 0 and record.summary.pct_in_tune is None


def test_tracker_lag_stays_inside_the_display_budget(tracker):
    """Audio-time lag of the newest scored frame behind the audio fed so far (tracker side of the 0.25 s goal)."""
    audio = np.concatenate([quiet(0.3), harmonic(hz_at(0), 3.0)])
    take = LiveTake(tracker, NearestNote(), SR)
    lags = []
    for i in range(0, len(audio), 512):
        got = take.feed(audio[i : i + 512])
        if got and got[-1].t > 0.6:
            lags.append((i + 512) / SR - got[-1].t)
    assert lags and max(lags) < 0.25


def test_runner_makes_a_take_from_start_to_stop_and_saves_it(tracker, tmp_path):
    from sing_pitch.audio_input import ArrayInput
    from sing_pitch.live import LiveRunner
    from sing_pitch.take_store import load_frames

    audio = np.concatenate([quiet(0.4), harmonic(hz_at(30), 2.0), quiet(0.4)])
    runner = LiveRunner(ArrayInput(audio, SR, chunk=256), tracker, NearestNote(), home=tmp_path)
    runner.start()                      # ArrayInput delivers everything at once
    record = runner.stop()

    live = runner.drain()               # the scores the UI would have drawn
    assert {s.band for s in live if s.band} == {Band.YELLOW}
    assert record.summary.tendency == "sharp"
    assert (tmp_path / "takes.csv").exists()
    assert len(load_frames(tmp_path, record.id)) == len(record.frames) > 100

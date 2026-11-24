"""Time-stretch for Speed: pitch stays, length scales. Real Rubber Band on a few seconds of a harmonic tone."""
import math

import numpy as np
import pytest
from audio import SR, harmonic

from sing_pitch.pitch.swiftf0 import SwiftF0Tracker
from sing_pitch.playback.stretch import time_stretch

SECONDS = 3.0


@pytest.fixture(scope="module")
def tracker():
    return SwiftF0Tracker()


def median_cents(tracker, mono: np.ndarray, ref_hz: float = 220.0) -> float:
    """Median pitch of the confidently voiced frames, in cents from ref_hz, ignoring the first and last 0.3 s."""
    tracker.reset()
    frames = tracker.push(mono, SR) + tracker.flush()
    end = len(mono) / SR
    voiced = [f.hz for f in frames if f.confidence > 0.6 and 0.3 < f.t < end - 0.3]
    assert len(voiced) > 50
    return 1200 * math.log2(float(np.median(voiced)) / ref_hz)


def stereo(mono: np.ndarray) -> np.ndarray:
    return np.stack([mono, mono], axis=1)           # (frames, 2): the layout of the player's arrays


@pytest.mark.parametrize("speed", [0.5, 0.7, 1.2])
def test_a_stretched_220_hz_tone_keeps_its_pitch_and_has_the_original_length_over_the_speed(tracker, speed):
    tone = harmonic(220.0, SECONDS)
    out = time_stretch(stereo(tone), SR, speed)
    assert out.shape[1] == 2 and out.dtype == np.float32
    assert abs(len(out) - len(tone) / speed) <= 1               # exact to the frame
    # the detector has a bias of its own on this tone: compare with the unstretched measurement, within a few cents
    assert abs(median_cents(tracker, out[:, 0]) - median_cents(tracker, tone)) < 5.0
    assert abs(median_cents(tracker, out[:, 0])) < 15.0


def test_speed_one_returns_the_audio_unchanged():
    x = stereo(harmonic(220.0, 0.5))
    assert np.array_equal(time_stretch(x, SR, 1.0), x)

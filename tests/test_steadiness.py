import math

from frames import hz_at, silence, sung
from steadiness_helpers import contour, vibrato

from sing_pitch.reference import NearestNote
from sing_pitch.take_engine import TakeEngine


def take(frames):
    engine = TakeEngine(NearestNote())
    engine.push(frames)
    return engine.stop()


def test_a_steady_held_note_has_steadiness_near_zero():
    s = take(sung(hz_at(0), 1.5)).summary
    assert s.steadiness_cents is not None and s.steadiness_cents < 0.5


def test_a_scoop_at_the_attack_does_not_worsen_steadiness():
    # 100 ms slide up from 60 cents flat, then a steady held note
    scoop = contour(lambda s: -60 * (1 - s / 0.1) if s < 0.1 else 0.0, 1.5)
    s = take(scoop).summary
    assert s.steadiness_cents < 0.5


def test_a_scoop_would_matter_if_it_were_not_trimmed():
    # the trim is what protects Steadiness: a slow 300 ms slide reaches into the stable middle
    slow = contour(lambda s: -40 * (1 - s / 0.5) if s < 0.5 else 0.0, 1.5)
    assert take(slow).summary.steadiness_cents > 3


def test_six_hz_vibrato_of_40_cents_has_sd_about_28():
    s = take(vibrato(40, 6.0, 2.0)).summary
    assert abs(s.steadiness_cents - 40 / math.sqrt(2)) < 2.0


def test_each_note_is_measured_around_its_own_median():
    # two steady notes, 200 cents apart and each 20 cents sharp of its own pitch class: spread within notes is 0
    frames = sung(hz_at(20), 1.0) + sung(hz_at(220), 1.0, start=1.0)
    assert take(frames).summary.steadiness_cents < 0.5


def test_notes_shorter_than_half_a_second_are_left_out():
    frames = vibrato(40, 6.0, 0.4) + silence(0.3, start=0.4) + sung(hz_at(0), 1.0, start=0.7)
    s = take(frames).summary
    assert s.steadiness_cents < 0.5          # the wobbly 0.4 s note does not count


def test_no_held_note_gives_no_steadiness():
    assert take(sung(hz_at(0), 0.4)).summary.steadiness_cents is None


def test_consistently_20_cents_sharp_is_sharp_and_still_steady():
    s = take(sung(hz_at(20), 1.5)).summary
    assert s.tendency == "sharp"
    assert s.steadiness_cents < 0.5


def sung_through_the_tracker(audio):
    import numpy as np
    from audio import SR

    from sing_pitch.live import LiveTake
    from sing_pitch.pitch.swiftf0 import SwiftF0Tracker

    live = LiveTake(SwiftF0Tracker(), NearestNote(), SR)
    for i in range(0, len(audio), 256):
        live.feed(audio[i:i + 256])
    return live.finish().summary


def test_synthesised_voice_held_20_cents_sharp_reads_sharp_and_steady():
    from audio import harmonic, quiet
    from frames import hz_at
    import numpy as np

    s = sung_through_the_tracker(np.concatenate([quiet(0.3), harmonic(hz_at(20), 2.0), quiet(0.3)]))
    assert s.tendency == "sharp"
    assert s.steadiness_cents < 3.0


def test_synthesised_voice_with_vibrato_has_the_vibrato_spread():
    import numpy as np
    from audio import SR, quiet

    t = np.arange(round(2.0 * SR)) / SR
    cents = 30 * np.sin(2 * np.pi * 6.0 * t)
    f0 = 440.0 * 2 ** (cents / 1200)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voice = sum(np.sin(k * phase) / k for k in range(1, 21))
    voice = (0.3 * voice / np.max(np.abs(voice))).astype(np.float32)
    s = sung_through_the_tracker(np.concatenate([quiet(0.3), voice, quiet(0.3)]))
    assert abs(s.steadiness_cents - 30 / math.sqrt(2)) < 5.0

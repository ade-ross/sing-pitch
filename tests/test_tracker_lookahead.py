"""The live tracker's look-ahead: SwiftF0 releases a frame once LOOKAHEAD_FRAMES later frames exist (the library waits
for 10, ~197 ms). Measured on a real vocal and on it one and two octaves down (2026-11-25, notes in
~/Builds/.notes/sing-pitch/latency/lookahead/): 6 frames releases at ~133 ms with 0.7 % of frames over 50 cents
from the 10-frame result on the lowest clip; fewer frames cost accuracy fast (4: 3.1 %, 3: 4.1 %)."""
import numpy as np
import pytest
from frames import hz_at

from sing_pitch.pitch import swiftf0
from sing_pitch.pitch.swiftf0 import LOOKAHEAD_FRAMES, SwiftF0Tracker

SR = 48_000
BLOCK = 256


def sung(notes_cents, note_s=0.6, sr=SR):
    """A voice-like tone: a few harmonics, a little vibrato, one note per entry (cents from A4)."""
    out = []
    for c in notes_cents:
        t = np.arange(int(note_s * sr)) / sr
        f = hz_at(c) * 2 ** (12 * np.sin(2 * np.pi * 5.5 * t) / 1200)
        phase = 2 * np.pi * np.cumsum(f) / sr
        out.append(sum(0.3 / k * np.sin(k * phase) for k in range(1, 6)))
    return np.concatenate(out).astype(np.float32)


def stream(tracker, audio):
    frames, ages, pushed = [], [], 0
    for i in range(0, len(audio) - BLOCK + 1, BLOCK):
        new = tracker.push(audio[i:i + BLOCK], SR)
        pushed += BLOCK
        frames += new
        ages += [pushed / SR - f.t for f in new]
    return frames, np.array(ages)


@pytest.fixture(scope="module")
def tracker():
    return SwiftF0Tracker()


def test_the_live_tracker_looks_six_frames_ahead():
    assert LOOKAHEAD_FRAMES == 6


def test_a_frame_is_released_about_133_ms_after_its_moment(tracker):
    tracker.reset()
    _, ages = stream(tracker, sung([-1200, -900, -500]))          # A3 C4 E4
    assert 0.120 <= np.median(ages) <= 0.145                      # 10 frames would be ~0.197


def test_at_ten_frames_it_gives_exactly_what_the_library_stream_gives(tracker, monkeypatch):
    """Guards the copy of the library's release step against a library change."""
    audio = sung([-2100, -1700, -1200])                            # C3 E3 A3: a low voice
    monkeypatch.setattr(swiftf0, "LOOKAHEAD_FRAMES", 10)
    tracker.reset()
    ours, _ = stream(tracker, audio)
    lib = tracker._detector.stream(fmin=swiftf0.FMIN_HZ, fmax=swiftf0.FMAX_HZ)
    theirs = []
    for i in range(0, len(audio) - BLOCK + 1, BLOCK):
        r = lib.push(audio[i:i + BLOCK], SR)
        theirs += list(zip(r.timestamps, r.pitch_hz, r.confidence))
    assert len(ours) == len(theirs)
    assert np.allclose([(f.t, f.hz, f.confidence) for f in ours], theirs)


def test_six_frames_agree_with_ten_on_a_low_sung_line(tracker, monkeypatch):
    audio = sung([-2700, -2400, -2100, -1700, -2100])               # F#2 A2 C3 E3 C3: the bottom of a man's range
    tracker.reset()
    short, _ = stream(tracker, audio)
    monkeypatch.setattr(swiftf0, "LOOKAHEAD_FRAMES", 10)
    tracker.reset()
    full, _ = stream(tracker, audio)
    n = min(len(short), len(full))
    a = np.array([(f.hz, f.confidence) for f in short[:n]])
    b = np.array([(f.hz, f.confidence) for f in full[:n]])
    voiced = (a[:, 1] >= 0.6) & (b[:, 1] >= 0.6)
    assert voiced.mean() > 0.8
    off = 1200 * np.abs(np.log2(a[voiced, 0] / b[voiced, 0]))
    assert np.mean(off > 50) < 0.02 and np.median(off) < 2.0


def test_flush_still_releases_every_frame(tracker):
    tracker.reset()
    audio = sung([-900])
    frames, _ = stream(tracker, audio)
    frames += tracker.flush()
    assert len(frames) == len(audio) * 16_000 // SR // 256

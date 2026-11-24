"""Melody Takes from audio: a synthetic voice singing the reference melody late by the latency, through the real
SwiftF0 tracker and the live glue (with a fake Backing player standing in for the headphones)."""
import numpy as np
import pytest
from audio import SR, harmonic, quiet
from frames import hz_at
from melody_helpers import reference_line

from sing_pitch.audio_input import ArrayInput
from sing_pitch.live import LiveRunner, LiveTake
from sing_pitch.pitch.swiftf0 import SwiftF0Tracker
from sing_pitch.playback import SongTimeLog
from sing_pitch.reference import Melody
from sing_pitch.take_engine import LinearClock

MELODY = [(-900, 0.6), (-700, 0.6), (-500, 0.6), (-400, 0.6), (-200, 0.6), (-500, 0.6), (-900, 1.0)]  # C D E F G E C


@pytest.fixture(scope="module")
def tracker():
    return SwiftF0Tracker()


def voice_audio(delay_s: float, tail_s: float = 0.4) -> np.ndarray:
    """The melody as a harmonic voice, starting delay_s after the first input sample. Notes are joined without
    gaps, like a legato line."""
    return np.concatenate([quiet(delay_s)] + [harmonic(hz_at(c), s) for c, s in MELODY] + [quiet(tail_s)])


def test_audio_voice_on_the_melody_late_by_the_latency_scores_about_100_in_tune(tracker):
    latency = 0.21
    take = LiveTake(tracker, Melody(reference_line(MELODY), "do-re-mi"), SR, song_clock=LinearClock(latency))
    audio = voice_audio(latency)
    for i in range(0, len(audio), 256):
        take.feed(audio[i : i + 256])
    s = take.finish().summary
    assert s.n_scored * 0.016 > 4.0
    assert s.pct_in_tune >= 95.0          # SwiftF0 smears ~1-2 frames at each of the 6 note changes
    assert s.median_abs_cents < 5.0


class FakeBacking:
    """Stands in for the Backing player: its song-time log shows how far the song already is when the input starts."""

    def __init__(self, position_at_input_start: float):
        self.position = position_at_input_start
        self.song_log = SongTimeLog(SR)
        self.song_log.append(round(position_at_input_start * SR), 0.0)
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def position_s(self) -> float:
        return self.position

    def stop(self):
        self.stopped = True


def test_runner_corrects_for_the_backing_starting_before_the_input(tracker, tmp_path):
    # The Backing was 0.08 s into the song when the mic started, and the calibrated loop latency is 0.25 s,
    # so on the input clock the voice is only 0.17 s behind the song.
    backing = FakeBacking(0.08)
    audio = voice_audio(0.25 - 0.08)
    runner = LiveRunner(ArrayInput(audio, SR, 256), tracker, Melody(reference_line(MELODY), "do-re-mi"), tmp_path,
                        backing=backing, latency_s=0.25)
    runner.start()
    record = runner.stop()
    assert backing.started and backing.stopped
    assert abs(record.latency_offset_s - 0.17) < 0.01
    assert record.song_id == "do-re-mi"
    assert record.summary.pct_in_tune >= 95.0

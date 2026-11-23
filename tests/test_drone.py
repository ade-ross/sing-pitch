import numpy as np
import pytest
from audio import SR

from sing_pitch.playback import Drone, DronePlayer, drone_midi
from sing_pitch.reference import Key


def peak_hz(x, sr=SR):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / sr)[int(np.argmax(spec))]


def render(drone, seconds=1.0, block=1024):
    n = int(seconds * SR)
    return np.concatenate([drone.render(min(block, n - i)) for i in range(0, n, block)])


@pytest.mark.parametrize("key,hz", [("C major", 130.81), ("A minor", 220.0), ("F# major", 185.0), ("Bb minor", 233.08)])
def test_drone_sounds_the_key_note_in_a_singable_octave(key, hz):
    audio = render(Drone(Key.parse(key).tonic, SR))
    assert abs(peak_hz(audio[SR // 4 :]) - hz) < 1.5


def test_drone_midi_stays_in_one_octave_c3_to_b3():
    assert [drone_midi(pc) for pc in range(12)] == list(range(48, 60))


def test_drone_is_audible_but_not_loud_and_never_clips():
    audio = render(Drone(0, SR))
    assert 0.05 < np.abs(audio).max() <= 0.5
    assert audio.dtype == np.float32


def test_drone_fades_in_without_a_click_and_is_phase_continuous_across_blocks():
    d = Drone(9, SR)
    a, b = d.render(48), d.render(48000)
    assert abs(a[0]) < 1e-3                                  # starts from silence
    whole = render(Drone(9, SR), seconds=1.0, block=1000)
    assert np.allclose(whole, render(Drone(9, SR), seconds=1.0, block=333), atol=1e-6)
    assert np.abs(np.diff(b)).max() < 0.1                    # no jump between samples


def test_stopping_fades_to_silence_and_reports_finished():
    d = Drone(0, SR)
    render(d, 0.5)
    d.release()
    tail = render(d, 0.3)
    assert not d.sounding and np.abs(tail[-1000:]).max() == 0.0


class FakeStream:
    def __init__(self, **kw):
        self.kw, self.started, self.closed = kw, False, False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


def test_player_opens_an_output_stream_whose_callback_plays_the_key_note():
    streams = []
    player = DronePlayer(stream_factory=lambda **kw: streams.append(FakeStream(**kw)) or streams[-1])
    player.start(Key.parse("A minor"))
    s = streams[0]
    assert s.started and s.kw["channels"] == 2 and s.kw["samplerate"] == SR
    out = np.zeros((SR, 2), dtype=np.float32)
    s.kw["callback"](out, SR, None, None)
    assert abs(peak_hz(out[SR // 4 :, 0]) - 220.0) < 1.5
    assert np.array_equal(out[:, 0], out[:, 1])
    player.stop()
    assert player.playing is False


def test_player_stop_without_start_is_harmless_and_start_twice_replaces_the_drone():
    streams = []
    player = DronePlayer(stream_factory=lambda **kw: streams.append(FakeStream(**kw)) or streams[-1])
    player.stop()
    player.start(Key.parse("C major"))
    player.start(Key.parse("D major"))
    assert streams[0].closed and streams[1].started
    player.stop()
    assert streams[1].closed


def test_player_survives_a_missing_output_device():
    def broken(**kw):
        raise OSError("no output device")

    player = DronePlayer(stream_factory=broken)
    with pytest.raises(OSError):
        player.start(Key.parse("C major"))
    assert player.playing is False

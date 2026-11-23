"""Backing playback without an audio device: a fake output stream drives the callback by hand."""
import numpy as np
import soundfile as sf

from sing_pitch.playback import BackingPlayer


class FakeStream:
    def __init__(self, **kwargs):
        self.kwargs, self.started, self.closed = kwargs, False, False

    def start(self):
        self.started = True

    def stop(self):
        pass

    def close(self):
        self.closed = True

    def pull(self, frames: int) -> np.ndarray:
        out = np.full((frames, self.kwargs["channels"]), 9.0, np.float32)    # junk: the callback must overwrite it
        self.kwargs["callback"](out, frames, None, None)
        return out


def player_with(audio, sr=48000):
    streams = []

    def factory(**kw):
        streams.append(FakeStream(**kw))
        return streams[-1]

    return BackingPlayer(audio, sr, stream_factory=factory, sample_rate=48000), streams


def test_backing_plays_the_song_in_order_and_counts_the_song_clock():
    song = np.stack([np.arange(3000, dtype=np.float32) / 3000] * 2, axis=1)
    player, streams = player_with(song)
    assert player.position_s() == 0.0
    player.start()
    a, b = streams[0].pull(1024), streams[0].pull(1024)
    assert np.allclose(np.concatenate([a, b]), song[:2048])
    assert player.position_s() == 2048 / 48000


def test_backing_pads_with_silence_after_the_song_ends_and_says_it_finished():
    player, streams = player_with(np.ones((1500, 2), np.float32))
    player.start()
    streams[0].pull(1024)
    tail = streams[0].pull(1024)
    assert np.all(tail[:476] == 1.0) and np.all(tail[476:] == 0.0)
    assert player.finished


def test_a_mono_44k1_backing_is_resampled_to_the_stream_rate_and_played_on_both_ears(tmp_path):
    t = np.arange(44100) / 44100
    path = tmp_path / "backing.wav"
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 44100)
    player = BackingPlayer.from_file(path, stream_factory=FakeStream, sample_rate=48000)
    assert abs(player.duration_s - 1.0) < 1e-3
    player.start()
    out = player._stream.pull(48000)
    assert np.allclose(out[:, 0], out[:, 1])
    spec = np.abs(np.fft.rfft(out[:, 0]))
    assert abs(np.fft.rfftfreq(48000, 1 / 48000)[np.argmax(spec)] - 440) < 2


def test_stop_closes_the_stream_and_is_safe_twice():
    player, streams = player_with(np.zeros((100, 2), np.float32))
    player.start()
    player.stop()
    player.stop()
    assert streams[0].closed

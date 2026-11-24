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


def player_with_vocal(backing, vocal, **kw):
    streams = []

    def factory(**k):
        streams.append(FakeStream(**k))
        return streams[-1]

    return BackingPlayer(backing, 48000, vocal=vocal, stream_factory=factory, sample_rate=48000, **kw), streams


def test_the_guide_vocal_is_off_by_default_and_the_backing_plays_as_before():
    player, streams = player_with_vocal(np.full((3000, 2), 0.5, np.float32), np.full(3000, 0.25, np.float32))
    player.start()
    assert np.allclose(streams[0].pull(1024), 0.5)


def test_gains_mix_the_vocal_stem_linearly_with_the_backing():
    player, streams = player_with_vocal(np.full((8192, 2), 0.5, np.float32), np.full((8192, 2), 0.25, np.float32))
    player.set_gains(backing=0.5, vocal=2.0)
    player.start()                                        # start() takes the gains as they are: no ramp from silence
    assert np.allclose(streams[0].pull(1024), 0.5 * 0.5 + 0.25 * 2.0)


def test_a_gain_change_while_playing_applies_from_the_next_block_with_a_ramp_and_no_step():
    player, streams = player_with_vocal(np.zeros((8192, 2), np.float32), np.ones((8192, 2), np.float32))
    player.set_gains(backing=1.0, vocal=0.0)
    player.start()
    assert np.allclose(streams[0].pull(1024), 0.0)
    player.set_gains(backing=1.0, vocal=1.0)              # guide vocal switched on mid-Take
    ramp = streams[0].pull(1024)[:, 0]
    assert ramp[0] < 0.01 and abs(ramp[-1] - 1.0) < 1e-6 and np.max(np.diff(ramp)) < 0.002
    assert np.allclose(streams[0].pull(1024), 1.0)
    player.set_gains(backing=1.0, vocal=0.0)              # and off again: silent after one block
    down = streams[0].pull(1024)[:, 0]
    assert abs(down[0] - 1.0) < 0.002 and down[-1] == 0.0
    assert np.allclose(streams[0].pull(1024), 0.0)


def test_the_vocal_stem_is_mono_resampled_and_aligned_to_the_backing():
    player, streams = player_with_vocal(np.zeros((2000, 2), np.float32), np.ones(1000, np.float32))
    player.set_gains(backing=1.0, vocal=1.0)
    player.start()
    out = streams[0].pull(2048)
    assert np.allclose(out[:1000], 1.0) and np.allclose(out[1000:], 0.0)       # shorter vocal pads with silence
    assert np.allclose(out[:, 0], out[:, 1])


def test_from_file_loads_the_vocal_stem_beside_the_backing(tmp_path):
    sf.write(str(tmp_path / "backing.wav"), np.full((4800, 2), 0.5, np.float32), 48000)
    sf.write(str(tmp_path / "vocals.wav"), np.full(4800, 0.25, np.float32), 48000)
    player = BackingPlayer.from_file(tmp_path / "backing.wav", tmp_path / "vocals.wav", stream_factory=FakeStream,
                                     sample_rate=48000)
    player.set_gains(backing=1.0, vocal=1.0)
    player.start()
    assert np.allclose(player._stream.pull(1024), 0.75)


def test_the_player_pauses_and_resumes_through_its_transport_and_logs_song_time():
    song = np.stack([np.arange(4096, dtype=np.float32) / 4096] * 2, axis=1)
    player, streams = player_with(song)
    player.start()
    streams[0].pull(1024)
    player.pause()
    assert np.all(streams[0].pull(1024) == 0.0) and player.position_s() == 1024 / 48000
    player.play()
    assert np.allclose(streams[0].pull(1024), song[1024:2048])
    log = player.song_log
    assert log.rendered_s == 3072 / 48000
    assert log.song_time(1500 / 48000) is None and abs(log.song_time(2500 / 48000) - 1476 / 48000) < 1e-9

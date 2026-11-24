"""The Transport's pure core: block request in, audio block + song time of each rendered frame out. No sound card."""
import numpy as np

from sing_pitch.playback.transport import Transport

SR = 1000      # a small rate keeps the numbers readable: frame i of the song is at song time i / 1000 s


def ramp_song(frames: int) -> np.ndarray:
    """A stereo 'song' whose sample value is its own frame index, so every block shows where it came from."""
    return np.stack([np.arange(frames, dtype=np.float32)] * 2, axis=1)


def test_playing_renders_the_song_in_order_with_the_song_time_of_every_frame():
    tr = Transport(ramp_song(3000), sample_rate=SR)
    a, b = tr.render(100), tr.render(100)
    assert np.array_equal(a.audio[:, 0], np.arange(100)) and np.array_equal(b.audio[:, 0], np.arange(100, 200))
    assert np.allclose(b.song_t, np.arange(100, 200) / SR)
    assert tr.position_s == 0.2


def test_a_paused_transport_renders_silence_with_no_song_time_and_resumes_where_it_stopped():
    tr = Transport(ramp_song(3000), sample_rate=SR)
    tr.render(100)
    tr.pause()
    paused = tr.render(50)
    assert not tr.playing and np.all(paused.audio == 0.0) and np.all(np.isnan(paused.song_t))
    assert tr.position_s == 0.1
    tr.play()
    after = tr.render(50)
    assert after.audio[0, 0] == 100.0 and np.isclose(after.song_t[0], 0.1)


def test_the_song_time_log_answers_for_any_output_time_rendered_so_far():
    tr = Transport(ramp_song(3000), sample_rate=SR)
    tr.render(100)                       # output 0.00-0.10 s plays song 0.00-0.10 s
    tr.pause()
    tr.render(200)                       # output 0.10-0.30 s: paused
    tr.play()
    tr.render(100)                       # output 0.30-0.40 s plays song 0.10-0.20 s
    log = tr.log
    assert log.rendered_s == 0.4
    assert np.isclose(log.song_time(0.05), 0.05)
    assert log.song_time(0.2) is None
    assert np.isclose(log.song_time(0.35), 0.15)
    assert np.isclose(log.song_time(-0.02), -0.02)     # before the first block: the first stretch, extended back
    assert np.isclose(log.song_time(0.45), 0.25)       # not rendered yet: the last stretch, extended forward


def test_pause_and_resume_are_reported_as_events_at_the_block_where_they_take_effect():
    tr = Transport(ramp_song(3000), sample_rate=SR)
    assert tr.render(100).events == ()
    tr.pause()
    (ev,) = tr.render(100).events
    assert (ev.kind, ev.output_s, ev.song_t) == ("paused", 0.1, 0.1)
    tr.play()
    (ev,) = tr.render(100).events
    assert (ev.kind, ev.output_s, ev.song_t) == ("resumed", 0.2, 0.1)


def test_gains_mix_backing_and_guide_vocal_and_change_from_the_next_block():
    backing = np.ones((1000, 2), np.float32)
    vocal = np.full((1000, 2), 2.0, np.float32)
    tr = Transport(backing, vocal, sample_rate=SR)
    assert np.allclose(tr.render(10).audio, 1.0)              # default: Backing 1, Guide vocal 0 (off)
    tr.set_gains(0.5, 0.5)
    ramp = tr.render(10).audio[:, 0]
    assert np.isclose(ramp[-1], 0.5 + 1.0) and ramp[0] > 1.0   # ramped within the block, ends on the target
    assert np.allclose(tr.render(10).audio, 1.5)


def test_after_the_song_ends_it_renders_silence_and_says_finished():
    tr = Transport(np.ones((150, 2), np.float32), sample_rate=SR)
    tr.render(100)
    tail = tr.render(100)
    assert np.all(tail.audio[:50] == 1.0) and np.all(tail.audio[50:] == 0.0)
    assert tr.finished and np.isclose(tail.song_t[-1], 0.199)

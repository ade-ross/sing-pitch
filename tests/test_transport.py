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


# --- Speed: the Transport plays pre-stretched arrays and keeps song time in original seconds -----------------------

def stretched(song: np.ndarray, speed: float) -> np.ndarray:
    """Stand-in for the time-stretcher: frame j of the result is the song's frame j * speed."""
    n = round(len(song) / speed)
    return song[np.minimum(np.round(np.arange(n) * speed).astype(int), len(song) - 1)]


def test_at_70_percent_song_time_advances_seven_tenths_of_a_second_per_output_second():
    song = ramp_song(3000)
    tr = Transport(song, sample_rate=SR)
    tr.set_speed(0.7, stretched(song, 0.7))
    a, b = tr.render(100), tr.render(100)
    assert a.events == ()                                          # set before the first block: no change to report
    assert np.allclose(b.song_t, (70 + np.arange(100) * 0.7) / SR)
    assert np.allclose(b.audio[:, 0], np.round((100 + np.arange(100)) * 0.7))      # the stretched array, in order
    assert np.isclose(tr.position_s, 0.14) and tr.speed == 0.7
    assert np.isclose(tr.log.song_time(0.15), 0.105)
    assert np.isclose(tr.duration_s, 3.0)                          # still the song's own length


def test_a_speed_change_mid_song_continues_from_the_same_song_time_and_is_reported_as_an_event():
    song = ramp_song(3000)
    tr = Transport(song, sample_rate=SR)
    tr.render(200)
    tr.set_speed(0.5, stretched(song, 0.5))
    block = tr.render(100)
    (ev,) = block.events
    assert (ev.kind, ev.output_s, ev.song_t) == ("speed_changed", 0.2, 0.2)
    assert np.isclose(block.song_t[0], 0.2) and np.isclose(block.song_t[1] - block.song_t[0], 0.5 / SR)
    assert np.isclose(tr.position_s, 0.25)
    assert np.isclose(tr.log.song_time(0.1), 0.1) and np.isclose(tr.log.song_time(0.25), 0.225)
    tr.set_speed(1.0, song)
    after = tr.render(10)
    assert np.isclose(after.audio[-1, 0], 259, atol=1) and np.isclose(tr.position_s, 0.26)


def test_the_first_block_after_a_speed_change_crossfades_so_it_does_not_click():
    song = np.ones((3000, 2), np.float32)
    tr = Transport(song, sample_rate=SR)
    tr.render(100)
    tr.set_speed(0.5, np.full((6000, 2), -1.0, np.float32))        # a stretch whose content differs from the old one
    blend = tr.render(100).audio[:, 0]
    assert blend[0] > 0.9 and blend[-1] < -0.9 and np.all(np.diff(blend) < 0)
    assert np.all(tr.render(100).audio == -1.0)


def test_a_speed_change_while_paused_applies_without_sound_and_the_song_ends_by_its_own_length():
    song = ramp_song(300)
    tr = Transport(song, sample_rate=SR)
    tr.render(50)
    tr.pause()
    tr.set_speed(0.5, stretched(song, 0.5))
    tr.render(10)
    tr.play()
    tr.render(10)
    assert not tr.finished and np.isclose(tr.position_s, 0.055)
    while not tr.finished:
        tr.render(100)
    assert tr.position_s >= 0.3

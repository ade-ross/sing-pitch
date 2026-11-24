"""Section loop in the Transport's pure core: a looped Section wraps from its end to start - Lead-in, block-exactly,
and reports each Pass. Song frame i has sample value i, so every frame shows where it came from."""
import numpy as np

from sing_pitch.playback.transport import Loop, Transport

SR = 1000


def ramp_song(frames: int) -> np.ndarray:
    return np.stack([np.arange(frames, dtype=np.float32)] * 2, axis=1)


def kinds(block):
    return [(e.kind, e.pass_index) for e in block.events]


def test_turning_the_loop_on_goes_to_start_minus_lead_in_and_starts_pass_1():
    tr = Transport(ramp_song(10000), sample_rate=SR)
    tr.render(100)
    tr.set_loop(Loop(3.0, 4.0, "Chorus", lead_in_s=2.0))
    b = tr.render(100)
    assert kinds(b) == [("pass_started", 1)]
    assert b.events[0].output_s == 0.1 and b.events[0].song_t == 1.0 and b.events[0].loop.name == "Chorus"
    assert b.audio[0, 0] == 1000.0 and np.isclose(b.song_t[0], 1.0)


def test_the_loop_wraps_from_the_section_end_to_start_minus_lead_in_inside_a_block():
    tr = Transport(ramp_song(10000), sample_rate=SR)
    tr.set_loop(Loop(3.0, 4.0, lead_in_s=2.0))
    tr.render(2950)                          # 1.000 .. 3.950
    b = tr.render(100)                       # 3.950 .. 3.999 then 1.000 .. 1.049
    assert np.array_equal(b.audio[:50, 0], np.arange(3950, 4000))
    assert np.array_equal(b.audio[50:, 0], np.arange(1000, 1050))
    assert np.allclose(b.song_t[50:], np.arange(1000, 1050) / SR)
    assert kinds(b) == [("pass_ended", 1), ("pass_started", 2)]
    assert all(np.isclose(e.output_s, 3.0) and np.isclose(e.song_t, 1.0) for e in b.events[1:])
    # the log knows both sides of the wrap
    assert np.isclose(tr.log.song_time(2.99), 3.99) and np.isclose(tr.log.song_time(3.01), 1.01)


def test_three_wraps_give_passes_1_to_4_and_the_lead_in_is_clamped_at_the_song_start():
    tr = Transport(ramp_song(10000), sample_rate=SR)
    tr.set_loop(Loop(0.5, 1.5, lead_in_s=2.0))     # lead-in would start at -1.5: it starts at 0
    first = tr.render(64)
    assert first.song_t[0] == 0.0 and kinds(first) == [("pass_started", 1)]
    started = []
    for _ in range(80):                            # 80 * 64 = 5.12 s of output = 3 full Passes of 1.5 s + a bit
        started += [e for e in tr.render(64).events if e.kind == "pass_started"]
    assert [e.pass_index for e in started] == [2, 3, 4]
    assert np.allclose([e.output_s for e in started], [1.5, 3.0, 4.5])


def test_turning_the_loop_off_lets_the_song_run_on_and_says_the_loop_ended():
    tr = Transport(ramp_song(10000), sample_rate=SR)
    tr.set_loop(Loop(3.0, 4.0, lead_in_s=2.0))
    tr.render(500)
    tr.set_loop(None)
    b = tr.render(3000)
    assert kinds(b) == [("pass_ended", 1), ("loop_ended", None)]
    assert np.array_equal(b.audio[:, 0], np.arange(1500, 4500))      # straight through the old Section end


def test_a_seek_during_the_loop_ends_the_loop():
    tr = Transport(ramp_song(10000), sample_rate=SR)
    tr.set_loop(Loop(3.0, 4.0, lead_in_s=2.0))
    tr.render(500)
    tr.seek(6.0)
    b = tr.render(2000)
    assert [e.kind for e in b.events] == ["pass_ended", "loop_ended", "jumped"]
    assert np.array_equal(b.audio[:, 0], np.arange(6000, 8000)) and tr.loop is None


def test_at_half_speed_a_pass_lasts_twice_as_long_and_still_wraps_on_the_section_end():
    tr = Transport(ramp_song(10000), sample_rate=SR)
    tr.set_speed(0.5, ramp_song(20000))            # a fake 'stretched' array: index i holds value i
    tr.set_loop(Loop(3.0, 4.0, lead_in_s=2.0))
    starts, t = [], []
    for _ in range(70):                            # 7 s of output: a Pass is 3 s of song = 6 s of output
        b = tr.render(100)
        starts += [e.output_s for e in b.events if e.kind == "pass_started"]
        t.append(b.song_t)
    t = np.concatenate(t)
    assert np.allclose(starts, [0.0, 6.0]) and t.max() < 4.0 and np.isclose(t[6000], 1.0)

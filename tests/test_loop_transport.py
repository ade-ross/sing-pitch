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


# --- a Speed change during the loop applies at the next Pass (ticket #15) -------------------------------------------

def test_a_speed_change_mid_loop_lets_the_current_pass_finish_at_the_old_speed_and_the_next_use_the_new_one():
    tr = Transport(ramp_song(10000), sample_rate=SR)
    tr.set_loop(Loop(3.0, 4.0, lead_in_s=2.0))      # a Pass: song 1.0 .. 4.0 = 3000 frames at Speed 1
    tr.render(500)
    tr.set_speed(0.5, ramp_song(20000))
    mid = tr.render(1500)                           # still inside Pass 1
    assert [e.kind for e in mid.events] == [] and tr.speed == 1.0
    assert np.allclose(np.diff(mid.song_t), 1 / SR)
    b = tr.render(1500)                             # 1000 frames of Pass 1 left: 1000 old-Speed frames, then Pass 2
    assert kinds(b) == [("pass_ended", 1), ("speed_changed", None), ("pass_started", 2)]
    assert all(np.isclose(e.output_s, 3.0) and np.isclose(e.song_t, 1.0) for e in b.events[1:])
    assert b.events[1].speed == 0.5 and b.events[1].loop is not None
    assert np.allclose(np.diff(b.song_t[:1000]), 1 / SR)
    assert np.isclose(b.song_t[1000], 1.0) and np.allclose(np.diff(b.song_t[1000:]), 0.5 / SR)
    assert tr.speed == 0.5
    assert np.isclose(tr.log.song_time(2.9), 3.9) and np.isclose(tr.log.song_time(3.1), 1.05)


def test_a_speed_asked_for_in_a_loop_that_ends_first_applies_at_the_next_block_without_a_loop():
    tr = Transport(ramp_song(10000), sample_rate=SR)
    tr.set_loop(Loop(3.0, 4.0, lead_in_s=2.0))
    tr.render(500)
    tr.set_speed(0.5, ramp_song(20000))
    tr.set_loop(None)
    tr.render(100)                                  # loop_ended is reported here; the Speed waits
    b = tr.render(100)
    assert [e.kind for e in b.events] == ["speed_changed"] and b.events[0].loop is None and tr.speed == 0.5


def test_outside_a_loop_a_speed_change_still_applies_at_the_next_block():
    tr = Transport(ramp_song(10000), sample_rate=SR)
    tr.render(100)
    tr.set_speed(0.5, ramp_song(20000))
    b = tr.render(100)
    assert [e.kind for e in b.events] == ["speed_changed"] and b.events[0].speed == 0.5 and tr.speed == 0.5


def test_a_gain_change_is_reported_at_the_next_block_with_the_new_gains():
    tr = Transport(ramp_song(10000), sample_rate=SR, gains=(1.0, 0.0))
    assert tr.render(100).events == ()
    tr.set_gains(0.5, 0.8)
    b = tr.render(100)
    assert [(e.kind, e.gains) for e in b.events] == [("gains_changed", (0.5, 0.8))]
    assert np.isclose(b.events[0].output_s, 0.1)
    assert tr.render(100).events == ()


def render_with_timeout(tr, frames, seconds=3.0):
    """render() in a daemon thread: a spinning render fails the test instead of hanging the suite."""
    import threading

    out = []
    worker = threading.Thread(target=lambda: out.append(tr.render(frames)), daemon=True)
    worker.start()
    worker.join(seconds)
    assert out, "render() did not return: the loop spins"
    return out[0]


def test_an_empty_section_is_widened_and_render_returns():
    tr = Transport(ramp_song(10000), sample_rate=SR)
    tr.set_loop(Loop(0.0, 0.0, lead_in_s=2.0))
    b = render_with_timeout(tr, 1000)
    assert len(b.audio) == 1000 and tr.loop is not None and tr.loop.end_s > tr.loop.pass_start_s


def test_a_section_after_the_song_end_is_moved_inside_the_song_and_render_returns():
    tr = Transport(ramp_song(10000), sample_rate=SR)          # 10 s
    tr.set_loop(Loop(20.0, 25.0, lead_in_s=2.0))              # Pass start 18 s, past the song
    b = render_with_timeout(tr, 1000)
    assert len(b.audio) == 1000
    assert 0.0 <= tr.loop.start_s < tr.loop.end_s <= 10.0


def test_a_section_whose_lead_in_starts_at_the_song_end_renders():
    tr = Transport(ramp_song(10000), sample_rate=SR)
    tr.set_loop(Loop(12.0, 12.0, lead_in_s=2.0))              # start - Lead-in == song length
    assert len(render_with_timeout(tr, 1000).audio) == 1000


def test_a_seek_reports_which_loop_its_pass_ended():
    tr = Transport(ramp_song(10000), sample_rate=SR)
    section = Loop(3.0, 4.0, "Chorus", lead_in_s=2.0)
    tr.set_loop(section)
    tr.render(500)
    tr.seek(6.0)
    ended = next(e for e in tr.render(100).events if e.kind == "pass_ended")
    assert ended.loop == section and ended.pass_index == 1


# --- the Pass counter (ticket #23) ----------------------------------------------------------------------------------

def test_pass_index_counts_every_wrap_and_is_zero_without_a_loop():
    tr = Transport(ramp_song(10000), sample_rate=SR)
    assert tr.pass_index == 0
    tr.set_loop(Loop(3.0, 4.0, lead_in_s=2.0))      # a Pass: song 1.0 .. 4.0 = 3000 frames
    seen = []
    for _ in range(20):
        tr.render(500)
        seen.append(tr.pass_index)
    assert seen[0] == 1 and seen[-1] == 4 and seen == sorted(seen) and set(seen) == {1, 2, 3, 4}
    tr.set_loop(None)
    tr.render(100)
    assert tr.pass_index == 0


def test_a_seek_inside_the_looping_section_jumps_there_as_the_next_pass_and_keeps_looping():
    """#42: a click inside the Section ends the Pass playing and starts the next one from there (no Lead-in); at the
    Section end it wraps to start - Lead-in as usual."""
    tr = Transport(ramp_song(10000), sample_rate=SR)
    tr.set_loop(Loop(3.0, 4.0, "Chorus", lead_in_s=2.0))
    tr.render(2500)                                  # song 1.0 .. 3.5, Pass 1
    tr.seek(3.2)
    b = tr.render(1000)                              # 3.2 .. 3.999 then the wrap to 1.0
    assert kinds(b) == [("pass_ended", 1), ("pass_started", 2), ("pass_ended", 2), ("pass_started", 3)]
    jump = b.events[1]
    assert np.isclose(jump.output_s, 2.5) and np.isclose(jump.song_t, 3.2) and jump.loop.name == "Chorus"
    assert np.array_equal(b.audio[:800, 0], np.arange(3200, 4000))
    assert np.array_equal(b.audio[800:, 0], np.arange(1000, 1200))
    assert tr.loop is not None and tr.pass_index == 3


def test_a_seek_into_the_lead_in_ends_the_loop():
    tr = Transport(ramp_song(10000), sample_rate=SR)
    tr.set_loop(Loop(3.0, 4.0, lead_in_s=2.0))
    tr.render(1500)
    tr.seek(2.0)                                     # before the Section start: outside it
    b = tr.render(100)
    assert [e.kind for e in b.events] == ["pass_ended", "loop_ended", "jumped"] and tr.loop is None


def test_through_the_player_a_seek_inside_the_section_keeps_the_loop_and_one_outside_drops_it():
    from sing_pitch.playback.backing import BackingPlayer

    player = BackingPlayer(ramp_song(10000), SR, stream_factory=lambda **kw: None, sample_rate=SR)
    loop = Loop(3.0, 4.0, lead_in_s=2.0)
    player.set_loop(loop)
    player.seek(3.5)
    assert player.loop == loop
    player.seek(6.0)
    assert player.loop is None


def test_inside_or_outside_a_section_is_decided_on_the_fitted_loop():
    """Review 5 #4: a Section past the song end loops up to the song end; one shorter than MIN_LOOP_S is widened.
    seek() says whether the loop goes on, judged on that fitted loop, and the next block does exactly that."""
    tr = Transport(ramp_song(3000), sample_rate=SR)               # a 3 s song
    tr.set_loop(Loop(2.5, 4.0, lead_in_s=0.5))                    # fitted: 2.5 .. 3.0
    tr.render(100)
    assert tr.seek(2.8) is True
    assert kinds(tr.render(100)) == [("pass_ended", 1), ("pass_started", 2)] and tr.loop is not None
    short = Transport(ramp_song(10000), sample_rate=SR)
    short.set_loop(Loop(5.0, 5.05, lead_in_s=0.5))                # widened: 5.0 .. 5.2
    short.render(100)
    assert short.seek(5.15) is True                               # outside the drawn Section, inside the loop
    short.render(10)
    assert short.loop is not None and 5.15 <= short.position_s < 5.2


def test_a_seek_past_the_fitted_loop_ends_it():
    tr = Transport(ramp_song(10000), sample_rate=SR)
    tr.set_loop(Loop(5.0, 5.05, lead_in_s=0.5))
    tr.render(100)
    assert tr.seek(5.3) is False
    assert [k for k, _ in kinds(tr.render(100))] == ["pass_ended", "loop_ended", "jumped"] and tr.loop is None


def test_a_seek_inside_a_loop_asked_for_in_the_same_block_starts_its_pass_1_there():
    tr = Transport(ramp_song(10000), sample_rate=SR)
    tr.render(100)
    tr.set_loop(Loop(3.0, 4.0))
    assert tr.seek(3.5) is True
    b = tr.render(100)
    assert kinds(b) == [("pass_started", 1)] and tr.loop is not None and b.audio[0, 0] == 3500.0
    tr.set_loop(None)
    assert tr.seek(3.5) is False                                  # no loop asked for any more: the seek ends it
    tr.render(100)
    assert tr.loop is None


def test_naming_the_looping_section_does_not_restart_the_loop_the_next_pass_carries_the_name():
    """Saving the unsaved mark that is looping only gives it a name: the song goes on where it is (paused or not),
    the Pass playing keeps its number and the next Pass is tagged with the name."""
    tr = Transport(ramp_song(10000), sample_rate=SR)
    tr.set_loop(Loop(2.0, 4.0))
    for _ in range(12):
        tr.render(100)                                           # Pass 1, at 1.2 s... into the Section
    here = tr.position_s
    tr.pause()
    tr.set_loop(Loop(2.0, 4.0, "Chorus"))
    block = tr.render(100)
    assert [e.kind for e in block.events] == ["paused"] and tr.position_s == here and tr.pass_index == 1
    tr.play()
    started = []
    for _ in range(40):
        started += [e for e in tr.render(100).events if e.kind == "pass_started"]
    assert started[0].pass_index == 2 and started[0].loop.name == "Chorus"

"""Device timestamps keep voice and song in sync when frame counting slips (#49).

A stream's frame count is a clock that slips: an output block delivered too late is a silence the count never sees, a
mic chunk lost before its callback is a gap the count closes, two crystals drift. StreamTimeline keeps what the device
said (the DAC / ADC time of each block on the host clock); a timed LogClock meets voice and output on that clock."""
import math

import numpy as np
from pytest import approx

from sing_pitch.playback import SongTimeLog
from sing_pitch.take_engine import LogClock, StreamTimeline, device_times

SR = 48000
BLOCK = 1024 / SR
CHUNK = 256 / SR
HOST0_OUT, HOST0_IN = 1000.0, 1000.3     # the host clock is not zero when the streams start


def steady_output(blocks, lost_after=None, lost_s=0.0, drift_ppm=0.0, host0=HOST0_OUT):
    """An output timeline as the BackingPlayer callback fills it: block k starts at output time k * BLOCK and leaves the
    DAC at host0 + k * BLOCK * (1 + drift), plus lost_s from block lost_after + 1 on (an underflow)."""
    times = StreamTimeline()
    for k in range(blocks):
        host = host0 + k * BLOCK / (1 + drift_ppm * 1e-6) + (lost_s if lost_after is not None and k > lost_after else 0)
        times.add(k * BLOCK, host)
    return times


def steady_input(chunks, lost_after=None, lost_s=0.0, host0=HOST0_IN):
    """An input timeline as SoundDeviceInput fills it: delivered chunk k starts at voice time k * CHUNK; chunks lost
    after chunk lost_after were captured but never delivered, so the next one was captured lost_s later."""
    times = StreamTimeline()
    for k in range(chunks):
        times.add(k * CHUNK, host0 + k * CHUNK + (lost_s if lost_after is not None and k > lost_after else 0))
    return times


def test_a_steady_stream_is_one_anchor_and_maps_linearly():
    t = steady_output(500)
    assert len(t) == 1 and t.glitches == 0
    assert t.host(3.0) == approx(HOST0_OUT + 3.0)
    assert t.stream(HOST0_OUT + 2.5) == approx(2.5)
    assert np.allclose(t.host(np.array([0.0, 1.0])), [HOST0_OUT, HOST0_OUT + 1.0])


def test_a_lost_block_is_one_anchor_one_glitch_and_the_frames_after_it_are_heard_later():
    t = steady_output(500, lost_after=100, lost_s=2 * BLOCK)
    assert len(t) == 2 and t.glitches == 1 and t.lost_s == approx(2 * BLOCK)
    assert t.host(50 * BLOCK) == approx(HOST0_OUT + 50 * BLOCK)                   # before: as counted
    assert t.host(200 * BLOCK) == approx(HOST0_OUT + 202 * BLOCK)                 # after: two blocks later
    gap_start = HOST0_OUT + 101 * BLOCK                                           # the DAC played silence here
    assert t.stream(gap_start + 0.5 * BLOCK) == approx(101 * BLOCK)               # the song waits at the next frame
    assert t.stream(gap_start + 2 * BLOCK) == approx(101 * BLOCK)


def test_slow_drift_is_followed_within_the_tolerance_and_is_not_a_glitch():
    t = steady_output(3000, drift_ppm=200)               # 64 s, 13 ms of drift in all
    assert t.glitches == 0 and 2 < len(t) < 60
    for k in (0, 1000, 2999):
        true = HOST0_OUT + k * BLOCK / (1 + 200e-6)
        assert abs(t.host(k * BLOCK) - true) <= StreamTimeline.TOLERANCE_S + 1e-9


def test_without_timestamps_host_time_is_the_frame_count():
    t = StreamTimeline()
    assert t.host(2.0) == 2.0 and t.stream(5.0) == 5.0


def test_timelines_are_used_together_only_when_both_are_on_one_clock():
    out, inp = steady_output(3), steady_input(3)
    assert device_times(out, inp) == (out, inp)
    assert device_times(out, None) is None                                        # the mic keeps no timeline
    other = StreamTimeline(clock=object())
    other.add(0.0, 1.0)
    assert device_times(out, other) is None                                       # two different clocks


# --- the song clock ---------------------------------------------------------------------------------------------------

LATENCY = 0.15
RENDERED_AT_FIRST_CHUNK = 3 * BLOCK      # the Backing had rendered three blocks when the first mic chunk arrived


def linear_log(seconds=30.0):
    log = SongTimeLog(SR)
    log.append(round(seconds * SR), 0.0)          # the song from 0, linearly
    return log


def aligned_input_host0():
    """The input's host start that makes the timestamps tell the same story as the counted frames at the alignment:
    the first chunk's end (input) and the output's next frame to render were captured / heard at one moment."""
    return HOST0_OUT + RENDERED_AT_FIRST_CHUNK - CHUNK


def clock(out=None, inp=None):
    devices = None if out is None else (out, inp)
    return LogClock(linear_log(), LATENCY, RENDERED_AT_FIRST_CHUNK - CHUNK, devices, aligned_at_s=CHUNK)


def test_untimed_the_clock_is_the_old_frame_count_mapping():
    c = clock()
    assert not c.timed and c.voice_now() is None
    assert c.offset_s == approx(LATENCY - (RENDERED_AT_FIRST_CHUNK - CHUNK))
    for t in (0.5, 7.0, 20.0):
        assert c(t) == approx(t - c.offset_s)
        assert c.to_voice(c.to_output(t)) == approx(t)


def test_a_calibrated_latency_means_the_same_timed_or_untimed_when_nothing_is_lost():
    """Whatever the host clock reads, and whatever gap there is between the two streams' start times, the timed clock
    scores every voice frame against the same song time as frame counting did: a latency calibrated before #49 is
    still right."""
    for in_host0 in (aligned_input_host0(), aligned_input_host0() + 0.037, 5.0):
        timed = clock(steady_output(900), steady_input(4000, host0=in_host0))
        old = clock()
        assert timed.timed
        t = np.linspace(0.2, 20.0, 50)
        assert np.allclose(timed.song_times(t), old.song_times(t), atol=1e-9)
        assert timed.offset_s == approx(old.offset_s)


def test_after_a_lost_output_block_the_voice_is_scored_against_the_song_heard_not_the_song_counted():
    """Two output blocks (43 ms) are lost at output time ~2.1 s: from then on everything is heard 43 ms later than
    counted. A voice frame sung later is answering that later song, and the timed clock says so; counting would score
    it 43 ms early for the rest of the Take."""
    lost = 2 * BLOCK
    out = steady_output(900, lost_after=100, lost_s=lost)
    timed, old = clock(out, steady_input(4000, host0=aligned_input_host0())), clock()
    t = 10.0
    assert timed(t) == approx(old(t) - lost, abs=1e-6)
    assert timed(1.0) == approx(old(1.0), abs=1e-9)          # before the loss nothing changes


def test_after_a_dropped_mic_chunk_the_voice_after_it_is_timed_where_it_was_sung():
    """Three mic chunks (16 ms) are lost: the voice counted after them was sung 16 ms later than counted."""
    lost = 3 * CHUNK
    inp = steady_input(4000, lost_after=200, lost_s=lost, host0=aligned_input_host0())
    timed, old = clock(steady_output(900), inp), clock()
    assert timed(10.0) == approx(old(10.0) + lost, abs=1e-6)
    assert inp.glitches == 1 and inp.lost_s == approx(lost)


def test_drift_between_the_devices_does_not_add_up():
    """The output crystal runs 300 ppm fast against the mic: counting drifts 6 ms in 20 s, the timed clock stays put."""
    out = steady_output(1000, drift_ppm=300)
    timed, old = clock(out, steady_input(4000, host0=aligned_input_host0())), clock()
    t = 20.0
    true = old(t) + 300e-6 * (t - old.offset_s)      # the output ran 300 ppm more song by the time it was heard
    assert abs(timed(t) - true) < 0.001
    assert abs(old(t) - true) > 0.005


def test_a_clock_whose_output_has_no_timestamp_yet_counts_until_it_has_one_then_follows_the_timestamps():
    """Nothing was played yet when the first mic chunk came (rendered 0): the clock counts until the Backing's first
    block is stamped, then reads the gap from it. That first gap is what the calibration measured too (the render
    ahead), so it maps as counting did; a loss after it shows."""
    out, inp = StreamTimeline(), steady_input(4000, host0=HOST0_OUT - CHUNK)
    c = LogClock(linear_log(), LATENCY, 0.0 - CHUNK, (out, inp), aligned_at_s=CHUNK)
    old = LogClock(linear_log(), LATENCY, 0.0 - CHUNK, aligned_at_s=CHUNK)
    assert not c.timed and c(5.0) == approx(old(5.0))
    lost = 2 * BLOCK
    for k in range(900):
        out.add(k * BLOCK, HOST0_OUT + 0.040 + k * BLOCK + (lost if k > 100 else 0.0))
    assert c.timed
    assert c(1.0) == approx(old(1.0), abs=1e-6)
    assert c(10.0) == approx(old(10.0) - lost, abs=1e-6)


def test_cuts_and_pauses_are_moved_onto_the_voice_clock_through_the_timestamps():
    lost = 2 * BLOCK
    out = steady_output(900, lost_after=100, lost_s=lost)
    log = SongTimeLog(SR)
    log.append(round(5.0 * SR), 0.0)
    log.append(round(1.0 * SR), None)                # paused from output 5 s to 6 s
    log.append(round(9.0 * SR), 5.0 * SR)
    c = LogClock(log, LATENCY, RENDERED_AT_FIRST_CHUNK - CHUNK, (out, steady_input(4000, host0=aligned_input_host0())),
                 aligned_at_s=CHUNK)
    old = LogClock(log, LATENCY, RENDERED_AT_FIRST_CHUNK - CHUNK, aligned_at_s=CHUNK)
    (start, end), = c.paused_spans()
    (old_start, old_end), = old.paused_spans()
    assert start == approx(old_start + lost) and end == approx(old_end + lost)
    assert c.to_voice(math.inf) == math.inf


def test_while_a_block_is_late_the_song_waits_at_the_last_frame_handed_over():
    """The callback for the block after 2.0 s has not run yet (the GIL is held elsewhere): nothing beyond 2.0 s of
    output can be playing, so the clock stands there instead of running on into the silence."""
    log = SongTimeLog(SR)
    log.append(round(2.0 * SR), 0.0)                 # rendered up to output 2.0 s
    out = StreamTimeline()
    out.add(0.0, HOST0_OUT)
    inp = steady_input(4000, host0=aligned_input_host0())
    c = LogClock(log, LATENCY, RENDERED_AT_FIRST_CHUNK - CHUNK, (out, inp), aligned_at_s=CHUNK)
    t_heard_at_2 = c.to_voice(2.0)
    assert c(t_heard_at_2 - 0.1) == approx(1.9)
    assert c.to_output(t_heard_at_2 + 0.08) == approx(2.0) and c(t_heard_at_2 + 0.08) == approx(2.0)
    assert np.allclose(c.song_times(np.array([t_heard_at_2 - 0.1, t_heard_at_2 + 0.5])), [1.9, 2.0])


def test_a_clock_aligned_before_anything_was_played_aligns_again_at_the_first_chunk_after():
    """The Backing's first block came after the first mic chunk: nothing rendered, no frontier to align on. At the next
    chunk after the first block the runner aligns again; the clock is then the one it would have been."""
    log = linear_log()
    first = LogClock(log, LATENCY, 0.0 - CHUNK, aligned_at_s=CHUNK)
    assert not first.aligned
    first.realign(RENDERED_AT_FIRST_CHUNK - 4 * CHUNK, 4 * CHUNK)
    fresh = LogClock(log, LATENCY, RENDERED_AT_FIRST_CHUNK - 4 * CHUNK, aligned_at_s=4 * CHUNK)
    assert first.aligned and first(5.0) == approx(fresh(5.0)) and first.offset_s == approx(fresh.offset_s)
    out, inp = StreamTimeline(), steady_input(4000, host0=aligned_input_host0() - 3 * CHUNK)
    timed = LogClock(log, LATENCY, 0.0 - CHUNK, (out, inp), aligned_at_s=CHUNK)
    for k in range(900):                            # the output's stamps arrive
        out.add(k * BLOCK, HOST0_OUT + k * BLOCK)
    timed.realign(RENDERED_AT_FIRST_CHUNK - 4 * CHUNK, 4 * CHUNK)
    assert timed.timed and timed(5.0) == approx(fresh(5.0), abs=1e-9)


class Clock:
    def __init__(self, now):
        self.t = now

    def now(self):
        return self.t


def test_while_a_block_is_late_the_song_ahead_goes_on_from_now_not_from_where_rendering_stopped():
    """The plot's right half projects the song on from the end of what is rendered. While the next block is late that
    end falls behind the voice now: the song then goes on no earlier than now (it waits in the silence)."""
    clk = Clock(HOST0_OUT + 3.0)
    log = SongTimeLog(SR)
    log.append(round(2.0 * SR), 0.0)
    out, inp = StreamTimeline(clk), StreamTimeline(clk)
    out.add(0.0, HOST0_OUT)
    for k in range(int(3.0 / CHUNK)):
        inp.add(k * CHUNK, aligned_input_host0() + k * CHUNK)
    c = LogClock(log, LATENCY, RENDERED_AT_FIRST_CHUNK - CHUNK, (out, inp), aligned_at_s=CHUNK)
    voice_at_end = c.to_voice(2.0)
    assert c.voice_now() > voice_at_end                    # the output ran dry at 2.0 s, now is later
    assert c.tail() == (approx(c.voice_now()), approx(2.0), 1.0)
    clk.t = HOST0_OUT + 1.0                               # before it ran dry: the tail is where rendering ends
    assert c.tail()[0] == approx(voice_at_end)

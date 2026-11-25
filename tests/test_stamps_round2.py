"""Round 2 of the device-timestamp sync (#62, #63, jitter guard), on the clock alone with exact or noisy stamps.

Device model as in the #61 review (align_phase.py): the output renders 1024-frame blocks, each heard RENDER_AHEAD after
its callback; the mic delivers 256-frame chunks, each captured IN_LAT + one chunk before its callback."""
import time

import numpy as np
import pytest
from pytest import approx

from sing_pitch.playback.transport import SongTimeLog
from sing_pitch.take_engine import LogClock, StreamTimeline

SR = 48000
B_OUT, B_IN = 1024, 256
T_OUT, T_IN = B_OUT / SR, B_IN / SR
RENDER_AHEAD = 0.0364
IN_LAT = 0.005


class Clk:
    now = staticmethod(lambda: 0.0)


def device_run(phase_s: float, first_late_s: float = 0.0, latency_s: float = 0.0, loop_latency: bool = True,
               seconds: float = 3.0):
    """A Take whose first mic callback comes phase_s after an output callback (and first_late_s later still)."""
    clk = Clk()
    out, inp = StreamTimeline(clk), StreamTimeline(clk)
    log = SongTimeLog(SR)
    events = [(k * T_OUT, 0, k) for k in range(int(seconds / T_OUT) + 2)]
    t0 = 5 * T_OUT + phase_s
    events += [(max(t0 + j * T_IN, t0 + first_late_s) + 1e-9 * j, 1, j) for j in range(int((seconds - t0) / T_IN))]
    events.sort()
    clock = None
    for h, kind, i in events:
        if kind == 0:
            out.add(log.rendered_s, h + RENDER_AHEAD)
            log.append(B_OUT, song0=log.rendered_s * SR)
        else:
            inp.add(i * T_IN, t0 + i * T_IN - T_IN - IN_LAT)
            end = (i + 1) * T_IN
            if clock is None:
                clock = LogClock(log, latency_s, log.rendered_s - end, (out, inp), aligned_at_s=end,
                                 loop_latency=loop_latency)
            elif not clock.aligned and log.rendered_s > 0:
                clock.realign(log.rendered_s - end, end)
    return clock, out, inp


def calibrated(phase_s: float, loop_latency: bool) -> float:
    """What CalibrationRun measures: a click at output time c answered (0 acoustic delay) at the voice captured at
    the click's DAC time, minus the click placed on the voice clock by a latency-0 clock."""
    clock, out, inp = device_run(phase_s, latency_s=0.0, loop_latency=loop_latency)
    return float(inp.stream(out.host(1.5))) - clock.to_voice(1.5)


def take_error_s(cal_phase, take_phase, first_late_s=0.0, loop_latency=True):
    latency = calibrated(cal_phase, loop_latency)
    clock, out, inp = device_run(take_phase, first_late_s, latency, loop_latency)
    t = 2.0
    return clock.to_output(t) - float(out.stream(inp.host(t))), latency


@pytest.mark.parametrize("phase_ms", [0.0, 5.3, 10.7, 16.0, 20.0])
def test_calibrated_on_the_device_clocks_the_latency_and_the_sync_do_not_depend_on_the_first_callback(phase_ms):
    """#62: with the pure DAC-to-ADC loop nothing depends on when the first mic callback ran."""
    assert calibrated(phase_ms / 1000, True) == approx(calibrated(0.0, True), abs=1e-6)
    error, _ = take_error_s(0.0, phase_ms / 1000)
    assert abs(error) < 1e-6


@pytest.mark.parametrize("late_ms", [30, 60])
def test_a_first_mic_callback_held_up_at_start_does_not_shift_the_take(late_ms):
    error, _ = take_error_s(0.0, 0.0, first_late_s=late_ms / 1000)
    assert abs(error) < 1e-6


def test_a_latency_calibrated_the_old_way_still_carries_the_phase():
    """The stale path (loop_latency=False) is kept for latencies stored before #62: it works as before, with its
    random offset, which is why the window asks for one more calibration."""
    errors = [take_error_s(0.0, ph / 1000, loop_latency=False)[0] for ph in (0.0, 10.7, 20.0)]
    assert max(errors) - min(errors) > 0.015


def test_a_cut_stays_on_the_voice_that_answers_the_jump_when_the_mic_loses_a_chunk_after_it():
    """#63: the cut is kept as a host time; the frame compared with it on its own capture time."""
    clk = Clk()
    out, inp = StreamTimeline(clk), StreamTimeline(clk)
    log = SongTimeLog(SR)
    for k in range(200):
        out.add(log.rendered_s, k * T_OUT + 0.036)
        log.append(B_OUT, log.rendered_s * SR)
    for j in range(300):
        inp.add(j * T_IN, 0.1 + j * T_IN)
    clock = LogClock(log, 0.15, 0.5, (out, inp), aligned_at_s=1.0, loop_latency=True)
    cut = clock.answer_host(2.2)                      # computed when the jump is dequeued
    for j in range(300, 600):                         # then 100 ms of mic input is lost, and the mic goes on
        inp.add(j * T_IN, 0.1 + j * T_IN + 0.100)
    answering = clock.to_voice(2.2)                   # the voice time that really answers the jump, read now
    assert clock.voice_host(answering) == approx(cut, abs=1e-9)
    assert clock.voice_host(answering - 0.001) < cut <= clock.voice_host(answering + 0.001)


def test_the_plot_axis_never_steps_back_when_the_mic_loses_a_chunk():
    """#63: plot time = capture host time - a fixed origin. The voice after a loss is placed later (a gap)."""
    clk = Clk()
    out, inp = StreamTimeline(clk), StreamTimeline(clk)
    log = SongTimeLog(SR)
    for k in range(200):
        out.add(log.rendered_s, k * T_OUT + 0.036)
        log.append(B_OUT, log.rendered_s * SR)
    for j in range(100):
        inp.add(j * T_IN, 0.1 + j * T_IN)
    clock = LogClock(log, 0.15, 0.2, (out, inp), aligned_at_s=0.1, loop_latency=True)
    before = clock.plot_t(100 * T_IN - 0.001)
    for j in range(100, 200):
        inp.add(j * T_IN, 0.1 + j * T_IN + 0.08)
    after = clock.plot_t(100 * T_IN)
    assert after - before == approx(0.081, abs=1e-6)            # a gap of the lost 80 ms, never a step back
    assert clock.plot_t(0.5) == approx(0.5)                      # before the loss: plot time = voice time
    nows = []
    for h in np.linspace(0.3, 1.5, 200):
        clk.now = staticmethod(lambda h=h: h)
        nows.append(clock.plot_now())
    assert np.all(np.diff(nows) > 0)


# --- jitter guard -------------------------------------------------------------------------------------------------

def jittered(n, block_s, sigma_s, seed=0, host0=100.0):
    rng = np.random.default_rng(seed)
    t = StreamTimeline()
    noise = rng.normal(0.0, sigma_s, n)
    for k in range(n):
        t.add(k * block_s, host0 + k * block_s + noise[k], round(block_s * SR))
    return t


def test_an_hour_of_jittery_output_stamps_adds_almost_no_anchors_and_a_paint_stays_cheap():
    """1 h of 1024-frame blocks with 1 ms jitter (worse than anything measured: the Scarlett showed 24-75 us)."""
    t = jittered(168_750, T_OUT, 0.001)
    assert len(t) < 40 and t.glitches == 0
    assert 0.6e-3 < t.jitter_s < 1.6e-3                        # it measured the jitter
    x = np.linspace(0, 3500, 1600)
    started = time.perf_counter()
    for _ in range(10):
        t.host(x), t.stream(100.0 + x)
    assert (time.perf_counter() - started) / 10 < 0.005
    assert np.abs(t.host(x) - (100.0 + x)).max() < 4 * t.tolerance_s


def test_jittery_mic_stamps_stay_bounded_too():
    t = jittered(200_000, T_IN, 0.0004, seed=1)                # ~18 min of 256-frame chunks
    assert len(t) < 40 and t.glitches == 0
    assert t.health()["jitter_p99_us"] < 2000


def test_a_lost_block_is_still_found_under_jitter():
    rng = np.random.default_rng(2)
    t = StreamTimeline()
    for k in range(2000):
        t.add(k * T_OUT, 100.0 + k * T_OUT + rng.normal(0, 0.0003) + (2 * T_OUT if k >= 1000 else 0.0))
    assert t.glitches == 1 and t.lost_s == approx(2 * T_OUT, abs=0.002)
    assert t.host(1500 * T_OUT) == approx(100.0 + 1502 * T_OUT, abs=0.002)


def steady(n=400, at=None, value=0.0, length=1):
    t = StreamTimeline()
    for k in range(n):
        bad = at is not None and at <= k < at + length
        t.add(k * T_OUT, 100.0 + k * T_OUT + (value if bad else 0.0))
    return t


@pytest.mark.parametrize("spike", [0.300, -0.050])
def test_a_single_spike_leaves_no_anchor_and_counts_no_lost_audio(spike):
    t = steady(at=200, value=spike)
    assert len(t) == 1 and t.glitches == 0 and t.lost_s == 0 and t.spikes == 1 and t.steps_back == 0
    assert t.host(300 * T_OUT) == approx(100.0 + 300 * T_OUT)


def test_a_lasting_step_back_in_the_reported_latency_is_followed_deliberately_and_counted():
    t = steady(at=200, value=-0.030, length=400)             # e.g. a Bluetooth renegotiation reports 30 ms less
    assert t.steps_back == 1 and t.steps_back_s == approx(0.030) and t.glitches == 0
    assert t.host(300 * T_OUT) == approx(100.0 + 300 * T_OUT - 0.030)
    n, s, h = t._arrays()
    assert np.all(np.diff(h[:n]) > 0)                          # the host axis stays in order


def test_a_step_back_larger_than_the_time_since_the_last_anchor_keeps_host_time_in_order():
    t = steady(at=2, value=-0.200, length=50)
    n, s, h = t._arrays()
    assert np.all(np.diff(h[:n]) > 0)
    assert np.all(np.diff(t.stream(np.linspace(99.0, 101.0, 500))) >= 0)

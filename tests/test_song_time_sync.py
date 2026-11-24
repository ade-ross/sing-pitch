"""Melody sync through the song-time log: a voice that exactly follows what it heard, delayed by the calibrated
latency, is ~100 % in tune however the song was played (linear, or with a pause mid-Take). The Transport core is
rendered by hand, block by block, as the output stream would; the voice is built from the play/pause SCHEDULE,
not from the log, so the log is checked against an independent source. Also seeks, Speed changes, and both."""
import numpy as np
import pytest
from frames import HOP, hz_at
from melody_helpers import reference_line

from sing_pitch.pitch import PitchFrame
from sing_pitch.playback.transport import SongTimeLog, Transport
from sing_pitch.reference import Melody
from sing_pitch.take_engine import LinearClock, LogClock, PauseFold, SampledClock, TakeEngine

MELODY = [(-900, 0.6), (-700, 0.6), (-500, 0.6), (-400, 0.6), (-200, 0.6), (-500, 0.6), (-900, 1.0)]  # C D E F G E C
SR = 48000
BLOCK = 1024
LATENCY = 0.23                 # calibrated output -> ear -> mic -> input loop
OUT_AT_INPUT_ZERO = 0.05       # the Backing had rendered 0.05 s when the input clock was at 0


def note_at(song_t):
    start = 0.0
    for cents, seconds in MELODY:
        if start <= song_t < start + seconds:
            return cents
        start += seconds
    return None


def voice(song_time_heard, seconds):
    """Tracker frames of a singer copying exactly what the song played LATENCY s before each input moment.
    song_time_heard(output_s) -> song time, or None for silence (paused)."""
    out = []
    for i in range(round(seconds / HOP)):
        t = i * HOP
        song_t = song_time_heard(t - LATENCY + OUT_AT_INPUT_ZERO)
        cents = None if song_t is None else note_at(song_t)
        out.append(PitchFrame(t, 0.0, 0.0, -140.0) if cents is None else PitchFrame(t, hz_at(cents), 0.98, -20.0))
    return out


def render(transport, seconds, pause_at=None, pause_for=0.0, seek_at=None, seek_to=0.0):
    """Drive the Transport like the output callback for `seconds` of output, pausing at output time pause_at and
    seeking to song time seek_to at output time seek_at."""
    t, paused_until, sought = 0.0, None, False
    while t < seconds:
        if seek_at is not None and t >= seek_at and not sought:
            transport.seek(seek_to)
            sought = True
        if pause_at is not None and t >= pause_at and paused_until is None:
            transport.pause()
            paused_until = t + pause_for
        if paused_until is not None and t >= paused_until:
            transport.play()
        transport.render(BLOCK)
        t += BLOCK / SR


def take(song_clock, frames):
    engine = TakeEngine(Melody(reference_line(MELODY), "do-re-mi"), song_clock=song_clock)
    engine.push(frames)
    return engine.stop()


def silent_song():
    return np.zeros((round(4.6 * SR), 2), np.float32)


def test_linear_playback_scores_about_100_in_tune_through_the_log():
    tr = Transport(silent_song(), sample_rate=SR)
    render(tr, 6.0)
    clock = LogClock(tr.log, LATENCY, OUT_AT_INPUT_ZERO)
    s = take(clock, voice(lambda out_s: out_s, 5.0)).summary
    assert s.n_scored > 250 and s.pct_in_tune >= 98.0 and s.median_abs_cents < 1.0


def test_a_pause_and_resume_mid_take_still_scores_about_100_in_tune():
    tr = Transport(silent_song(), sample_rate=SR)
    render(tr, 7.0, pause_at=2.0, pause_for=1.0)
    # the schedule, independently of the log: the pause took effect at the first block boundary at or after 2.0 s
    p0 = np.ceil(2.0 * SR / BLOCK) * BLOCK / SR
    p1 = np.ceil((p0 + 1.0) * SR / BLOCK) * BLOCK / SR

    def heard(out_s):
        if out_s < p0:
            return out_s
        if out_s < p1:
            return None
        return out_s - (p1 - p0)

    frames = voice(heard, 6.0)
    record = take(LogClock(tr.log, LATENCY, OUT_AT_INPUT_ZERO), frames)
    assert record.summary.n_scored > 250
    assert record.summary.pct_in_tune >= 98.0 and record.summary.median_abs_cents < 1.0

    # the old way, one constant offset for the whole Take, is lost after the pause
    old = take(LinearClock(LATENCY - OUT_AT_INPUT_ZERO), frames)
    assert old.summary.pct_in_tune < 70.0


def test_a_take_with_a_pause_saves_its_song_times_and_rescores_to_the_same_numbers(tmp_path):
    from melody_helpers import library_song

    from sing_pitch.take_store import rescore_take, save_take

    library_song(tmp_path, reference_line(MELODY))           # song id "do-re-mi"
    tr = Transport(silent_song(), sample_rate=SR)
    render(tr, 7.0, pause_at=2.0, pause_for=1.0)
    frames = voice(lambda out_s: tr.log.song_time(out_s), 6.0)
    saved = save_take(tmp_path, take(LogClock(tr.log, LATENCY, OUT_AT_INPUT_ZERO), frames))
    again = rescore_take(tmp_path, saved.id)
    assert again.summary == saved.summary and saved.summary.pct_in_tune >= 98.0
    assert again.song_times == saved.song_times


def test_a_seek_mid_take_still_scores_about_100_in_tune_and_splits_into_two_takes_that_both_do():
    tr = Transport(silent_song(), sample_rate=SR)
    render(tr, 6.0, seek_at=2.0, seek_to=0.6)
    # the schedule, independently of the log: the seek took effect at the first block boundary at or after 2.0 s
    p0 = np.ceil(2.0 * SR / BLOCK) * BLOCK / SR

    def heard(out_s):
        return out_s if out_s < p0 else 0.6 + (out_s - p0)

    frames = voice(heard, 5.0)
    clock = LogClock(tr.log, LATENCY, OUT_AT_INPUT_ZERO)
    whole = take(clock, frames).summary
    assert whole.n_scored > 250 and whole.pct_in_tune >= 98.0 and whole.median_abs_cents < 1.0

    cut = p0 + LATENCY - OUT_AT_INPUT_ZERO         # input time of the first voice frame that answers the jump
    before, after = take(clock, [f for f in frames if f.t < cut]), take(clock, [f for f in frames if f.t >= cut])
    for part in (before, after):
        assert part.summary.n_scored > 80 and part.summary.pct_in_tune >= 98.0

    # the old way, one constant offset for the whole Take, is lost after the jump
    old = take(LinearClock(LATENCY - OUT_AT_INPUT_ZERO), frames)
    assert old.summary.pct_in_tune < 70.0


# --- Speed: the voice follows the reference at the chosen Speed, and across a change of Speed ------------------------

def silent_stretched(speed):
    return np.zeros((round(4.6 * SR / speed), 2), np.float32)


def render_with_speed_changes(transport, seconds, changes):
    """Drive the Transport for `seconds` of output; changes = [(output_s, speed)] requested from that time on. Returns
    the schedule [(output_s where the new Speed took effect, song_s there, speed)] built from the requests alone."""
    schedule, pending, t, song_s, speed = [(0.0, 0.0, 1.0)], sorted(changes), 0.0, 0.0, 1.0
    while t < seconds:
        if pending and t >= pending[0][0]:
            _, speed = pending.pop(0)
            transport.set_speed(speed, silent_stretched(speed))
            schedule.append((t, song_s, speed))                 # takes effect at this block: the first one at/after t
        transport.render(BLOCK)
        t += BLOCK / SR
        song_s += speed * BLOCK / SR
    return schedule


def heard_from(schedule):
    def heard(out_s):
        out0, song0, speed = [s for s in schedule if s[0] <= out_s or s is schedule[0]][-1]
        return song0 + (out_s - out0) * speed
    return heard


def test_a_voice_following_the_reference_at_70_percent_scores_about_100_in_tune():
    tr = Transport(silent_song(), sample_rate=SR)
    tr.set_speed(0.7, silent_stretched(0.7))
    render_with_speed_changes(tr, 8.0, [])
    frames = voice(lambda out_s: 0.7 * out_s, 7.0)
    s = take(LogClock(tr.log, LATENCY, OUT_AT_INPUT_ZERO), frames).summary
    assert s.n_scored > 250 and s.pct_in_tune >= 98.0 and s.median_abs_cents < 1.0
    # one constant offset, the old way, is lost as soon as the song runs at another Speed
    old = take(LinearClock(LATENCY - OUT_AT_INPUT_ZERO), frames).summary
    assert old.pct_in_tune < 70.0


def test_a_speed_change_mid_take_still_scores_about_100_in_tune():
    tr = Transport(silent_song(), sample_rate=SR)
    # 100 % -> 70 % -> 50 %. The change takes effect at the first block boundary at or after the request
    schedule = render_with_speed_changes(tr, 9.0, [(1.5, 0.7), (4.0, 0.5)])
    assert [round(sp, 1) for _, _, sp in schedule] == [1.0, 0.7, 0.5]
    frames = voice(heard_from(schedule), 8.0)
    s = take(LogClock(tr.log, LATENCY, OUT_AT_INPUT_ZERO), frames).summary
    assert s.n_scored > 250 and s.pct_in_tune >= 98.0 and s.median_abs_cents < 1.0
    assert take(LinearClock(LATENCY - OUT_AT_INPUT_ZERO), frames).summary.pct_in_tune < 70.0


# --- Seek and Speed together ----------------------------------------------------------------------------------------

def drive(transport, seconds, actions):
    """Drive the Transport for `seconds` of output; actions = [(output_s, call)] made before the first block at or
    after output_s. Returns the output time each action took effect at, in order."""
    pending, t, took = sorted(actions, key=lambda a: a[0]), 0.0, []
    while t < seconds:
        while pending and t >= pending[0][0]:
            pending.pop(0)[1]()
            took.append(t)
        transport.render(BLOCK)
        t += BLOCK / SR
    return took


def split_takes(clock, frames, out_cut):
    cut = out_cut + LATENCY - OUT_AT_INPUT_ZERO         # input time of the first voice frame that answers the jump
    return take(clock, [f for f in frames if f.t < cut]), take(clock, [f for f in frames if f.t >= cut])


def test_a_seek_at_70_percent_and_a_speed_change_after_a_seek_keep_each_take_about_100_in_tune():
    # a seek while at 70 %: the song goes on at 70 % from the new place
    tr = Transport(silent_song(), sample_rate=SR)
    tr.set_speed(0.7, silent_stretched(0.7))
    (p0,) = drive(tr, 8.0, [(2.0, lambda: tr.seek(0.6))])

    def heard(out_s):
        return 0.7 * out_s if out_s < p0 else 0.6 + 0.7 * (out_s - p0)

    frames = voice(heard, 7.0)
    for part in split_takes(LogClock(tr.log, LATENCY, OUT_AT_INPUT_ZERO), frames, p0):
        assert part.summary.n_scored > 80 and part.summary.pct_in_tune >= 98.0 and part.summary.median_abs_cents < 1.0

    # a seek at 100 %, then 70 % inside the second Take
    tr = Transport(silent_song(), sample_rate=SR)
    p0, p1 = drive(tr, 8.0, [(1.5, lambda: tr.seek(0.3)), (3.0, lambda: tr.set_speed(0.7, silent_stretched(0.7)))])

    def heard2(out_s):
        if out_s < p0:
            return out_s
        if out_s < p1:
            return 0.3 + (out_s - p0)
        return 0.3 + (p1 - p0) + 0.7 * (out_s - p1)

    frames = voice(heard2, 7.0)
    for part in split_takes(LogClock(tr.log, LATENCY, OUT_AT_INPUT_ZERO), frames, p0):
        assert part.summary.n_scored > 80 and part.summary.pct_in_tune >= 98.0 and part.summary.median_abs_cents < 1.0


def test_a_seek_in_the_crossfade_block_of_a_speed_change_does_not_glitch_song_time():
    tr = Transport(silent_song(), sample_rate=SR)
    tr.render(BLOCK)
    tr.set_speed(0.7, silent_stretched(0.7))
    tr.seek(2.0)
    fade = tr.render(BLOCK)                              # crossfades the Speeds AND starts at the new place
    assert [e.kind for e in fade.events] == ["speed_changed", "jumped"]
    assert np.allclose(fade.song_t, 2.0 + 0.7 * np.arange(BLOCK) / SR)
    nxt = tr.render(BLOCK)
    assert np.allclose(nxt.song_t, 2.0 + 0.7 * (BLOCK + np.arange(BLOCK)) / SR)      # no step, no repeat
    out_fade = BLOCK / SR
    assert np.isclose(tr.log.song_time(out_fade + 0.01), 2.0 + 0.7 * 0.01)
    # a seek in the block right after the change (the old Speed may still be ringing out) is just as clean
    tr.set_speed(1.0, silent_song())
    tr.render(BLOCK)
    tr.seek(1.0)
    after = tr.render(BLOCK)
    assert [e.kind for e in after.events] == ["jumped"] and np.allclose(after.song_t, 1.0 + np.arange(BLOCK) / SR)


def test_a_pause_fold_turns_voice_time_into_view_time_with_every_pause_folded_away():
    fold = PauseFold([(2.0, 3.0), (5.0, 7.0)])                    # paused 2 -> 3 and 5 -> 7 s of voice time
    assert fold.to_view(np.array([1.0, 2.5, 3.0, 4.0, 6.0, 8.0])) == pytest.approx([1.0, 2.0, 2.0, 3.0, 4.0, 5.0])
    assert fold.to_voice(np.array([1.0, 2.0, 3.0, 4.0, 5.0])) == pytest.approx([1.0, 3.0, 4.0, 7.0, 8.0])
    assert fold.paused(2.5) and not fold.paused(3.0) and not fold.paused(1.9)
    still = PauseFold([(2.0, np.inf)])                            # still paused
    assert still.to_view(10.0) == 2.0 and still.to_voice(4.0) == 2.0 and still.paused(10.0)
    assert PauseFold().to_view(4.2) == 4.2 and PauseFold().to_voice(4.2) == 4.2


def test_the_log_and_the_clocks_report_where_the_song_was_paused():
    log = SongTimeLog(100)
    log.append(200, 0.0)
    log.append(100, None)                                         # paused at output 2 -> 3 s
    log.append(100, 200.0)
    log.append(50, None)                                          # paused again from 4 s, still paused
    assert log.paused_spans() == [(2.0, 3.0), (4.0, np.inf)]
    spans = LogClock(log, 0.3, 0.1).paused_spans()                # voice time = output time + latency - 0.1
    assert np.allclose(spans, [(2.2, 3.2), (4.2, np.inf)])
    clock = SampledClock([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, None, None, 1.0, None])
    assert clock.paused_spans() == [(1.0, 3.0), (4.0, np.inf)]

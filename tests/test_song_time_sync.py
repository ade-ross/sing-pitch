"""Melody sync through the song-time log: a voice that exactly follows what it heard, delayed by the calibrated
latency, is ~100 % in tune however the song was played (linear, or with a pause mid-Take). The Transport core is
rendered by hand, block by block, as the output stream would; the voice is built from the play/pause SCHEDULE,
not from the log, so the log is checked against an independent source."""
import numpy as np
from frames import HOP, hz_at
from melody_helpers import reference_line

from sing_pitch.pitch import PitchFrame
from sing_pitch.playback.transport import Transport
from sing_pitch.reference import Melody
from sing_pitch.take_engine import LinearClock, LogClock, TakeEngine

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

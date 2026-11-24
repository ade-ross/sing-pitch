"""A jump closes the current Take and opens a new one, through the live runner: a fake mic fed by hand, a fake tracker
that emits the frames of a synthetic voice, and a fake Backing whose song-time log was rendered by the real Transport
core. No sound card, no model. The voice is built from the seek SCHEDULE, not from the log."""
import csv

import numpy as np
from frames import HOP, hz_at
from melody_helpers import reference_line

from sing_pitch.live import LiveRunner
from sing_pitch.pitch import PitchFrame
from sing_pitch.playback.transport import SongTimeLog, Transport, TransportEvent
from sing_pitch.reference import Melody

MELODY = [(-900, 0.6), (-700, 0.6), (-500, 0.6), (-400, 0.6), (-200, 0.6), (-500, 0.6), (-900, 1.0)]  # 4.6 s
SR, MIC_SR, CHUNK, BLOCK = 48000, 16000, 160, 1024
LATENCY, OUT_AT_ZERO = 0.23, 0.05


def note_at(song_t):
    start = 0.0
    for cents, seconds in MELODY:
        if start <= song_t < start + seconds:
            return cents
        start += seconds
    return None


class LogView:
    """The Transport's real log, but 'rendered so far' is what the runner would have seen at the first mic chunk."""

    def __init__(self, log, rendered_s):
        self._log, self.rendered_s, self.sample_rate = log, rendered_s, log.sample_rate

    def song_time(self, output_s):
        return self._log.song_time(output_s)


class Backing:
    def __init__(self, log):
        self.song_log, self.on_event, self.started, self.stopped = log, None, False, False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def position_s(self):
        return 0.0

    def set_gains(self, backing, vocal):
        pass


class Mic:
    sample_rate = MIC_SR

    def start(self, on_chunk):
        self.on_chunk = on_chunk

    def stop(self):
        pass


class VoiceTracker:
    """Emits the synthetic voice frames whose time has been heard so far (as a streaming tracker would)."""

    def __init__(self, frames):
        self._frames, self._fed, self._next = frames, 0, 0

    def reset(self):
        self._fed = self._next = 0       # a real Take start; the runner must NOT call this at a jump

    def push(self, chunk, sample_rate):
        self._fed += len(chunk)
        out = []
        while self._next < len(self._frames) and self._frames[self._next].t < self._fed / sample_rate:
            out.append(self._frames[self._next])
            self._next += 1
        return out

    def flush(self):
        out, self._next = self._frames[self._next:], len(self._frames)
        return out


def voice(heard, seconds, out_at_zero=OUT_AT_ZERO):
    """Frames of a singer copying exactly what the song played LATENCY s before each input moment."""
    out = []
    for i in range(round(seconds / HOP)):
        t = i * HOP
        song_t = heard(t - LATENCY + out_at_zero)
        cents = None if song_t is None else note_at(song_t)
        out.append(PitchFrame(t, 0.0, 0.0, -140.0) if cents is None else PitchFrame(t, hz_at(cents), 0.98, -20.0))
    return out


def song_with_seeks(seeks, seconds):
    """Transport core rendered by hand; seeks = [(output_s, song_s)]. Returns (log, [first output_s of each jump])."""
    tr = Transport(np.zeros((round(4.6 * SR), 2), np.float32), sample_rate=SR)
    todo, jumps, t = sorted(seeks), [], 0.0
    while t < seconds:
        if todo and t >= todo[0][0]:
            tr.seek(todo.pop(0)[1])
        jumps += [e.output_s for e in tr.render(BLOCK).events]
        t += BLOCK / SR
    return tr.log, jumps


def sing(tmp_path, seeks, seconds):
    """One Start..Stop with the jumps announced to the runner before the voice that answers them. Returns the stop()
    record and the takes.csv rows."""
    log, jumps = song_with_seeks(seeks, seconds + 1)
    schedule = [(p, s) for p, (_, s) in zip(jumps, sorted(seeks))]

    def heard(out_s):
        pos = out_s
        for p, s in schedule:
            if out_s >= p:
                pos = s + (out_s - p)
        return pos

    frames = voice(heard, seconds)
    backing = Backing(LogView(log, OUT_AT_ZERO + CHUNK / MIC_SR))
    mic = Mic()
    runner = LiveRunner(mic, VoiceTracker(frames), Melody(reference_line(MELODY), "do-re-mi"), tmp_path,
                        backing=backing, latency_s=LATENCY)
    runner.start()
    pending = [(p + LATENCY - OUT_AT_ZERO, p, s) for p, s in schedule]     # (input time of the cut, output_s, song_s)
    fed = 0
    while fed < round(seconds * MIC_SR):
        while pending and fed / MIC_SR >= pending[0][0] - 0.1:             # the event arrives before its voice
            _, out_s, song_s = pending.pop(0)
            backing.on_event(TransportEvent("jumped", out_s, song_s))
        mic.on_chunk(np.zeros(CHUNK, np.float32))
        fed += CHUNK
    record = runner.stop()
    return runner, record, list(csv.DictReader(open(tmp_path / "takes.csv")))


def test_a_jump_closes_the_take_and_opens_a_new_one_and_both_score_in_tune(tmp_path):
    runner, record, rows = sing(tmp_path, seeks=[(3.5, 0.0)], seconds=7.5)
    assert len(rows) == 2 and record.id == rows[1]["id"]
    for row in rows:
        assert row["song_id"] == "do-re-mi" and float(row["pct_in_tune"]) >= 98.0
        assert float(row["duration_s"]) > 3.0
    first, second = (float(r["duration_s"]) for r in rows)
    assert 3.55 < first < 3.85 and abs(first + second - 7.5) < 0.1      # cut at the jump's voice, not elsewhere
    assert [t.id for t in runner.drain_saved()] == [rows[0]["id"]]       # the UI hears about the Take saved mid-run


def test_a_take_under_3_s_of_scored_singing_is_discarded_at_a_jump(tmp_path):
    runner, record, rows = sing(tmp_path, seeks=[(1.0, 0.0)], seconds=6.0)
    assert len(rows) == 1 and record.id == rows[0]["id"]          # the 1 s Take before the jump is gone
    assert float(rows[0]["pct_in_tune"]) >= 98.0 and float(rows[0]["duration_s"]) > 4.0
    assert runner.drain_saved() == []


def test_several_jumps_make_several_takes_and_a_short_last_one_is_dropped(tmp_path):
    runner, record, rows = sing(tmp_path, seeks=[(3.5, 0.0), (8.0, 1.0)], seconds=9.0)
    assert len(rows) == 2 and record.id == rows[1]["id"]          # 3.5 s | 4.5 s | 1 s: the last is under 3 s
    assert all(float(r["pct_in_tune"]) >= 98.0 for r in rows)


def test_a_run_that_never_reached_3_s_still_saves_its_last_take(tmp_path):
    _, record, rows = sing(tmp_path, seeks=[(0.5, 0.0)], seconds=2.0)
    assert len(rows) == 1 and record.id == rows[0]["id"]          # Start and Stop always leave one Take

"""Timing shift per Phrase (QA #44, "forgive timing, not pitch"): each Phrase gets ONE time shift (-100..+100 ms in 10 ms
steps), the one whose scored frames have the smallest median absolute error, chosen jointly with the Phrase's Octave
shift; every frame of the Phrase is then scored strictly at that shift. A voice that is late is forgiven; a voice that
is out of tune stays out of tune, even on a line with vibrato where some reference frame within 100 ms is always near
it. Seams: score_take / TakeEngine with a Melody, takes.csv via save_take / row_config."""
import csv
import math
from datetime import datetime

import numpy as np
import pytest
from frames import HOP, hz_at, silence
from melody_helpers import REF_HOP, library_song, reference_line, voice_following

from sing_pitch.config import ScoreConfig
from sing_pitch.library.ports import ReferenceLine
from sing_pitch.pitch import PitchFrame
from sing_pitch.reference import Band, Melody
from sing_pitch.take_engine import LinearClock, TakeEngine, score_take
from sing_pitch.take_store import rescore_take, row_config, save_take, summary_path

SECONDS = 4.0


def vibrato_cents(s: float) -> float:
    """A sung line: 0.5 s notes with a 40-cent, 5.5 Hz vibrato (a real vocal's), so within +-100 ms the line passes
    through 80 cents of pitch around each note."""
    notes = [-900, -700, -500, -700, -400, -200, -500, -900]
    return notes[min(int(s / 0.5), len(notes) - 1)] + 40.0 * math.sin(2 * math.pi * 5.5 * s)


def vibrato_line() -> ReferenceLine:
    t = np.arange(round(SECONDS / REF_HOP)) * REF_HOP
    return ReferenceLine(t.astype(np.float32), np.array([hz_at(vibrato_cents(x)) for x in t], np.float32),
                         np.full(len(t), 0.9, np.float32))


def singer(late_s: float = 0.0, cents_off: float = 0.0, start_s: float = 0.0, until_s: float = SECONDS):
    """Tracker frames of a voice singing the vibrato line late_s late (negative: early), cents_off out of tune. Each
    frame's own t is the song time it answers (LinearClock 0)."""
    out = []
    for i in range(round(start_s / HOP), round(until_s / HOP)):
        t = i * HOP
        sung_at = t - late_s
        if 0.0 <= sung_at < SECONDS:
            out.append(PitchFrame(t, hz_at(vibrato_cents(sung_at) + cents_off), 0.98, -20.0))
    return out


MELODY = Melody(vibrato_line(), "vib")


@pytest.mark.parametrize("late_s", [0.0, 0.05, -0.05, 0.1, -0.1])
def test_a_singer_on_pitch_up_to_100_ms_late_or_early_scores_in_tune(late_s):
    s = score_take(singer(late_s), MELODY)
    assert s.pct_in_tune >= 95.0
    assert s.timing_shift_ms == pytest.approx(1000 * late_s, abs=10)


@pytest.mark.parametrize("late_s", [0.0, 0.05])
def test_a_singer_40_cents_sharp_stays_out_of_green_and_shows_about_40_cents(late_s):
    """The per-frame window (closest reference within 100 ms) scored him ~75 % in tune, ~8 cents off on a real vocal."""
    s = score_take(singer(late_s, cents_off=40), MELODY)
    assert s.pct_in_tune <= 15.0
    assert 35.0 <= s.median_abs_cents <= 45.0
    assert s.tendency == "sharp"


def test_a_singer_40_cents_flat_reads_flat():
    s = score_take(singer(0.03, cents_off=-40), MELODY)
    assert s.pct_in_tune <= 15.0 and s.tendency == "flat"


def test_a_singer_a_semitone_sharp_scores_about_nothing():
    s = score_take(singer(0.0, cents_off=100), MELODY)
    assert s.pct_in_tune <= 2.0
    assert 90.0 <= s.median_abs_cents <= 110.0


def test_each_phrase_gets_its_own_shift():
    # two Phrases with a 1 s pause: the first sung 80 ms late, the second 60 ms early
    first = [f for f in singer(0.08, until_s=1.5) if f.t < 1.5]
    second = [f for f in singer(-0.06, start_s=2.5) if f.t >= 2.5]
    first += silence(2.5 - first[-1].t - HOP, start=first[-1].t + HOP)
    s = score_take(first + second, MELODY)
    assert s.pct_in_tune >= 95.0
    from sing_pitch.take_engine.engine import replay_scores
    shifts = {round(1000 * sc.timing_shift_s) for sc in replay_scores(first + second, MELODY) if sc.cents is not None}
    assert shifts == {80, -60}


def test_an_octave_low_late_singer_gets_both_his_octave_and_his_shift():
    s = score_take(singer(0.07, cents_off=-1200), MELODY)
    assert s.octave_shift == -1
    assert s.pct_in_tune >= 95.0
    assert s.timing_shift_ms == pytest.approx(70, abs=10)


def test_a_singer_250_ms_late_is_not_forgiven():
    scale = [(-1500 + 300 * k, 0.2) for k in range(8)]       # rising: no note comes back
    frames = voice_following(scale, 0.25)
    assert score_take(frames, Melody(reference_line(scale), "scale")).pct_in_tune < 40.0


def test_live_frames_are_scored_at_the_phrase_shift_as_it_grows():
    engine = TakeEngine(MELODY, song_clock=LinearClock(0.0))
    late = engine.push(singer(0.06))
    scored = [s for s in late if s.band is not None]
    assert sum(s.band is Band.GREEN for s in scored) / len(scored) >= 0.9
    sharp = TakeEngine(MELODY, song_clock=LinearClock(0.0)).push(singer(0.06, cents_off=40))
    scored = [s for s in sharp if s.band is not None]
    assert sum(s.band is Band.GREEN for s in scored) / len(scored) <= 0.15


def test_unscored_stretches_never_count_and_the_shift_scores_no_extra_frame():
    from melody_helpers import with_confidence

    line = with_confidence(vibrato_line(), 1.0, 2.0, 0.05)        # an Unscored stretch from 1 s to 2 s
    frames = singer(0.08)
    strict = score_take(frames, Melody(line, "vib"), ScoreConfig(timing_window_s=0.0))
    shifted = score_take(frames, Melody(line, "vib"))
    assert shifted.n_scored == strict.n_scored            # a frame answering an Unscored stretch stays unscored


# --- takes.csv --------------------------------------------------------------------------------------------------------

def test_a_take_records_its_median_timing_shift_and_rescores_to_the_same_numbers(tmp_path):
    song = library_song(tmp_path, vibrato_line(), seconds=SECONDS)
    engine = TakeEngine(Melody(vibrato_line(), song.id), started_at=datetime(2026, 11, 24, 21, 0, 0))
    engine.push(singer(0.05, cents_off=10))
    saved = save_take(tmp_path, engine.stop())
    (row,) = csv.DictReader(open(summary_path(tmp_path)))
    assert float(row["timing_shift_ms"]) == pytest.approx(50, abs=10)
    assert row_config(row) == ScoreConfig()
    assert rescore_take(tmp_path, saved.id).summary == saved.summary


def test_a_row_from_41_without_a_timing_shift_rescores_with_the_old_per_frame_window():
    old = row_config({"tolerance_cents": "25.0", "timing_window_s": "0.1", "note_scoring": "note"})
    assert old.timing_per_phrase is False and old.timing_window_s == pytest.approx(0.1)
    frames = singer(0.0, cents_off=40)
    # the old rule: the closest reference frame within 100 ms, so the vibrato hides the 40 cents (as it scored then)
    assert score_take(frames, MELODY, old).pct_in_tune >= 50.0


def test_the_line_is_read_between_its_frames_so_a_wide_vibrato_is_not_rounded_to_the_10_ms_grid():
    """Voice frames (16 ms apart) fall between the line's 10 ms frames; on a 100-cent vibrato the nearest frame is up to
    5 ms away, up to ~19 cents. Read between frames, a perfect singer stays inside a ±10 cent Tolerance."""
    def wide(s):
        return -500 + 100.0 * math.sin(2 * math.pi * 6.0 * s)

    t = np.arange(round(SECONDS / REF_HOP)) * REF_HOP
    line = ReferenceLine(t.astype(np.float32), np.array([hz_at(wide(x)) for x in t], np.float32),
                         np.full(len(t), 0.9, np.float32))
    frames = [PitchFrame(i * HOP + 0.003, hz_at(wide(i * HOP + 0.003)), 0.98, -20.0)
              for i in range(round((SECONDS - 0.01) / HOP))]
    s = score_take(frames, Melody(line, "wide"), ScoreConfig.with_tolerance(10))
    assert s.pct_in_tune >= 98.0

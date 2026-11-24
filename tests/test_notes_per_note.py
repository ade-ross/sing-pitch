"""Notes mode per-note scoring (#41, QA #38): each note is scored once, on the voice's median pitch over that note (its
frames, after the Timing window); green if within the Tolerance of the bar. % in tune = the share of sung notes, by
note length, in green. A singer who sings what the original sings scores high whatever the Minimum note length."""
import math

import numpy as np
import pytest
from frames import HOP, hz_at
from melody_helpers import REF_HOP, reference_line, voice_following

from sing_pitch.config import ScoreConfig
from sing_pitch.library.ports import ReferenceLine
from sing_pitch.pitch import PitchFrame
from sing_pitch.reference import Band, Melody
from sing_pitch.reference.notes import PER_PHRASE
from sing_pitch.take_engine import LinearClock, NoteTally, TakeEngine, note_verdicts, replay_scores, score_take

# a sung line: notes 0.5 s long with a 40-cent, 5.5 Hz vibrato and a scoop into each, like a real vocal
NOTES = [-900, -700, -500, -700, -400, -200, -500, -900]


def cents_at(s: float) -> float:
    k = min(int(s / 0.5), len(NOTES) - 1)
    into = s - 0.5 * k
    scoop = -60.0 * max(0.0, 1 - into / 0.08)            # 80 ms scoop up into the note
    return NOTES[k] + scoop + 40.0 * math.sin(2 * math.pi * 5.5 * s)


SECONDS = 0.5 * len(NOTES)


def vocal_line() -> ReferenceLine:
    t = np.arange(round(SECONDS / REF_HOP)) * REF_HOP
    return ReferenceLine(t.astype(np.float32), np.array([hz_at(cents_at(x)) for x in t], np.float32),
                         np.full(len(t), 0.9, np.float32))


def perfect_singer(cents_off: float = 0.0, until_s: float = SECONDS):
    return [PitchFrame(i * HOP, hz_at(cents_at(i * HOP) + cents_off), 0.98, -20.0)
            for i in range(round(until_s / HOP))]


@pytest.mark.parametrize("length", [0.1, 0.3])
def test_a_perfect_singer_scores_high_in_notes_mode_when_the_bars_are_the_notes_he_sings(length):
    notes = Melody(vocal_line(), "s").notes(length)
    assert [n.midi for n in notes.note_list] == [69 + c // 100 for c in NOTES]
    per_frame = score_take(perfect_singer(), notes, ScoreConfig(notes_per_note=False)).pct_in_tune
    per_note = score_take(perfect_singer(), notes).pct_in_tune
    assert per_frame < 75.0                    # the old rule: vibrato and scoops against a flat bar
    assert per_note >= 95.0


def test_a_bar_that_merges_several_sung_notes_is_judged_on_the_voices_median_against_the_bar():
    # one note per Phrase: the bar is the semitone held longest, one of the eight sung notes; the voice's median over
    # the whole Phrase is not within 25 cents of it, so a perfect singer's one note is not green (the rule as decided)
    notes = Melody(vocal_line(), "s").notes(PER_PHRASE)
    (verdict,) = note_verdicts(replay_scores(perfect_singer(), notes), notes, ScoreConfig())
    assert verdict.band is not Band.GREEN


def test_a_note_sung_40_cents_flat_is_not_green_and_the_share_is_by_note_length():
    line = reference_line([(-900, 1.0), (-500, 0.5)])
    voice = voice_following([(-900, 1.0)], 0.0) + [
        PitchFrame(f.t + 1.0, f.hz, f.confidence, f.loudness_db) for f in voice_following([(-540, 0.5)], 0.0)]
    s = score_take(voice, Melody(line, "s").notes(0.3))
    assert s.pct_in_tune == pytest.approx(100 * 1.0 / 1.5, abs=1.5)      # the 1 s note green, the 0.5 s note not
    assert s.median_abs_cents == pytest.approx(0.0, abs=1.0)              # by note length too: the long note is on


def test_a_note_that_was_not_sung_does_not_count():
    line = reference_line([(-900, 1.0), (-500, 1.0)])
    s = score_take(voice_following([(-900, 1.0)], 0.0), Melody(line, "s").notes(0.3))
    assert s.pct_in_tune == 100.0


def test_the_verdict_uses_the_tolerance():
    line = reference_line([(-900, 1.0)])
    voice = voice_following([(-860, 1.0)], 0.0)                         # 40 cents sharp
    notes = Melody(line, "s").notes(0.3)
    assert score_take(voice, notes).pct_in_tune == 0.0
    assert score_take(voice, notes, ScoreConfig.with_tolerance(50)).pct_in_tune == 100.0
    (verdict,) = note_verdicts(replay_scores(voice, notes, ScoreConfig.with_tolerance(50)), notes,
                               ScoreConfig.with_tolerance(50))
    assert verdict.band is Band.GREEN and verdict.cents == pytest.approx(40, abs=0.5)


def test_live_frames_still_colour_per_frame_and_each_note_gets_its_verdict_once_it_has_ended():
    notes = Melody(vocal_line(), "s").notes(0.1)
    config = ScoreConfig()
    scores = TakeEngine(notes, config, song_clock=LinearClock(0.0)).push(perfect_singer())
    assert any(s.band is Band.YELLOW for s in scores)       # the dots: vibrato swings out of green against the bar
    tally = NoteTally(notes, config)
    first = [s for s in scores if s.t < 0.7]                # the voice has reached 0.7 s of song
    tally.add(first)
    done = tally.close_until(0.7)
    assert [v.note for v in done] == [0] and done[0].band is Band.GREEN     # note 0 ended at 0.5 s (+ the window)
    assert tally.close_until(0.7) == []                     # once
    tally.add([s for s in scores if s.t >= 0.7])
    rest = tally.close_until(SECONDS + 1.0)
    assert [v.note for v in rest] == list(range(1, len(notes.note_list)))
    assert all(v.band is Band.GREEN for v in rest)


def test_the_tally_and_the_summary_agree():
    notes = Melody(vocal_line(), "s").notes(0.5)
    voice = perfect_singer(cents_off=15)
    scores = replay_scores(voice, notes)
    tally = NoteTally(notes, ScoreConfig())
    tally.add(scores)
    assert tally.close_until(math.inf) == note_verdicts(scores, notes, ScoreConfig())


def test_the_line_is_still_scored_frame_by_frame():
    line = Melody(vocal_line(), "s")
    s = score_take(perfect_singer(cents_off=20), line)
    frames = [x for x in replay_scores(perfect_singer(cents_off=20), line) if x.band is not None]
    assert s.pct_in_tune == pytest.approx(100 * sum(x.band is Band.GREEN for x in frames) / len(frames))

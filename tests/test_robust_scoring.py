"""Steadiness and Range describe how the person sang (ticket #40, QA #33): a tracker octave blip does not move them,
and Steadiness does not change with the Notes slider."""
import dataclasses

import pytest
from frames import HOP, hz_at, sung
from melody_helpers import reference_line
from steadiness_helpers import contour
from test_octave_per_phrase import CHORUS, REST, VERSE, TAKE, reference_of, voice_of

from sing_pitch.pitch import PitchFrame
from sing_pitch.reference import Melody, NearestNote
from sing_pitch.reference.notes import PER_PHRASE
from sing_pitch.take_engine import TakeEngine, score_take


def with_blip(frames, start_s, seconds, octaves=1):
    """The same frames with the pitch one or more octaves off from start_s for `seconds`: what the tracker does."""
    return [PitchFrame(f.t, f.hz * 2.0 ** octaves, f.confidence, f.loudness_db) if start_s <= f.t < start_s + seconds
            else f for f in frames]


def nearest(frames):
    engine = TakeEngine(NearestNote())
    engine.push(frames)
    return engine.stop().summary


# --- octave blips -----------------------------------------------------------------------------------------------

def melody_take(frames, notes=((0, 2.0),)):
    """Frames scored against a Melody that holds its notes: the blip stays inside the reference's held note."""
    return score_take(frames, Melody(reference_line(list(notes)), "x"))


def test_a_short_octave_blip_inside_a_held_note_does_not_move_steadiness():
    s = melody_take(with_blip(sung(hz_at(0), 2.0), 0.8, 0.16))
    assert s.steadiness_cents is not None and s.steadiness_cents < 1.0


def test_a_downward_blip_does_not_move_steadiness_either():
    s = melody_take(with_blip(sung(hz_at(0), 2.0), 0.8, 0.16, octaves=-1))
    assert s.steadiness_cents < 1.0


def test_a_short_octave_blip_does_not_set_the_range():
    s = nearest(with_blip(sung(hz_at(0), 2.0), 0.8, 0.16))            # 8 % of the frames: more than a percentile cuts
    assert (s.lowest_note, s.highest_note) == ("A4", "A4")


def test_a_note_sung_an_octave_up_and_held_is_not_a_blip():
    frames = sung(hz_at(0), 1.5) + sung(hz_at(1200), 1.5, start=1.5)
    s = nearest(frames)
    assert (s.lowest_note, s.highest_note) == ("A4", "A5")
    assert s.steadiness_cents < 1.0


def test_a_blip_at_the_start_of_a_note_is_dropped_too():
    s = nearest(with_blip(sung(hz_at(0), 2.0), 0.0, 0.1))
    assert (s.lowest_note, s.highest_note) == ("A4", "A4")


# --- range from the middle 90 % of the scored frames ------------------------------------------------------------

def test_a_stray_excursion_of_a_few_percent_of_the_frames_does_not_set_the_range():
    # 0.2 s of 6 s, 3 % of the frames, 9 semitones above the note: far less than an octave, so not a blip, only outlying
    frames = sung(hz_at(0), 3.0) + sung(hz_at(900), 0.2, start=3.0) + sung(hz_at(0), 2.8, start=3.2)
    s = nearest(frames)
    assert (s.lowest_note, s.highest_note) == ("A4", "A4")


def test_a_real_wide_range_is_still_reported():
    frames = (sung(hz_at(-900), 1.0) + sung(hz_at(-500), 1.0, start=1.0) + sung(hz_at(0), 1.0, start=2.0)
              + sung(hz_at(400), 1.0, start=3.0))
    s = nearest(frames)
    assert (s.lowest_note, s.highest_note) == ("C4", "C#5")


# --- Steadiness does not depend on Notes mode -------------------------------------------------------------------

STAIRS = [(-900, 1.5), (-700, 1.5), (-500, 1.5)]       # C4 D4 E4, close enough for one bar per Phrase


def wobble_free(notes):
    frames, edge = [], 0.0
    for cents, seconds in notes:
        frames += sung(hz_at(cents), seconds, start=edge)
        edge += seconds
    return frames


@pytest.mark.parametrize("length", [0.1, 0.5, 1.5, 4.0, PER_PHRASE])
def test_steadiness_is_the_same_whatever_the_minimum_note_length(length):
    frames = wobble_free(STAIRS)
    line = reference_line(STAIRS)
    off = score_take(frames, Melody(line, "x")).steadiness_cents
    on = score_take(frames, Melody(line, "x").notes(length)).steadiness_cents
    assert off < 1.0
    assert on == pytest.approx(off)


def test_a_wobbling_voice_gets_the_same_steadiness_in_notes_mode():
    wobbly = contour(lambda s: -900 + (30 if int(s / 0.1) % 2 else -30), 1.5)
    frames = wobbly + [PitchFrame(f.t + 1.5, f.hz, f.confidence, f.loudness_db)
                       for f in contour(lambda s: -700 + (30 if int(s / 0.1) % 2 else -30), 1.5)]
    line = reference_line([(-900, 1.5), (-700, 1.5)])
    off = score_take(frames, Melody(line, "x")).steadiness_cents
    assert off > 20
    assert score_take(frames, Melody(line, "x").notes(PER_PHRASE)).steadiness_cents == pytest.approx(off)


# --- the summary says how the octave varied --------------------------------------------------------------------

def test_the_summary_counts_the_phrases_at_each_octave():
    segs = VERSE + REST + CHORUS + REST + VERSE
    s = score_take(voice_of(segs), Melody(reference_of(segs), "x"))
    assert s.octave_phrases == ((-1, 2), (0, 1))


def test_a_take_at_one_octave_has_one_entry():
    s = score_take(voice_of(VERSE), Melody(reference_of(VERSE), "x"))
    assert s.octave_phrases == ((-1, 1),)


def test_a_take_scored_against_nearest_note_has_no_octave_spread():
    assert nearest(sung(hz_at(0), 1.0)).octave_phrases == ()


def melody_record(segs):
    engine = TakeEngine(Melody(reference_of(segs), "x"))
    engine.push(voice_of(segs))
    record = engine.stop()
    return dataclasses.replace(record, song_id="tulip")


def test_the_summary_line_shows_the_spread_when_phrases_were_sung_at_different_octaves():
    from sing_pitch.ui.main_window import describe

    text = describe(melody_record(VERSE + REST + CHORUS + REST + VERSE))
    assert "octave per Phrase: -1 ×2, 0 ×1" in text
    assert "octave shift" not in text


def test_the_summary_line_keeps_one_octave_shift_when_every_phrase_was_at_the_same_octave():
    from sing_pitch.ui.main_window import describe

    text = describe(melody_record(VERSE + REST + VERSE))
    assert "octave shift -1 (auto)" in text and "per Phrase" not in text

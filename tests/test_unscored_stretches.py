"""Unscored stretches through the Take engine seam: where the reference is unreliable, nothing is scored."""
from frames import HOP, hz_at
from melody_helpers import reference_line, voice_following, with_confidence

from sing_pitch.pitch import PitchFrame
from sing_pitch.reference import Melody
from sing_pitch.take_engine import TakeEngine, replay_scores, score_take

NOTES = [(-900, 1.5), (-700, 1.5), (-500, 1.5)]       # C4 D4 E4, the middle one made unreliable below


def test_an_injected_unreliable_stretch_is_left_unscored_and_out_of_every_number():
    line = with_confidence(reference_line(NOTES), 1.5, 3.0, 0.05)
    frames = voice_following(NOTES, 0.0)
    # the voice is wild exactly there: way sharp, and high above everything else
    frames = [PitchFrame(f.t, hz_at(1500), f.confidence, f.loudness_db) if 1.5 <= f.t < 3.0 else f for f in frames]
    scores = replay_scores(frames, Melody(line, "x"))
    inside = [s for s in scores if 1.6 <= s.t < 2.9]
    assert inside and all(s.midi is not None and s.band is None and s.cents is None for s in inside)
    s = score_take(frames, Melody(line, "x"))
    assert s.pct_in_tune >= 98.0 and s.median_abs_cents < 1.0 and s.tendency == "centred"
    assert (s.lowest_note, s.highest_note) == ("C4", "E4")          # the wild stretch is not part of the range
    assert s.steadiness_cents < 1.0
    assert s.n_voiced > s.n_scored                                  # it was sung, just not judged


def test_an_unscored_stretch_does_not_pollute_steadiness():
    line = with_confidence(reference_line([(-900, 3.0)]), 1.0, 2.0, 0.05)
    frames = voice_following([(-900, 3.0)], 0.0)
    wobble = [PitchFrame(f.t, hz_at(-900 + (90 if i % 2 else -90)), f.confidence, f.loudness_db)
              if 1.0 <= f.t < 2.0 else f for i, f in enumerate(frames)]
    s = score_take(wobble, Melody(line, "x"))
    assert s.steadiness_cents is None or s.steadiness_cents < 1.0   # each scored piece is steady on its own


def test_a_dip_of_a_few_frames_inside_a_sung_note_does_not_flicker_to_unscored():
    line = with_confidence(reference_line([(-900, 3.0)]), 1.50, 1.56, 0.05)        # 60 ms dip
    scores = replay_scores(voice_following([(-900, 3.0)], 0.0), Melody(line, "x"))
    assert all(s.band is not None for s in scores if 0.2 < s.t < 2.8)


def test_a_dip_that_lasts_a_quarter_second_is_an_unscored_stretch():
    line = with_confidence(reference_line([(-900, 3.0)]), 1.5, 1.75, 0.05)
    scores = replay_scores(voice_following([(-900, 3.0)], 0.0), Melody(line, "x"))
    assert all(s.band is None for s in scores if 1.55 < s.t < 1.70)
    assert all(s.band is not None for s in scores if 0.2 < s.t < 1.4 or 1.9 < s.t < 2.8)


def test_a_brief_reliable_blip_inside_an_unreliable_region_stays_unscored():
    line = with_confidence(reference_line([(-900, 3.0)]), 0.0, 3.0, 0.05)
    line = with_confidence(line, 1.50, 1.58, 0.9)                                   # 80 ms of "confidence"
    scores = replay_scores(voice_following([(-900, 3.0)], 0.0), Melody(line, "x"))
    assert all(s.band is None for s in scores)


def test_an_unscored_stretch_is_marked_for_the_display_in_song_time():
    line = with_confidence(reference_line(NOTES), 1.5, 3.0, 0.05)
    spans = Melody(line, "x").unscored_spans()
    assert len(spans) == 1 and abs(spans[0][0] - 1.5) < 0.02 and abs(spans[0][1] - 3.0) < 0.02


def test_live_scores_leave_the_unreliable_stretch_unscored_too():
    line = with_confidence(reference_line(NOTES), 1.5, 3.0, 0.05)
    live = TakeEngine(Melody(line, "x")).push(voice_following(NOTES, 0.0))
    assert all(s.band is None for s in live if 1.6 < s.t < 2.9 and s.midi is not None)
    assert all(s.band is not None for s in live if (0.2 < s.t < 1.4 or 3.2 < s.t < 4.3) and s.midi is not None)

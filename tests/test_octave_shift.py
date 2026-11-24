"""Octave shift through the Take engine seam: one whole-octave k per Take, never per note."""
import pytest
from melody_helpers import reference_line, voice_following

from sing_pitch.reference import Band, Melody
from sing_pitch.take_engine import LinearClock, TakeEngine, linear_song_times, replay_scores, score_take

MELODY = [(-900, 0.6), (-700, 0.6), (-500, 0.6), (-400, 0.6), (-200, 0.6), (-500, 0.6), (-900, 1.0)]  # C D E F G E C


def sung_octaves_from(notes, octaves):
    return [(cents + 1200 * octaves, s) for cents, s in notes]


@pytest.mark.parametrize("k", [-2, -1, 0, 1, 2])
def test_a_melody_sung_whole_octaves_away_has_that_octave_shift_and_full_marks(k):
    line = reference_line(MELODY)
    frames = voice_following(sung_octaves_from(MELODY, k), 0.2)
    s = score_take(frames, Melody(line, "do-re-mi"), song_times=linear_song_times(frames, 0.2))
    assert s.octave_shift == k
    assert s.pct_in_tune >= 98.0 and s.median_abs_cents < 1.0


def test_a_melody_sung_an_octave_down_is_all_green_in_the_replay_too():
    line = reference_line(MELODY)
    frames = voice_following(sung_octaves_from(MELODY, -1), 0.2)
    scores = replay_scores(frames, Melody(line, "x"), song_times=linear_song_times(frames, 0.2))
    assert all(x.band is Band.GREEN for x in scores[40:100])


def test_one_note_jumping_an_octave_mid_take_is_red_and_not_forgiven():
    line = reference_line([(-900, 2.0)])
    frames = voice_following([(-900, 1.2), (300, 0.8)], 0.0)      # C4 for 1.2 s, then C5 (+1200 cents) for 0.8 s
    scores = replay_scores(frames, Melody(line, "x"))
    s = score_take(frames, Melody(line, "x"))
    assert s.octave_shift == 0                                    # the majority sits at k = 0
    assert all(x.band is Band.RED for x in scores if x.t > 1.3 and x.midi is not None)
    assert all(x.band is Band.GREEN for x in scores if x.t < 1.1 and x.midi is not None)
    assert 55.0 < s.pct_in_tune < 65.0                            # 1.2 s of 2.0 s


def test_octave_shift_is_stored_in_the_table_and_a_rescore_reproduces_it(tmp_path):
    import csv
    from datetime import datetime

    from melody_helpers import library_song

    from sing_pitch.take_store import rescore_take, save_take, summary_path

    song = library_song(tmp_path, reference_line(MELODY))
    eng = TakeEngine(Melody.from_song(song), song_clock=LinearClock(0.2), started_at=datetime(2026, 11, 23, 21, 0, 0))
    eng.push(voice_following(sung_octaves_from(MELODY, -1), 0.2))
    saved = save_take(tmp_path, eng.stop())
    assert saved.summary.octave_shift == -1
    assert next(csv.DictReader(open(summary_path(tmp_path))))["octave_shift"] == "-1"
    assert rescore_take(tmp_path, saved.id).summary == saved.summary


def test_nearest_note_and_key_takes_never_get_an_octave_shift():
    from frames import hz_at, sung

    from sing_pitch.reference import NearestNote

    assert score_take(sung(hz_at(-1200), 1.0), NearestNote()).octave_shift == 0


def test_live_display_estimates_the_octave_from_the_first_seconds_then_keeps_it():
    line = reference_line(MELODY)
    frames = voice_following(sung_octaves_from(MELODY, -1), 0.0)
    scores = TakeEngine(Melody(line, "x")).push(frames)
    settled = [x for x in scores if 2.5 < x.t < 4.0 and x.band is not None]
    assert settled and all(x.band is Band.GREEN for x in settled)


def test_live_octave_is_fixed_once_about_two_seconds_are_scored_so_a_later_jump_stays_red():
    line = reference_line([(-900, 6.0)])
    frames = voice_following([(-900, 2.5), (-2100, 3.5)], 0.0)      # on the note, then an octave below it
    eng = TakeEngine(Melody(line, "x"))
    scores = [x for i in range(0, len(frames), 5) for x in eng.push(frames[i : i + 5])]
    assert all(x.band is Band.RED for x in scores if x.t > 3.0 and x.midi is not None)
    assert score_take(frames, Melody(line, "x")).octave_shift == -1     # the final summary is 3.5 s of -1

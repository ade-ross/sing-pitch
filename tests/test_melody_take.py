"""Melody Takes through the Take engine seam: frames + a Melody Reference in, scores and a TakeRecord out."""
from frames import HOP, hz_at, silence
from melody_helpers import reference_line, voice_following

from sing_pitch.pitch import PitchFrame
from sing_pitch.reference import Band, Melody
from sing_pitch.take_engine import LinearClock, TakeEngine, score_take

MELODY = [(-900, 0.6), (-700, 0.6), (-500, 0.6), (-400, 0.6), (-200, 0.6), (-500, 0.6), (-900, 1.0)]  # C D E F G E C


def test_voice_exactly_on_the_melody_but_late_by_the_latency_scores_100_in_tune():
    line = reference_line(MELODY)
    latency = 0.23
    engine = TakeEngine(Melody(line, song_id="do-re-mi"), song_clock=LinearClock(latency))
    scores = engine.push(voice_following(MELODY, latency))
    s = engine.stop().summary
    assert s.n_scored > 200
    assert s.pct_in_tune >= 98.0
    assert s.median_abs_cents < 1.0
    # the only misses allowed: a frame within half a reference hop of a note change, where "which note" is a tie
    changes = [latency + sum(sec for _, sec in MELODY[:k]) for k in range(1, len(MELODY))]
    misses = [x.t for x in scores if x.band is not None and x.band is not Band.GREEN]
    assert all(min(abs(t - c) for c in changes) <= 0.006 for t in misses)


def test_the_same_late_voice_without_latency_correction_is_scored_much_worse():
    line = reference_line(MELODY)
    frames = voice_following(MELODY, 0.23)
    # 230 ms late: the Timing window forgives 100 ms of it, the rest of each 0.6 s note change is still off
    assert score_take(frames, Melody(line, "do-re-mi")).pct_in_tune < 85.0


def test_a_voice_20_cents_sharp_of_the_melody_is_green_and_sharp_30_cents_is_yellow():
    line = reference_line([(-900, 1.0)])
    green = TakeEngine(Melody(line, "x"), song_clock=LinearClock(0.1)).push(voice_following([(-900, 1.0)], 0.1, cents_off=20))
    yellow = TakeEngine(Melody(line, "x"), song_clock=LinearClock(0.1)).push(voice_following([(-900, 1.0)], 0.1, cents_off=30))
    assert green[40].band is Band.GREEN and abs(green[40].cents - 20) < 0.5
    assert yellow[40].band is Band.YELLOW and yellow[40].note_name == "C4"


def test_singing_where_the_reference_is_unreliable_is_left_unscored():
    line = reference_line([(-900, 1.0)], confidence=0.1)       # RMVPE unsure everywhere
    eng = TakeEngine(Melody(line, "x"))
    scores = eng.push([PitchFrame(i * HOP, hz_at(-900), 0.98, -20.0) for i in range(50)])
    assert all(s.midi is not None and s.band is None and s.cents is None for s in scores)
    s = eng.stop().summary
    assert s.n_voiced == 50 and s.n_scored == 0 and s.pct_in_tune is None


def test_singing_before_the_song_starts_or_after_it_ends_is_unscored():
    line = reference_line([(-900, 1.0)])
    frames = [PitchFrame(t, hz_at(-900), 0.98, -20.0) for t in (0.05, 0.1, 0.15)]     # song starts at 0.3 s
    assert all(s.band is None for s in TakeEngine(Melody(line, "x"), song_clock=LinearClock(0.3)).push(frames))
    late = [PitchFrame(2.0 + i * HOP, hz_at(-900), 0.98, -20.0) for i in range(5)]
    assert all(s.band is None for s in TakeEngine(Melody(line, "x")).push(late))


def test_a_melody_take_records_its_song_and_latency():
    line = reference_line(MELODY)
    eng = TakeEngine(Melody(line, "do-re-mi"), song_clock=LinearClock(0.2))
    eng.push(voice_following(MELODY, 0.2) + silence(0.2, start=6.0))
    record = eng.stop()
    assert (record.reference_kind, record.song_id, record.latency_offset_s) == ("melody", "do-re-mi", 0.2)
    assert record.key is None


def test_a_saved_melody_take_has_its_song_in_the_table_and_rescores_from_the_song_id(tmp_path):
    import csv
    from datetime import datetime

    from melody_helpers import library_song

    from sing_pitch.take_store import rescore_take, save_take, saved_reference, summary_path

    line = reference_line(MELODY)
    song = library_song(tmp_path, line)
    eng = TakeEngine(Melody.from_song(song), song_clock=LinearClock(0.25), started_at=datetime(2026, 11, 23, 21, 0, 0))
    eng.push(voice_following(MELODY, 0.25))
    saved = save_take(tmp_path, eng.stop())

    row = next(csv.DictReader(open(summary_path(tmp_path))))
    assert (row["reference"], row["song_id"], float(row["latency_offset_s"])) == ("melody", song.id, 0.25)

    ref = saved_reference(tmp_path, saved.id)                  # no ValueError for Melody Takes any more
    assert ref.kind == "melody" and ref.song_id == song.id
    again = rescore_take(tmp_path, saved.id)                   # same song, same saved latency
    assert again.summary == saved.summary and again.summary.pct_in_tune >= 98.0

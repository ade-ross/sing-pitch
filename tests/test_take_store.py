import csv
from datetime import datetime

from frames import hz_at, silence, sung

from sing_pitch.reference import NearestNote
from sing_pitch.take_engine import TakeEngine, score_take
from sing_pitch.config import ScoreConfig
from sing_pitch.take_store import load_frames, rescore_take, save_take, summary_path


def make_take(started_at, cents=30):
    engine = TakeEngine(NearestNote(), started_at=started_at)
    engine.push(sung(hz_at(cents), 1.0) + silence(0.5, start=1.0))
    return engine.stop()


def test_a_saved_take_is_one_raw_line_file_plus_one_summary_row(tmp_path):
    record = make_take(datetime(2026, 11, 23, 21, 5, 9))
    saved = save_take(tmp_path, record)

    rows = list(csv.DictReader(open(summary_path(tmp_path))))
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == saved.id == "20261123-210509"
    assert row["reference"] == "nearest"
    assert row["started_at"] == "2026-11-23T21:05:09"
    assert float(row["pct_in_tune"]) == 0.0           # +30 cents is yellow throughout
    assert abs(float(row["median_abs_cents"]) - 30) < 0.5
    assert row["tendency"] == "sharp"
    assert int(row["n_scored_frames"]) == record.summary.n_scored
    assert [p.name for p in (tmp_path / "takes").iterdir()] == ["20261123-210509.npz"]


def test_the_raw_line_comes_back_exactly_and_rescores_to_the_same_numbers(tmp_path):
    record = make_take(datetime(2026, 11, 23, 21, 5, 9))
    save_take(tmp_path, record)
    frames = load_frames(tmp_path, record.id)
    assert frames == list(record.frames)
    assert score_take(frames, NearestNote()) == record.summary


def test_each_take_appends_a_row_and_the_header_is_written_once(tmp_path):
    save_take(tmp_path, make_take(datetime(2026, 11, 23, 21, 5, 9), cents=30))
    save_take(tmp_path, make_take(datetime(2026, 11, 23, 21, 9, 41), cents=10))
    lines = open(summary_path(tmp_path)).read().splitlines()
    assert len(lines) == 3 and lines[0].startswith("id,")
    assert len(list((tmp_path / "takes").iterdir())) == 2


def test_two_takes_in_the_same_second_never_overwrite_each_other(tmp_path):
    t = datetime(2026, 11, 23, 21, 5, 9)
    a = save_take(tmp_path, make_take(t, cents=30))
    b = save_take(tmp_path, make_take(t, cents=10))
    assert a.id != b.id
    assert len(list((tmp_path / "takes").iterdir())) == 2


def test_the_summary_row_carries_steadiness_tendency_and_range(tmp_path):
    from steadiness_helpers import vibrato

    engine = TakeEngine(NearestNote(), started_at=datetime(2026, 11, 23, 21, 5, 9))
    engine.push(vibrato(25, 6.0, 2.0, centre=20))
    save_take(tmp_path, engine.stop())
    row = next(csv.DictReader(open(summary_path(tmp_path))))
    assert abs(float(row["steadiness_cents"]) - 17.7) < 2.0
    assert row["tendency"] == "sharp"
    assert row["lowest_note"] == row["highest_note"] == "A4"


def test_rescoring_a_saved_take_gives_the_numbers_it_was_scored_with(tmp_path):
    record = save_take(tmp_path, make_take(datetime(2026, 11, 23, 21, 5, 9)))
    assert rescore_take(tmp_path, record.id).summary == record.summary


def test_rescoring_with_new_bands_changes_the_row_in_the_table(tmp_path):
    record = save_take(tmp_path, make_take(datetime(2026, 11, 23, 21, 5, 9), cents=30))   # +30: yellow
    save_take(tmp_path, make_take(datetime(2026, 11, 23, 21, 9, 41), cents=10))
    wider = ScoreConfig(green_cents=35.0)
    updated = rescore_take(tmp_path, record.id, wider, write=True)
    assert updated.summary.pct_in_tune == 100.0
    rows = {r["id"]: r for r in csv.DictReader(open(summary_path(tmp_path)))}
    assert float(rows[record.id]["pct_in_tune"]) == 100.0
    assert float(rows["20261123-210941"]["pct_in_tune"]) == 100.0
    assert len(rows) == 2 and rows[record.id]["started_at"] == "2026-11-23T21:05:09"


def test_a_key_take_rescores_and_replays_against_its_own_key_without_being_told_the_key(tmp_path):
    from sing_pitch.reference import Key
    from sing_pitch.take_store import rescore_take, saved_reference

    engine = TakeEngine(Key.parse("C major"), started_at=datetime(2026, 11, 23, 21, 5, 9))
    engine.push(sung(hz_at(-300), 1.0))                       # F#4: out of key in C major
    saved = save_take(tmp_path, engine.stop())
    assert saved_reference(tmp_path, saved.id) == Key.parse("C major")
    again = rescore_take(tmp_path, saved.id)
    assert again.summary.out_of_key_count == 1 and again.key == "C major"

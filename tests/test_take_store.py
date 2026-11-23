import csv
from datetime import datetime

from frames import hz_at, silence, sung

from sing_pitch.reference import NearestNote
from sing_pitch.take_engine import TakeEngine, score_take
from sing_pitch.take_store import load_frames, save_take, summary_path


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

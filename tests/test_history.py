import csv
from datetime import datetime

from sing_pitch.history import WHOLE_SONG, History, load_history
from sing_pitch.take_store import COLUMNS


def write_table(home, rows):
    with open(home / "takes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def row(i, day, reference="nearest", pct="50", med="20", song="", **kw):
    return {"id": i, "started_at": f"2026-11-{day:02d}T10:00:00", "reference": reference, "song_id": song,
            "pct_in_tune": pct, "median_abs_cents": med, **kw}


def test_no_table_gives_an_empty_history(tmp_path):
    h = load_history(tmp_path)
    assert h.is_empty and h.references == [] and h.songs == []
    s = h.series()
    assert s.started_at == [] and s.pct_in_tune == [] and s.median_abs_cents == []


def test_header_only_table_is_empty(tmp_path):
    write_table(tmp_path, [])
    assert load_history(tmp_path).is_empty


def test_series_is_in_date_order_with_both_metrics(tmp_path):
    write_table(tmp_path, [row("b", 20, pct="70", med="12.5"), row("a", 10, pct="55", med="20")])
    s = load_history(tmp_path).series()
    assert s.started_at == [datetime(2026, 11, 10, 10), datetime(2026, 11, 20, 10)]
    assert s.pct_in_tune == [55.0, 70.0]
    assert s.median_abs_cents == [20.0, 12.5]


def test_takes_with_nothing_scored_are_left_out(tmp_path):
    write_table(tmp_path, [row("a", 10, pct="", med=""), row("b", 11, pct="60", med="18")])
    s = load_history(tmp_path).series()
    assert s.pct_in_tune == [60.0]


def test_filter_by_reference(tmp_path):
    write_table(tmp_path, [
        row("a", 10, reference="nearest", pct="50"),
        row("b", 11, reference="key", pct="80"),
        row("c", 12, reference="nearest", pct="60"),
    ])
    h = load_history(tmp_path)
    assert h.references == ["key", "nearest"]
    assert h.series(reference="nearest").pct_in_tune == [50.0, 60.0]
    assert h.series(reference="key").pct_in_tune == [80.0]
    assert h.series(reference="melody").pct_in_tune == []
    assert h.series().pct_in_tune == [50.0, 80.0, 60.0]


def test_filter_by_song_only_when_melody_takes_exist(tmp_path):
    write_table(tmp_path, [
        row("a", 10, reference="nearest", pct="50"),
        row("b", 11, reference="melody", song="s1", pct="40"),
        row("c", 12, reference="melody", song="s2", pct="45"),
        row("d", 13, reference="melody", song="s1", pct="48"),
    ])
    h = load_history(tmp_path)
    assert h.songs == ["s1", "s2"]
    assert h.series(song="s1").pct_in_tune == [40.0, 48.0]
    assert h.series(reference="melody", song="s2").pct_in_tune == [45.0]
    assert h.series(reference="nearest", song="s1").pct_in_tune == []


def test_no_songs_means_empty_song_list(tmp_path):
    write_table(tmp_path, [row("a", 10)])
    assert load_history(tmp_path).songs == []


def test_old_table_without_song_column_still_loads(tmp_path):
    (tmp_path / "takes.csv").write_text(
        "id,started_at,reference,pct_in_tune,median_abs_cents\n"
        "a,2026-11-10T10:00:00,nearest,55,20\n")
    h = load_history(tmp_path)
    assert h.songs == [] and h.series().pct_in_tune == [55.0]


def test_a_garbled_row_is_skipped_not_fatal(tmp_path):
    (tmp_path / "takes.csv").write_text(
        "id,started_at,reference,pct_in_tune,median_abs_cents\n"
        "a,not-a-date,nearest,55,20\n"
        "b,2026-11-10T10:00:00,nearest,55,20\n")
    assert load_history(tmp_path).series().pct_in_tune == [55.0]


# --- filters on the practice settings (ticket #15) ------------------------------------------------------------------

def practice_table(home):
    write_table(home, [
        row("a", 10, reference="melody", song="s1", pct="40", guide_vocal="on", guide_gain="1.0", speed_pct="100"),
        row("b", 11, reference="melody", song="s1", pct="50", guide_vocal="off", guide_gain="0.0", speed_pct="100",
            section_name="Chorus 1", pass_index="1"),
        row("c", 12, reference="melody", song="s1", pct="60", guide_vocal="off", guide_gain="0.0", speed_pct="70",
            section_name="Chorus 1", pass_index="2"),
        row("d", 13, reference="melody", song="s1", pct="65", guide_vocal="mixed", guide_gain="0.8", speed_pct="70",
            section_name="Verse 2", pass_index="1"),
        row("e", 14, reference="nearest", pct="30"),              # no settings at all: an old row
    ])
    return load_history(home)


def test_filter_by_guide_vocal(tmp_path):
    h = practice_table(tmp_path)
    assert h.guide_states == ["off", "on", "mixed"]
    assert h.series(guide_vocal="on").pct_in_tune == [40.0]
    assert h.series(guide_vocal="mixed").pct_in_tune == [65.0]
    assert h.series(guide_vocal="off").pct_in_tune == [50.0, 60.0, 30.0]       # the old row reads as guide off


def test_filter_by_speed(tmp_path):
    h = practice_table(tmp_path)
    assert h.speeds == [70, 100]
    assert h.series(speed=70).pct_in_tune == [60.0, 65.0]
    assert h.series(speed=100).pct_in_tune == [40.0, 50.0, 30.0]               # the old row reads as Speed 100


def test_filter_by_section_or_whole_song(tmp_path):
    h = practice_table(tmp_path)
    assert h.sections == ["Chorus 1", "Verse 2"]
    assert h.series(section="Chorus 1").pct_in_tune == [50.0, 60.0]
    assert h.series(section=WHOLE_SONG).pct_in_tune == [40.0, 30.0]
    assert h.series(section="Bridge").pct_in_tune == []


def test_filters_combine(tmp_path):
    h = practice_table(tmp_path)
    assert h.series(section="Chorus 1", speed=70, guide_vocal="off").pct_in_tune == [60.0]
    assert h.series(section="Chorus 1", guide_vocal="on").pct_in_tune == []
    assert h.series(song="s1", reference="melody", speed=100).pct_in_tune == [40.0, 50.0]


def test_a_table_from_before_these_columns_has_only_off_and_100(tmp_path):
    (tmp_path / "takes.csv").write_text(
        "id,started_at,reference,pct_in_tune,median_abs_cents\n"
        "a,2026-11-10T10:00:00,nearest,55,20\n")
    h = load_history(tmp_path)
    assert (h.guide_states, h.speeds, h.sections) == (["off"], [100], [])

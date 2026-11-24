"""Tolerance (#41): how far from the target still counts as in tune, ±10 to ±100 cents (default ±25). It sets the green
Band; yellow reaches twice the Tolerance; red beyond. Each Take records the Tolerance and Timing window it was scored
with, and a re-score uses them. Rows saved before #41 read as ±25 with no Timing window and re-score as before."""
import csv
from datetime import datetime

import pytest
from frames import hz_at, silence, sung

from sing_pitch.config import DEFAULT_SCORE_CONFIG, ScoreConfig
from sing_pitch.reference import Band, NearestNote
from sing_pitch.take_engine import TakeEngine
from sing_pitch.take_store import rescore_take, row_config, save_take, summary_path


def last_band(config: ScoreConfig, cents: float) -> Band:
    return TakeEngine(NearestNote(), config).push(sung(hz_at(cents), 0.5))[-1].band


@pytest.mark.parametrize("tolerance, cents, band", [
    (25, 20, Band.GREEN), (25, 40, Band.YELLOW), (25, 60, Band.RED),
    (50, 45, Band.GREEN), (50, 90, Band.YELLOW), (50, 110, Band.RED),        # 110 cents: the next semitone is nearer,
    (10, 8, Band.GREEN), (10, 15, Band.YELLOW), (10, 22, Band.RED),
])
def test_the_tolerance_sets_green_and_yellow_reaches_twice_it(tolerance, cents, band):
    if cents > 50:                          # beyond half a semitone Nearest note picks the next note: use a fixed one
        from test_take_engine import FixedNote

        got = TakeEngine(FixedNote(69.0), ScoreConfig.with_tolerance(tolerance)).push(sung(hz_at(cents), 0.5))[-1].band
    else:
        got = last_band(ScoreConfig.with_tolerance(tolerance), cents)
    assert got is band


def test_the_default_tolerance_is_25_cents():
    assert (DEFAULT_SCORE_CONFIG.green_cents, DEFAULT_SCORE_CONFIG.yellow_cents) == (25.0, 50.0)
    assert ScoreConfig.with_tolerance(25) == DEFAULT_SCORE_CONFIG


def take(config: ScoreConfig, cents: float = 40):
    engine = TakeEngine(NearestNote(), config, started_at=datetime(2026, 11, 24, 21, 0, 0))
    engine.push(sung(hz_at(cents), 1.0) + silence(0.3, start=1.0))
    return engine.stop()


def test_a_take_saves_the_tolerance_and_timing_window_it_was_scored_with(tmp_path):
    save_take(tmp_path, take(ScoreConfig.with_tolerance(50)))
    (row,) = csv.DictReader(open(summary_path(tmp_path)))
    assert float(row["pct_in_tune"]) == 100.0                  # 40 cents off is green at ±50
    assert (row["tolerance_cents"], row["timing_window_s"], row["note_scoring"]) == ("50.0", "0.1", "note")


def test_a_rescore_uses_the_saved_tolerance_not_the_default(tmp_path):
    saved = save_take(tmp_path, take(ScoreConfig.with_tolerance(50)))
    again = rescore_take(tmp_path, saved.id)
    assert again.summary == saved.summary and again.summary.pct_in_tune == 100.0
    assert again.config == ScoreConfig.with_tolerance(50)


def test_a_row_saved_before_tolerance_reads_as_25_cents_and_no_timing_window():
    config = row_config({"id": "x", "pct_in_tune": "80"})
    assert (config.green_cents, config.yellow_cents, config.timing_window_s) == (25.0, 50.0, 0.0)
    assert config.notes_per_note is False                      # Notes-mode Takes were scored per frame then


def test_a_new_row_gives_back_its_config():
    config = row_config({"tolerance_cents": "35.0", "timing_window_s": "0.1", "note_scoring": "note"})
    assert config == ScoreConfig.with_tolerance(35)


def test_the_new_columns_go_at_the_end_of_the_table():
    from sing_pitch.take_store import COLUMNS

    assert COLUMNS[-3:] == ("tolerance_cents", "timing_window_s", "note_scoring")

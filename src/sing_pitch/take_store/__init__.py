"""Take store: each Take = raw pitch line (takes/<id>.npz) + one row in takes.csv. Nothing else."""
from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

import numpy as np

from .. import __version__
from ..pitch import PitchFrame
from ..take_engine import TakeRecord

# The ONE summary table. Columns for later tickets (steadiness, octave shift, key, song) exist from the start
# so the file never changes shape; values not measured yet are left empty.
COLUMNS = (
    "id", "started_at", "reference", "key", "song_id", "octave_shift", "pct_in_tune", "median_abs_cents",
    "tendency", "steadiness_cents", "lowest_note", "highest_note", "n_voiced_frames", "n_scored_frames",
    "duration_s", "out_of_key_count", "latency_offset_s", "app_version",
)


def summary_path(home: Path) -> Path:
    return Path(home) / "takes.csv"


def _take_path(home: Path, take_id: str) -> Path:
    return Path(home) / "takes" / f"{take_id}.npz"


def _row(record: TakeRecord) -> dict:
    s = record.summary
    cell = lambda v: "" if v is None else v  # noqa: E731
    return {
        "id": record.id, "started_at": record.started_at.isoformat(timespec="seconds"),
        "reference": record.reference_kind, "key": cell(record.key), "song_id": cell(record.song_id),
        "octave_shift": s.octave_shift, "pct_in_tune": cell(s.pct_in_tune),
        "median_abs_cents": cell(s.median_abs_cents), "tendency": cell(s.tendency),
        "steadiness_cents": cell(s.steadiness_cents), "lowest_note": cell(s.lowest_note),
        "highest_note": cell(s.highest_note), "n_voiced_frames": s.n_voiced, "n_scored_frames": s.n_scored,
        "duration_s": round(s.duration_s, 3), "out_of_key_count": s.out_of_key_count,
        "latency_offset_s": record.latency_offset_s, "app_version": __version__,
    }


def save_take(home: Path, record: TakeRecord) -> TakeRecord:
    """Write the raw line and append the summary row. Returns the record with its final id
    (a suffix is added when another Take already has the same second)."""
    home = Path(home)
    (home / "takes").mkdir(parents=True, exist_ok=True)
    take_id, n = record.id, 1
    while _take_path(home, take_id).exists():
        n += 1
        take_id = f"{record.id}-{n}"
    record = dataclasses.replace(record, id=take_id)

    f = record.frames
    np.savez(
        _take_path(home, take_id),
        t=np.array([x.t for x in f], dtype=np.float64),
        hz=np.array([x.hz for x in f], dtype=np.float64),
        confidence=np.array([x.confidence for x in f], dtype=np.float64),
        loudness_db=np.array([x.loudness_db for x in f], dtype=np.float64),
    )
    path = summary_path(home)
    new_file = not path.exists()
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(_row(record))
    return record


def load_frames(home: Path, take_id: str) -> list[PitchFrame]:
    with np.load(_take_path(home, take_id)) as z:
        return [
            PitchFrame(float(t), float(hz), float(c), float(db))
            for t, hz, c, db in zip(z["t"], z["hz"], z["confidence"], z["loudness_db"])
        ]

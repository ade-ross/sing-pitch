"""Take store: each Take = raw pitch line (takes/<id>.npz) + one row in takes.csv. Nothing else.

The npz also keeps `song_t`, the song time each frame was scored against (NaN = no song playing). Takes saved
before the song-time log have no song_t: their song times are t - latency_offset_s, as they were scored then.
A Pass Take also keeps `clock_t`, the clock's song time of each frame with the Lead-in not masked out, for drawing
the Reference on a replay (Takes without it draw from song_t)."""
from __future__ import annotations

import csv
import dataclasses
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import __version__
from ..config import DEFAULT_SCORE_CONFIG, ScoreConfig
from ..library.player_settings import DEFAULT_MIN_NOTE_S
from ..pitch import PitchFrame
from ..reference import Key, Melody, NearestNote, Reference
from ..take_engine import AUTO, LEGACY, OctaveSetting, TakeRecord, linear_song_times, pinned, score_take

# The ONE summary table. Columns for later tickets (steadiness, octave shift, key, song) exist from the start
# so the file never changes shape; values not measured yet are left empty.
COLUMNS = (
    "id", "started_at", "reference", "key", "song_id", "octave_shift", "pct_in_tune", "median_abs_cents",
    "tendency", "steadiness_cents", "lowest_note", "highest_note", "n_voiced_frames", "n_scored_frames",
    "duration_s", "out_of_key_count", "latency_offset_s", "app_version", "section_name", "pass_index",
    "guide_vocal", "guide_gain", "speed_pct", "octave_mode", "note_mode", "min_note_s",
)
# Columns added after the first release go at the END; a table with an older header is rewritten with the new one
# on the next save, its old rows left empty in the new columns (no Section = not a Pass; an empty guide_vocal /
# speed_pct reads as guide off / Speed 100, see row_settings; an empty octave_mode = one Octave shift per Take, how
# Takes were scored before Octave shift per Phrase, and they re-score that way, see row_octave; an empty note_mode =
# scored against the line, see row_notes).

GUIDE_STATES = ("off", "on", "mixed")


def row_settings(row: dict) -> tuple[str, float, int]:
    """(guide_vocal, guide_gain, speed_pct) of a table row; rows saved before these columns read as guide off, 0,
    Speed 100."""
    try:
        gain = float(row.get("guide_gain") or 0.0)
    except ValueError:
        gain = 0.0
    try:
        speed = int(float(row.get("speed_pct") or 100))
    except ValueError:
        speed = 100
    guide = row.get("guide_vocal") or "off"
    return (guide if guide in GUIDE_STATES else "off"), gain, speed


def row_octave(row: dict) -> OctaveSetting:
    """How a saved Take's Octave shift was chosen: "auto" per Phrase, "pinned" at the row's octave_shift, anything
    else (rows from before the column) one shift for the whole Take."""
    mode = row.get("octave_mode") or ""
    if mode == "auto":
        return AUTO
    if mode == "pinned":
        try:
            return pinned(int(float(row["octave_shift"])))
        except (KeyError, TypeError, ValueError):     # a pinned row that lost its shift: do not invent a pinned 0
            return LEGACY
    return LEGACY


def row_notes(row: dict) -> tuple[str, float | None]:
    """(note_mode, min_note_s) of a table row: ("on", length) for a Notes-mode Take (length math.inf when its cell is
    "phrase": one note per Phrase; a lost or unreadable length reads as DEFAULT_MIN_NOTE_S, the mode is kept), else
    ("off", None); rows saved before these columns read as off."""
    if row.get("note_mode") != "on":
        return "off", None
    cell = row.get("min_note_s") or ""
    if cell == "phrase":
        return "on", math.inf
    try:
        return "on", float(cell)
    except ValueError:
        return "on", DEFAULT_MIN_NOTE_S


def _length_cell(record: TakeRecord) -> str | float:
    if record.note_mode != "on" or record.min_note_s is None:
        return ""
    return "phrase" if math.isinf(record.min_note_s) else round(record.min_note_s, 3)


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
        "section_name": cell(record.section_name), "pass_index": cell(record.pass_index),
        "guide_vocal": record.guide_vocal, "guide_gain": record.guide_gain, "speed_pct": record.speed_pct,
        "octave_mode": s.octave_mode, "note_mode": record.note_mode, "min_note_s": _length_cell(record),
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
        **({} if record.song_times is None else
           {"song_t": _nan_array(record.song_times)}),
        **({} if record.clock_times is None else {"clock_t": _nan_array(record.clock_times)}),
    )
    path = summary_path(home)
    new_file = not path.exists()
    if not new_file:
        _upgrade_header(path)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(_row(record))
    return record


def _nan_array(times) -> np.ndarray:
    return np.array([np.nan if x is None else x for x in times], dtype=np.float64)


def _write_rows(path: Path, rows: list[dict]) -> None:
    """Rewrite the whole table atomically: a temp file next to it, then os.replace, so a crash or a full disk
    halfway leaves the old table whole."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS, restval="", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _upgrade_header(path: Path) -> None:
    """A table written by an older version (fewer columns): rewrite it once with the current header."""
    with open(path, newline="") as fh:
        header = next(csv.reader(fh), [])
    if tuple(header) != COLUMNS:
        _write_rows(path, _rows(path.parent))


def load_frames(home: Path, take_id: str) -> list[PitchFrame]:
    with np.load(_take_path(home, take_id)) as z:
        return [
            PitchFrame(float(t), float(hz), float(c), float(db))
            for t, hz, c, db in zip(z["t"], z["hz"], z["confidence"], z["loudness_db"])
        ]


def _song_times(home: Path, take_id: str, row: dict, frames: list[PitchFrame]) -> list[float | None]:
    with np.load(_take_path(home, take_id)) as z:
        if "song_t" in z.files:
            return [None if np.isnan(x) else float(x) for x in z["song_t"]]
    return linear_song_times(frames, float(row["latency_offset_s"] or 0.0))


def saved_song_times(home: Path, take_id: str) -> list[float | None]:
    """Per frame, the song time the saved Take was scored against (old Takes: t - latency_offset_s)."""
    row = next(r for r in _rows(home) if r["id"] == take_id)
    return _song_times(home, take_id, row, load_frames(home, take_id))


def saved_clock_times(home: Path, take_id: str) -> list[float | None]:
    """Per frame, the song time the singer heard, for drawing the Reference on a replay: a Pass Take's clock times
    (its Lead-in included), else the song times it was scored against."""
    with np.load(_take_path(home, take_id)) as z:
        if "clock_t" in z.files:
            return [None if np.isnan(x) else float(x) for x in z["clock_t"]]
    return saved_song_times(home, take_id)


def _rows(home: Path) -> list[dict]:
    path = summary_path(home)
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def last_take_id(home: Path) -> str | None:
    rows = _rows(home)
    return rows[-1]["id"] if rows else None


def _reference_for(row: dict, home: Path) -> Reference:
    """Rebuild a saved Take's Reference from its table row (Melody: the song is read from the library).
    Raises ValueError when it cannot be rebuilt (e.g. the song was removed from the library)."""
    if row["reference"] == "nearest":
        return NearestNote()
    if row["reference"] == "key" and row["key"]:
        return Key.parse(row["key"])
    if row["reference"] == "melody" and row["song_id"]:
        from ..library import Library

        try:
            melody = Melody.from_song(Library(home).get(row["song_id"]))
        except KeyError as e:
            raise ValueError(f"song {row['song_id']!r} of this Take is no longer in the library") from e
        mode, length = row_notes(row)
        return melody.notes(length) if mode == "on" else melody     # a Notes Take is scored against its bars
    raise ValueError(f"re-scoring a {row['reference']} Take needs its Reference")


def saved_reference(home: Path, take_id: str) -> Reference:
    return _reference_for(next(r for r in _rows(home) if r["id"] == take_id), home)


def saved_row(home: Path, take_id: str) -> dict:
    """The summary-table row of a saved Take, as read from takes.csv."""
    return next(r for r in _rows(home) if r["id"] == take_id)


def saved_octave(home: Path, take_id: str) -> OctaveSetting:
    """How the saved Take's Octave shift was chosen: what a replay or re-score must use to reproduce its numbers."""
    return row_octave(next(r for r in _rows(home) if r["id"] == take_id))


def saved_latency(home: Path, take_id: str) -> float:
    """The song clock's offset at the Take start (0 for Nearest note and Key): where the reference line is drawn."""
    row = next(r for r in _rows(home) if r["id"] == take_id)
    return float(row["latency_offset_s"] or 0.0)


def rescore_take(
    home: Path,
    take_id: str,
    config: ScoreConfig = DEFAULT_SCORE_CONFIG,
    reference: Reference | None = None,
    write: bool = False,
) -> TakeRecord:
    """Score a saved Take again from its raw pitch line (Bands or gate may have changed since).
    write=True also replaces the Take's row in the summary table. Nearest note rebuilds itself from the row;
    Key and Melody Takes rebuild theirs from the row (Melody via the song id in the library)."""
    rows = _rows(home)
    row = next(r for r in rows if r["id"] == take_id)
    given = reference is not None
    if reference is None:
        reference = _reference_for(row, home)
    frames = load_frames(home, take_id)
    latency = float(row["latency_offset_s"] or 0.0)
    song_times = _song_times(home, take_id, row, frames)
    guide_vocal, guide_gain, speed_pct = row_settings(row)
    note_mode, min_note_s = row_notes(row)
    if given:                                                  # a Reference of the caller's: its mode, not the row's
        min_note_s = getattr(reference, "min_note_s", None)
        note_mode = "off" if min_note_s is None else "on"
    record = TakeRecord(
        id=take_id,
        started_at=datetime.fromisoformat(row["started_at"]),
        reference_kind=row["reference"],
        frames=tuple(frames),
        summary=score_take(frames, reference, config, song_times, row_octave(row)),
        key=row["key"] or None,
        song_id=row["song_id"] or None,
        latency_offset_s=latency,
        song_times=tuple(song_times),
        section_name=row.get("section_name") or None,
        pass_index=int(row["pass_index"]) if row.get("pass_index") else None,
        guide_vocal=guide_vocal, guide_gain=guide_gain, speed_pct=speed_pct,
        note_mode=note_mode, min_note_s=min_note_s,
    )
    if write:
        _write_rows(summary_path(home), [_row(record) if r["id"] == take_id else r for r in rows])
    return record

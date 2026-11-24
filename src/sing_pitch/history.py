"""History: % in tune and median cents off per Take over dates, read from the summary table (takes.csv)."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .take_store import GUIDE_STATES, row_config, row_notes, row_settings, summary_path

WHOLE_SONG = ""      # the Section filter value for Takes that were not a Pass of a Section


@dataclass(frozen=True)
class HistoryPoint:
    started_at: datetime
    reference: str
    song: str | None
    pct_in_tune: float
    median_abs_cents: float
    guide_vocal: str = "off"          # "off" | "on" | "mixed"; rows saved before the column read as off
    speed_pct: int = 100
    section: str | None = None
    note_mode: str = "off"            # "on" = scored against the Notes-mode bars; rows saved before the column: off
    take_id: str = ""                 # the Take's id in the summary table (what a replay asks for)
    tolerance_cents: float = 25.0     # the Tolerance it was scored with; rows saved before the column: 25


@dataclass(frozen=True)
class Series:
    """What gets plotted: parallel lists, oldest Take first."""

    started_at: list[datetime]
    pct_in_tune: list[float]
    median_abs_cents: list[float]
    points: list[HistoryPoint] = field(default_factory=list)      # the same Takes in full, oldest first
    tolerance_cents: list[float] = field(default_factory=list)


class History:
    def __init__(self, points: list[HistoryPoint]):
        self._points = sorted(points, key=lambda p: p.started_at)

    @property
    def is_empty(self) -> bool:
        return not self._points

    @property
    def references(self) -> list[str]:
        return sorted({p.reference for p in self._points})

    @property
    def songs(self) -> list[str]:
        return sorted({p.song for p in self._points if p.song})

    @property
    def guide_states(self) -> list[str]:
        return [g for g in GUIDE_STATES if any(p.guide_vocal == g for p in self._points)]

    @property
    def speeds(self) -> list[int]:
        return sorted({p.speed_pct for p in self._points})

    @property
    def sections(self) -> list[str]:
        return sorted({p.section for p in self._points if p.section})

    @property
    def note_modes(self) -> list[str]:
        return [m for m in ("off", "on") if any(p.note_mode == m for p in self._points)]

    @property
    def tolerances(self) -> list[float]:
        return sorted({p.tolerance_cents for p in self._points})

    def series(self, reference: str | None = None, song: str | None = None, guide_vocal: str | None = None,
               speed: int | None = None, section: str | None = None, note_mode: str | None = None,
               tolerance: float | None = None) -> Series:
        """The Takes matching every filter given (None = any). section: a Section name, or WHOLE_SONG for Takes that
        were not a Pass. tolerance: the Tolerance in cents the Takes were scored with."""
        pts = [p for p in self._points
               if (reference is None or p.reference == reference) and (song is None or p.song == song)
               and (guide_vocal is None or p.guide_vocal == guide_vocal) and (speed is None or p.speed_pct == speed)
               and (section is None or (p.section or WHOLE_SONG) == section)
               and (note_mode is None or p.note_mode == note_mode)
               and (tolerance is None or p.tolerance_cents == tolerance)]
        return Series([p.started_at for p in pts], [p.pct_in_tune for p in pts], [p.median_abs_cents for p in pts],
                      tolerance_cents=[p.tolerance_cents for p in pts], points=pts)


def load_history(home: Path) -> History:
    """Read the summary table. Missing file, missing columns, Takes with nothing scored and bad rows are
    all tolerated: they just add no point."""
    path = summary_path(home)
    if not path.exists():
        return History([])
    points = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                guide, _, speed = row_settings(r)
                points.append(HistoryPoint(
                    started_at=datetime.fromisoformat(r["started_at"]),
                    reference=r["reference"],
                    song=(r.get("song_id") or r.get("song") or None),
                    pct_in_tune=float(r["pct_in_tune"]),
                    median_abs_cents=float(r["median_abs_cents"]),
                    guide_vocal=guide, speed_pct=speed, section=(r.get("section_name") or None),
                    note_mode=row_notes(r)[0], take_id=r.get("id") or "", tolerance_cents=row_config(r).green_cents,
                ))
            except (KeyError, TypeError, ValueError):
                continue
    return History(points)

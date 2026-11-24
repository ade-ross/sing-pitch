"""History: % in tune and median cents off per Take over dates, read from the summary table (takes.csv)."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .take_store import summary_path


@dataclass(frozen=True)
class HistoryPoint:
    started_at: datetime
    reference: str
    song: str | None
    pct_in_tune: float
    median_abs_cents: float


@dataclass(frozen=True)
class Series:
    """What gets plotted: parallel lists, oldest Take first."""

    started_at: list[datetime]
    pct_in_tune: list[float]
    median_abs_cents: list[float]


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

    def series(self, reference: str | None = None, song: str | None = None) -> Series:
        pts = [p for p in self._points
               if (reference is None or p.reference == reference) and (song is None or p.song == song)]
        return Series([p.started_at for p in pts], [p.pct_in_tune for p in pts], [p.median_abs_cents for p in pts])


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
                points.append(HistoryPoint(
                    started_at=datetime.fromisoformat(r["started_at"]),
                    reference=r["reference"],
                    song=(r.get("song_id") or r.get("song") or None),
                    pct_in_tune=float(r["pct_in_tune"]),
                    median_abs_cents=float(r["median_abs_cents"]),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    return History(points)

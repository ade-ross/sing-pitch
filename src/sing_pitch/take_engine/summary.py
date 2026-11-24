from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Sequence

from ..config import ScoreConfig
from ..music import note_name
from ..pitch import PitchFrame
from ..reference import Band, Reference
from .steadiness import steadiness_cents


@dataclass(frozen=True)
class TakeSummary:
    """Numbers for one Take. None = not measurable (no scored frames, or not built yet)."""

    n_voiced: int
    n_scored: int
    duration_s: float
    pct_in_tune: float | None            # green time / scored time, 0..100
    median_abs_cents: float | None
    tendency: Literal["sharp", "flat", "centred"] | None
    lowest_note: str | None
    highest_note: str | None
    steadiness_cents: float | None = None    # ticket #2
    octave_shift: int = 0                    # ticket #7
    out_of_key_count: int = 0                # Key Takes: out-of-key notes sung (a held note counts once)


@dataclass(frozen=True)
class PassTag:
    """A Take that is one Pass of a looped Section. Only voice answering song time inside [start_s, end_s) is scored:
    the Lead-in before the start never is."""

    section_name: str        # the saved name, or "start-end s" for an unnamed mark
    start_s: float
    end_s: float
    pass_index: int          # 1 = the first Pass after the loop was switched on

    def scores(self, song_t: float | None) -> bool:
        return song_t is not None and self.start_s <= song_t < self.end_s


@dataclass(frozen=True)
class TakeRecord:
    id: str
    started_at: datetime
    reference_kind: str
    frames: tuple[PitchFrame, ...]       # the RAW line: Bands/gate can change and the Take be re-scored
    summary: TakeSummary
    key: str | None = None
    song_id: str | None = None
    latency_offset_s: float = 0.0        # the song clock's offset at the start (linear Takes: the whole story)
    song_times: tuple[float | None, ...] | None = None   # per frame, the song time it was scored against
    section_name: str | None = None     # a Pass Take: its Section (None = not a Pass)
    pass_index: int | None = None


def count_out_of_key(scores: Sequence) -> int:
    """Out-of-key notes: runs of consecutive out-of-key frames. Silence or an in-key frame ends a run."""
    count, inside = 0, False
    for s in scores:
        out = s.midi is not None and s.in_key is False
        count += out and not inside
        inside = out
    return count


def summarise(scores: Sequence, frames: Sequence[PitchFrame], config: ScoreConfig) -> TakeSummary:
    voiced = [s for s in scores if s.midi is not None]
    scored = [s for s in voiced if s.cents is not None]
    duration = frames[-1].t - frames[0].t if len(frames) > 1 else 0.0
    if not scored:
        return TakeSummary(len(voiced), 0, duration, None, None, None, None, None)
    signed = statistics.median(s.cents for s in scored)
    if abs(signed) <= config.tendency_deadband_cents:
        tendency = "centred"
    else:
        tendency = "sharp" if signed > 0 else "flat"
    midis = [s.midi for s in scored]          # the range is what was judged: unscored stretches are left out
    return TakeSummary(
        n_voiced=len(voiced),
        n_scored=len(scored),
        duration_s=duration,
        pct_in_tune=100.0 * sum(s.band is Band.GREEN for s in scored) / len(scored),
        median_abs_cents=statistics.median(abs(s.cents) for s in scored),
        tendency=tendency,
        lowest_note=note_name(round(min(midis))),
        highest_note=note_name(round(max(midis))),
        out_of_key_count=count_out_of_key(scores),
        steadiness_cents=steadiness_cents(scores, config),
    )

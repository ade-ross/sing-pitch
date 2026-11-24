from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Sequence

import numpy as np

from ..config import DEFAULT_SCORE_CONFIG, ScoreConfig
from ..music import note_name
from ..pitch import PitchFrame
from ..reference import Band, Reference
from .blips import without_blips
from .note_scores import weighted_median
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
    octave_shift: int = 0                    # ticket #7: pinned k, or (Auto) the most common Phrase shift
    octave_mode: str = ""                    # ticket #17: "auto" | "pinned"; "" = one shift per Take (old Takes, Nearest note, Key)
    out_of_key_count: int = 0                # Key Takes: out-of-key notes sung (a held note counts once)
    octave_phrases: tuple[tuple[int, int], ...] = ()   # Auto Takes: (shift, Phrases) per shift used; not saved in takes.csv
    timing_shift_ms: float | None = None     # QA #44, Melody: median timing shift of the scored frames (+ = sung late)


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
    guide_vocal: str = "off"            # "on" | "off" | "mixed" (switched during the Take)
    guide_gain: float = 0.0             # the Guide vocal volume (0-1) while it was on; 0 when off
    speed_pct: int = 100                # Speed the Take was sung at (a Speed change starts a new Take)
    clock_times: tuple[float | None, ...] | None = None   # a Pass Take: the clock's song time of each frame before
                                        # the Lead-in was masked out, for drawing the Reference (None = song_times)
    note_mode: str = "off"              # "on" = scored against the Notes-mode bars (#19)
    min_note_s: float | None = None     # Notes mode: the Minimum note length (math.inf = one note per Phrase)
    config: ScoreConfig = DEFAULT_SCORE_CONFIG   # how it was scored: Tolerance, Timing window, Notes per note (#41)

    @property
    def tolerance_cents(self) -> float:
        return self.config.green_cents


def count_out_of_key(scores: Sequence) -> int:
    """Out-of-key notes: runs of consecutive out-of-key frames. Silence or an in-key frame ends a run."""
    count, inside = 0, False
    for s in scores:
        out = s.midi is not None and s.in_key is False
        count += out and not inside
        inside = out
    return count


def summarise(scores: Sequence, frames: Sequence[PitchFrame], config: ScoreConfig,
              verdicts: Sequence | None = None) -> TakeSummary:
    """verdicts: Notes mode per note (note_scores.NoteVerdict); % in tune is then the green share of the sung notes'
    length, and median cents and tendency the length-weighted medians of the notes' cents. None: frame by frame."""
    voiced = [s for s in scores if s.midi is not None]
    scored = [s for s in voiced if s.cents is not None]
    duration = frames[-1].t - frames[0].t if len(frames) > 1 else 0.0
    if not scored:
        return TakeSummary(len(voiced), 0, duration, None, None, None, None, None)
    if verdicts:
        weights = [v.length_s for v in verdicts]
        signed = weighted_median([v.cents for v in verdicts], weights)
        pct = 100.0 * sum(w for v, w in zip(verdicts, weights) if v.band is Band.GREEN) / sum(weights)
        median_abs = weighted_median([abs(v.cents) for v in verdicts], weights)
    else:
        signed = statistics.median(s.cents for s in scored)
        pct = 100.0 * sum(s.band is Band.GREEN for s in scored) / len(scored)
        median_abs = statistics.median(abs(s.cents) for s in scored)
    if abs(signed) <= config.tendency_deadband_cents:
        tendency = "centred"
    else:
        tendency = "sharp" if signed > 0 else "flat"
    steady = without_blips(scores, config)    # Steadiness and Range describe the voice: tracker octave blips are not it
    midis = [s.midi for s in steady if s.midi is not None and s.cents is not None]   # unscored stretches are left out
    midis = midis or [s.midi for s in scored]
    low, high = np.percentile(midis, config.range_percentiles)     # the middle of what was sung: a stray frame is not range
    return TakeSummary(
        n_voiced=len(voiced),
        n_scored=len(scored),
        duration_s=duration,
        pct_in_tune=pct,
        median_abs_cents=median_abs,
        tendency=tendency,
        lowest_note=note_name(round(float(low))),
        highest_note=note_name(round(float(high))),
        out_of_key_count=count_out_of_key(scores),
        steadiness_cents=steadiness_cents(steady, config),
    )

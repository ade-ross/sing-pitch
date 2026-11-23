from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from ..config import DEFAULT_SCORE_CONFIG, ScoreConfig
from ..music import hz_to_midi, note_name
from ..pitch import PitchFrame
from ..reference import Band, Reference
from .gate import VoicingGate, voiced_mask
from .summary import TakeRecord, TakeSummary, summarise


@dataclass(frozen=True)
class FrameScore:
    """One frame as the live view needs it. midi/cents/band/note_name are None when unvoiced."""

    t: float
    midi: float | None
    cents: float | None
    band: Band | None
    note_name: str | None
    in_key: bool | None = None      # Key Reference only: False = the sung note is outside the key


def band_for(cents: float, config: ScoreConfig) -> Band:
    a = abs(cents)
    if a <= config.green_cents:
        return Band.GREEN
    if a <= config.yellow_cents:
        return Band.YELLOW
    return Band.RED


def score_frame(frame: PitchFrame, voiced: bool, reference: Reference, config: ScoreConfig) -> FrameScore:
    if not voiced:
        return FrameScore(frame.t, None, None, None, None)
    midi = hz_to_midi(frame.hz)
    target = reference.target(frame.t, midi)
    cents = 100.0 * (midi - target.midi)
    return FrameScore(frame.t, midi, cents, band_for(cents, config), note_name(round(target.midi)), target.in_key)


def score_take(
    frames: Sequence[PitchFrame], reference: Reference, config: ScoreConfig = DEFAULT_SCORE_CONFIG
) -> TakeSummary:
    """Final numbers for a whole Take. Also the re-score path: raw frames in, same numbers out."""
    mask = voiced_mask(frames, config)
    scores = [score_frame(f, v, reference, config) for f, v in zip(frames, mask)]
    return summarise(scores, frames, config)


class TakeEngine:
    """Frames plus a Reference in; live Band and cents per frame out; a TakeRecord on stop()."""

    def __init__(
        self,
        reference: Reference,
        config: ScoreConfig = DEFAULT_SCORE_CONFIG,
        started_at: datetime | None = None,
    ):
        self._reference = reference
        self._config = config
        self._gate = VoicingGate(config)
        self._frames: list[PitchFrame] = []
        self._started_at = started_at or datetime.now()

    def push(self, frames: Sequence[PitchFrame]) -> list[FrameScore]:
        """Live scoring, one FrameScore per frame. Uses the streaming gate (no look-ahead, no lag);
        stop() applies the full gate, which also drops runs shorter than 48 ms."""
        self._frames.extend(frames)
        return [score_frame(f, self._gate.step(f), self._reference, self._config) for f in frames]

    def stop(self) -> TakeRecord:
        return TakeRecord(
            id=self._started_at.strftime("%Y%m%d-%H%M%S"),
            started_at=self._started_at,
            reference_kind=self._reference.kind,
            frames=tuple(self._frames),
            summary=score_take(self._frames, self._reference, self._config),
            key=getattr(self._reference, "name", None) if self._reference.kind == "key" else None,
        )

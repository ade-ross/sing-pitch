from __future__ import annotations

import dataclasses
import statistics
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


def score_frame(
    frame: PitchFrame, voiced: bool, reference: Reference, config: ScoreConfig, latency_s: float = 0.0,
    octave_shift: int = 0,
) -> FrameScore:
    """latency_s: how late the voice arrives on the Take clock relative to the Reference (Melody only).
    The voice frame at t is scored against the Reference at t - latency_s.
    octave_shift k: the voice is k octaves from the original singer, so the target moves by 12 * k semitones."""
    if not voiced:
        return FrameScore(frame.t, None, None, None, None)
    midi = hz_to_midi(frame.hz)
    target = reference.target(frame.t - latency_s, midi)
    if target.midi is None:       # sung, but the Reference has nothing reliable here: shown, never scored
        return FrameScore(frame.t, midi, None, None, None, target.in_key)
    target_midi = target.midi + 12 * octave_shift
    cents = 100.0 * (midi - target_midi)
    return FrameScore(frame.t, midi, cents, band_for(cents, config), note_name(round(target_midi)), target.in_key)


def best_octave_shift(pairs: Sequence[tuple[float, float]], config: ScoreConfig = DEFAULT_SCORE_CONFIG) -> int:
    """The whole octave k whose shifted Reference (target + 12 k) gives the smallest median absolute error over
    (sung MIDI, Reference MIDI) pairs. Ties go to the smaller shift. No pairs: 0."""
    if not pairs:
        return 0
    best, best_err = 0, None
    for k in config.octave_shifts:
        err = statistics.median(abs(100.0 * (sung - ref - 12 * k)) for sung, ref in pairs)
        if best_err is None or err < best_err - 1e-9:
            best, best_err = k, err
    return best


def _octave_pairs(
    frames: Sequence[PitchFrame], voiced: Sequence[bool], reference: Reference, latency_s: float
) -> list[tuple[float, float]]:
    """(sung MIDI, Reference MIDI) for every voiced frame the Reference scores."""
    pairs = []
    for f, v in zip(frames, voiced):
        if v:
            midi = hz_to_midi(f.hz)
            target = reference.target(f.t - latency_s, midi)
            if target.midi is not None:
                pairs.append((midi, target.midi))
    return pairs


def take_octave_shift(
    frames: Sequence[PitchFrame], reference: Reference, config: ScoreConfig = DEFAULT_SCORE_CONFIG,
    latency_s: float = 0.0,
) -> int:
    """The one Octave shift of a whole Take. Only a Melody has an original singer to be an octave away from."""
    if reference.kind != "melody":
        return 0
    return best_octave_shift(_octave_pairs(frames, voiced_mask(frames, config), reference, latency_s), config)


def replay_scores(
    frames: Sequence[PitchFrame], reference: Reference, config: ScoreConfig = DEFAULT_SCORE_CONFIG,
    latency_s: float = 0.0, octave_shift: int | None = None,
) -> list[FrameScore]:
    """A saved Take's line as live-view scores, one per frame, under the full gate (what the summary counted).
    octave_shift None = the whole Take's own shift, the same one score_take applies."""
    if octave_shift is None:
        octave_shift = take_octave_shift(frames, reference, config, latency_s)
    mask = voiced_mask(frames, config)
    return [score_frame(f, v, reference, config, latency_s, octave_shift) for f, v in zip(frames, mask)]


def score_take(
    frames: Sequence[PitchFrame], reference: Reference, config: ScoreConfig = DEFAULT_SCORE_CONFIG,
    latency_s: float = 0.0,
) -> TakeSummary:
    """Final numbers for a whole Take. Also the re-score path: raw frames in, same numbers out."""
    k = take_octave_shift(frames, reference, config, latency_s)
    scores = replay_scores(frames, reference, config, latency_s, octave_shift=k)
    return dataclasses.replace(summarise(scores, frames, config), octave_shift=k)


class TakeEngine:
    """Frames plus a Reference in; live Band and cents per frame out; a TakeRecord on stop()."""

    def __init__(
        self,
        reference: Reference,
        config: ScoreConfig = DEFAULT_SCORE_CONFIG,
        started_at: datetime | None = None,
        latency_s: float = 0.0,
    ):
        """latency_s: total delay of the voice behind the Reference on the Take clock (Melody: calibrated
        input + output latency, plus the playback start offset). Settable until the first push()."""
        self.latency_s = latency_s
        self._reference = reference
        self._config = config
        self._gate = VoicingGate(config)
        self._frames: list[PitchFrame] = []
        # LIVE octave: re-estimated from the first scored frames, then fixed (stop() uses the whole Take's own)
        self._live_pairs: list[tuple[float, float]] = []
        self._live_octave = 0
        self._started_at = started_at or datetime.now()

    def push(self, frames: Sequence[PitchFrame]) -> list[FrameScore]:
        """Live scoring, one FrameScore per frame. Uses the streaming gate (no look-ahead, no lag);
        stop() applies the full gate, which also drops runs shorter than 48 ms.

        Melody: the live Octave shift is the best one over the scored frames so far, re-estimated on every push
        until octave_live_lock_frames of them exist (~2 s of singing), then kept for the rest of the Take. The
        summary uses the whole Take's own shift, so the first seconds on screen can differ from it."""
        self._frames.extend(frames)
        voiced = [self._gate.step(f) for f in frames]
        if self._reference.kind == "melody" and len(self._live_pairs) < self._config.octave_live_lock_frames:
            self._live_pairs += _octave_pairs(frames, voiced, self._reference, self.latency_s)
            self._live_octave = best_octave_shift(self._live_pairs, self._config)
        return [
            score_frame(f, v, self._reference, self._config, self.latency_s, self._live_octave)
            for f, v in zip(frames, voiced)
        ]

    def stop(self) -> TakeRecord:
        return TakeRecord(
            id=self._started_at.strftime("%Y%m%d-%H%M%S"),
            started_at=self._started_at,
            reference_kind=self._reference.kind,
            frames=tuple(self._frames),
            summary=score_take(self._frames, self._reference, self._config, self.latency_s),
            key=getattr(self._reference, "name", None) if self._reference.kind == "key" else None,
            song_id=getattr(self._reference, "song_id", None) if self._reference.kind == "melody" else None,
            latency_offset_s=self.latency_s,
        )

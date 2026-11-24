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
from .clock import LinearClock, SongClock
from .gate import VoicingGate, voiced_mask
from .summary import PassTag, TakeRecord, TakeSummary, summarise


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
    frame: PitchFrame, voiced: bool, reference: Reference, config: ScoreConfig, song_t: float | None,
    octave_shift: int = 0,
) -> FrameScore:
    """song_t: the song time this voice frame answers (see clock.py); None = no song was playing then, so a sung
    frame is shown but never scored. Nearest note and Key ignore it.
    octave_shift k: the voice is k octaves from the original singer, so the target moves by 12 * k semitones."""
    if not voiced:
        return FrameScore(frame.t, None, None, None, None)
    midi = hz_to_midi(frame.hz)
    if song_t is None:
        return FrameScore(frame.t, midi, None, None, None)
    target = reference.target(song_t, midi)
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
    frames: Sequence[PitchFrame], voiced: Sequence[bool], reference: Reference, song_times: Sequence[float | None]
) -> list[tuple[float, float]]:
    """(sung MIDI, Reference MIDI) for every voiced frame the Reference scores."""
    pairs = []
    for f, v, song_t in zip(frames, voiced, song_times):
        if v and song_t is not None:
            midi = hz_to_midi(f.hz)
            target = reference.target(song_t, midi)
            if target.midi is not None:
                pairs.append((midi, target.midi))
    return pairs


def _song_times(frames: Sequence[PitchFrame], song_times: Sequence[float | None] | None) -> Sequence[float | None]:
    return [f.t for f in frames] if song_times is None else song_times


def take_octave_shift(
    frames: Sequence[PitchFrame], reference: Reference, config: ScoreConfig = DEFAULT_SCORE_CONFIG,
    song_times: Sequence[float | None] | None = None,
) -> int:
    """The one Octave shift of a whole Take. Only a Melody has an original singer to be an octave away from."""
    if reference.kind != "melody":
        return 0
    song_times = _song_times(frames, song_times)
    return best_octave_shift(_octave_pairs(frames, voiced_mask(frames, config), reference, song_times), config)


def replay_scores(
    frames: Sequence[PitchFrame], reference: Reference, config: ScoreConfig = DEFAULT_SCORE_CONFIG,
    song_times: Sequence[float | None] | None = None, octave_shift: int | None = None,
) -> list[FrameScore]:
    """A saved Take's line as live-view scores, one per frame, under the full gate (what the summary counted).
    song_times: per frame, the song time it answers (None = each frame's own t).
    octave_shift None = the whole Take's own shift, the same one score_take applies."""
    song_times = _song_times(frames, song_times)
    if octave_shift is None:
        octave_shift = take_octave_shift(frames, reference, config, song_times)
    mask = voiced_mask(frames, config)
    return [score_frame(f, v, reference, config, st, octave_shift) for f, v, st in zip(frames, mask, song_times)]


def score_take(
    frames: Sequence[PitchFrame], reference: Reference, config: ScoreConfig = DEFAULT_SCORE_CONFIG,
    song_times: Sequence[float | None] | None = None,
) -> TakeSummary:
    """Final numbers for a whole Take. Also the re-score path: raw frames + their saved song times in, same
    numbers out."""
    song_times = _song_times(frames, song_times)
    k = take_octave_shift(frames, reference, config, song_times)
    scores = replay_scores(frames, reference, config, song_times, octave_shift=k)
    return dataclasses.replace(summarise(scores, frames, config), octave_shift=k)


class TakeEngine:
    """Frames plus a Reference in; live Band and cents per frame out; a TakeRecord on stop()."""

    def __init__(
        self,
        reference: Reference,
        config: ScoreConfig = DEFAULT_SCORE_CONFIG,
        started_at: datetime | None = None,
        song_clock: SongClock | None = None,
        pass_tag: PassTag | None = None,
    ):
        """song_clock: voice time -> the song time it answers (Melody: a LogClock over the Transport's song-time
        log). Default: each frame's own time. pass_tag: this Take is one Pass of a looped Section; voice answering
        song time outside the Section (the Lead-in) gets no song time, so it is shown and never scored. Both are
        settable until the first push()."""
        self.song_clock: SongClock = song_clock or LinearClock()
        self.pass_tag = pass_tag
        self._song_times: list[float | None] = []
        self._clock_times: list[float | None] = []     # the clock's song times before the Pass mask (for drawing)
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
        song_times = [self.song_clock(f.t) for f in frames]
        self._clock_times.extend(song_times)
        if self.pass_tag is not None:
            song_times = [st if self.pass_tag.scores(st) else None for st in song_times]
        self._song_times.extend(song_times)
        voiced = [self._gate.step(f) for f in frames]
        if self._reference.kind == "melody" and len(self._live_pairs) < self._config.octave_live_lock_frames:
            self._live_pairs += _octave_pairs(frames, voiced, self._reference, song_times)
            self._live_octave = best_octave_shift(self._live_pairs, self._config)
        return [
            score_frame(f, v, self._reference, self._config, st, self._live_octave)
            for f, v, st in zip(frames, voiced, song_times)
        ]

    def stop(self) -> TakeRecord:
        return TakeRecord(
            id=self._started_at.strftime("%Y%m%d-%H%M%S"),
            started_at=self._started_at,
            reference_kind=self._reference.kind,
            frames=tuple(self._frames),
            summary=score_take(self._frames, self._reference, self._config, self._song_times),
            key=getattr(self._reference, "name", None) if self._reference.kind == "key" else None,
            song_id=getattr(self._reference, "song_id", None) if self._reference.kind == "melody" else None,
            latency_offset_s=self.song_clock.offset_s,
            song_times=tuple(self._song_times),
            section_name=None if self.pass_tag is None else self.pass_tag.section_name,
            pass_index=None if self.pass_tag is None else self.pass_tag.pass_index,
            clock_times=None if self.pass_tag is None else tuple(self._clock_times),
        )

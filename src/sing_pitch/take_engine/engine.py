from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from ..config import DEFAULT_SCORE_CONFIG, ScoreConfig
from ..music import hz_to_midi, note_name
from ..pitch import PitchFrame
from ..reference import Band, Reference
from .clock import LinearClock, SongClock
from .gate import VoicingGate, voiced_mask
from .octave import AUTO, OctaveSetting, PhraseTracker, TakeOctaves, best_octave_shift, take_octaves
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
    octave: int = 0                 # the Octave shift this frame was scored at: the Reference is drawn at it


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
        return FrameScore(frame.t, None, None, None, None, octave=octave_shift)
    midi = hz_to_midi(frame.hz)
    if song_t is None:
        return FrameScore(frame.t, midi, None, None, None, octave=octave_shift)
    target = reference.target(song_t, midi)
    if target.midi is None:       # sung, but the Reference has nothing reliable here: shown, never scored
        return FrameScore(frame.t, midi, None, None, None, target.in_key, octave_shift)
    target_midi = target.midi + 12 * octave_shift
    cents = 100.0 * (midi - target_midi)
    return FrameScore(frame.t, midi, cents, band_for(cents, config), note_name(round(target_midi)), target.in_key,
                      octave_shift)


def _song_times(frames: Sequence[PitchFrame], song_times: Sequence[float | None] | None) -> Sequence[float | None]:
    return [f.t for f in frames] if song_times is None else song_times


def _scored(
    frames: Sequence[PitchFrame], reference: Reference, config: ScoreConfig, song_times: Sequence[float | None],
    octave: OctaveSetting,
) -> tuple[list[FrameScore], TakeOctaves]:
    mask = voiced_mask(frames, config)
    octaves = take_octaves(frames, mask, reference, song_times, config, octave)
    scores = [score_frame(f, v, reference, config, st, k)
              for f, v, st, k in zip(frames, mask, song_times, octaves.per_frame)]
    return scores, octaves


def replay_scores(
    frames: Sequence[PitchFrame], reference: Reference, config: ScoreConfig = DEFAULT_SCORE_CONFIG,
    song_times: Sequence[float | None] | None = None, octave: OctaveSetting = AUTO,
) -> list[FrameScore]:
    """A saved Take's line as live-view scores, one per frame, under the full gate (what the summary counted).
    song_times: per frame, the song time it answers (None = each frame's own t).
    octave: how the Take's Octave shift was chosen (see octave.py); each score says which shift it was scored at."""
    return _scored(frames, reference, config, _song_times(frames, song_times), octave)[0]


def score_take(
    frames: Sequence[PitchFrame], reference: Reference, config: ScoreConfig = DEFAULT_SCORE_CONFIG,
    song_times: Sequence[float | None] | None = None, octave: OctaveSetting = AUTO,
) -> TakeSummary:
    """Final numbers for a whole Take. Also the re-score path: raw frames + their saved song times in, same
    numbers out."""
    scores, octaves = _scored(frames, reference, config, _song_times(frames, song_times), octave)
    return dataclasses.replace(summarise(scores, frames, config), octave_shift=octaves.shift,
                               octave_mode=octaves.mode)


class TakeEngine:
    """Frames plus a Reference in; live Band and cents per frame out; a TakeRecord on stop()."""

    def __init__(
        self,
        reference: Reference,
        config: ScoreConfig = DEFAULT_SCORE_CONFIG,
        started_at: datetime | None = None,
        song_clock: SongClock | None = None,
        pass_tag: PassTag | None = None,
        octave: OctaveSetting = AUTO,
    ):
        """song_clock: voice time -> the song time it answers (Melody: a LogClock over the Transport's song-time
        log). Default: each frame's own time. pass_tag: this Take is one Pass of a looped Section; voice answering
        song time outside the Section (the Lead-in) gets no song time, so it is shown and never scored. Both are
        settable until the first push(). octave: AUTO (a shift per Phrase) or pinned(k) for the whole Take."""
        self.song_clock: SongClock = song_clock or LinearClock()
        self.pass_tag = pass_tag
        self._song_times: list[float | None] = []
        self._clock_times: list[float | None] = []     # the clock's song times before the Pass mask (for drawing)
        self._reference = reference
        self._config = config
        self._gate = VoicingGate(config)
        self._frames: list[PitchFrame] = []
        # LIVE octave: each Phrase re-estimates its shift from its first scored frames, then keeps it for the rest of
        # that Phrase (stop() uses the whole Phrase's own)
        self._octave = octave
        self._phrases = PhraseTracker(config)
        self._phrase_pairs: list[tuple[float, float]] = []    # scored (sung, Reference) pairs of the open Phrase
        self._run_pairs: list[tuple[float, float]] = []       # of the voiced run that is not yet long enough to count
        self._phrase_locked = False
        self._live_octave = octave.k if octave.mode == "pinned" else 0
        self._started_at = started_at or datetime.now()

    def phrase_spans(self) -> list[tuple[int, int]]:
        """The Phrases seen so far as frame index ranges (see octave.split_phrases)."""
        return [(a, b) for a, b in self._phrases.spans]

    def _pair(self, frame: PitchFrame, song_t: float | None) -> tuple[float, float] | None:
        if song_t is None:
            return None
        midi = hz_to_midi(frame.hz)
        target = self._reference.target(song_t, midi)
        return None if target.midi is None else (midi, target.midi)

    def _live_shift(self, frame: PitchFrame, voiced: bool, song_t: float | None) -> int:
        """The Octave shift this frame is scored at. Auto, Melody: the open Phrase's best shift over its scored frames
        so far, re-estimated while the Phrase grows until octave_live_lock_frames of them exist (~2 s of singing),
        then kept for the rest of that Phrase. A new Phrase starts afresh. Frames of a voiced run too short to count
        yet are scored at the previous Phrase's shift (before the first Phrase: at the run's own best)."""
        opened = self._phrases.step(frame.t, voiced)          # every frame, whatever the mode: phrase_spans() reports it
        if self._reference.kind != "melody":
            return 0                                          # only a Melody has an original singer an octave away
        if self._octave.mode != "auto":
            return self._live_octave                          # pinned k, or 0 for a legacy Take
        if not voiced:
            self._run_pairs = []
            return self._live_octave
        if opened:
            self._phrase_pairs, self._phrase_locked = [], False
        counted = self._phrases.counted
        if self._phrase_locked and counted:
            return self._live_octave
        pair = self._pair(frame, song_t)
        if pair is not None:
            (self._phrase_pairs if counted else self._run_pairs).append(pair)
        if counted and self._run_pairs:
            self._phrase_pairs += self._run_pairs
            self._run_pairs = []
        if not counted and not self._phrases.spans and self._run_pairs:      # nothing sung yet: the run's own best
            return best_octave_shift(self._run_pairs, self._config)
        if counted and self._phrase_pairs:
            self._live_octave = best_octave_shift(self._phrase_pairs, self._config)
            self._phrase_locked = len(self._phrase_pairs) >= self._config.octave_live_lock_frames
        return self._live_octave

    def push(self, frames: Sequence[PitchFrame]) -> list[FrameScore]:
        """Live scoring, one FrameScore per frame. Uses the streaming gate (no look-ahead, no lag);
        stop() applies the full gate, which also drops runs shorter than 48 ms.

        Melody: the live Octave shift follows the Phrase being sung (see _live_shift). The summary uses each Phrase's
        whole-Phrase shift, so the first seconds of a Phrase on screen can differ from it."""
        self._frames.extend(frames)
        song_times = [self.song_clock(f.t) for f in frames]
        self._clock_times.extend(song_times)
        if self.pass_tag is not None:
            song_times = [st if self.pass_tag.scores(st) else None for st in song_times]
        self._song_times.extend(song_times)
        voiced = [self._gate.step(f) for f in frames]
        return [
            score_frame(f, v, self._reference, self._config, st, self._live_shift(f, v, st))
            for f, v, st in zip(frames, voiced, song_times)
        ]

    def stop(self) -> TakeRecord:
        return TakeRecord(
            id=self._started_at.strftime("%Y%m%d-%H%M%S"),
            started_at=self._started_at,
            reference_kind=self._reference.kind,
            frames=tuple(self._frames),
            summary=score_take(self._frames, self._reference, self._config, self._song_times, self._octave),
            key=getattr(self._reference, "name", None) if self._reference.kind == "key" else None,
            song_id=getattr(self._reference, "song_id", None) if self._reference.kind == "melody" else None,
            latency_offset_s=self.song_clock.offset_s,
            song_times=tuple(self._song_times),
            section_name=None if self.pass_tag is None else self.pass_tag.section_name,
            pass_index=None if self.pass_tag is None else self.pass_tag.pass_index,
            clock_times=None if self.pass_tag is None else tuple(self._clock_times),
        )

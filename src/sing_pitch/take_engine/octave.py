"""Octave shift: the whole-octave offset between Simone's voice and the original singer, and the Phrase it is chosen for.

A Phrase is a run of singing between pauses of about half a second or more (config.phrase_gap_s). Auto picks one
shift per Phrase (the whole octave k in -2..+2 whose shifted Reference is nearest, by median absolute error, ties to
the smaller |k|). A pinned shift applies to the whole Take. LEGACY is how Takes saved before Octave shift per Phrase
were scored: one shift for the whole Take; their rows have no octave_mode and re-score this way.

Phrase boundaries are a pure function of the frames and their voiced mask (split_phrases). The live view runs the
same PhraseTracker one frame at a time, so a Phrase starts and ends at the same frames live and on re-score; only the
two frames of a voiced run before it is long enough to count (min_voiced_run_frames) are scored live under the
previous Phrase."""
from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Literal, Sequence

from ..config import DEFAULT_SCORE_CONFIG, ScoreConfig
from ..music import hz_to_midi
from ..pitch import PitchFrame
from ..reference import Reference

PHRASE_GAP_EPS = 1e-9


@dataclass(frozen=True)
class OctaveSetting:
    mode: Literal["auto", "pinned", "legacy"]
    k: int = 0                       # the pinned shift


AUTO = OctaveSetting("auto")
LEGACY = OctaveSetting("legacy")


def pinned(k: int) -> OctaveSetting:
    return OctaveSetting("pinned", k)


OctavePair = tuple[float, tuple[float, ...]]     # (sung MIDI, Reference MIDI scored at each of config.octave_shifts)


def scoring_target(reference: Reference, song_t: float, midi: float, config: ScoreConfig):
    """What a sung frame is scored against: a Melody within the Timing window (midi: the voice moved to the original
    singer's octave, so the closest reference pitch is the one he means); the others have no timing."""
    if reference.kind == "melody" and config.timing_window_s > 0:
        return reference.target(song_t, midi, config.timing_window_s)
    return reference.target(song_t, midi)


def octave_pair(reference: Reference, song_t: float, midi: float, config: ScoreConfig) -> OctavePair | None:
    """The frame's Reference at every candidate shift k, each picked as scoring picks it (the Timing window's target
    for the voice moved by -12 k); None where the Reference scores nothing (that does not depend on k)."""
    if not (reference.kind == "melody" and config.timing_window_s > 0):
        target = reference.target(song_t, midi)
        return None if target.midi is None else (midi, (target.midi,) * len(config.octave_shifts))
    refs = []
    for k in config.octave_shifts:
        target = scoring_target(reference, song_t, midi - 12 * k, config)
        if target.midi is None:
            return None
        refs.append(target.midi)
    return midi, tuple(refs)


def best_octave_shift(pairs: Sequence[OctavePair], config: ScoreConfig = DEFAULT_SCORE_CONFIG) -> int:
    """The whole octave k whose shifted Reference (target + 12 k) gives the smallest median absolute error over
    the pairs (see octave_pair). Ties go to the smaller shift. No pairs: 0."""
    if not pairs:
        return 0
    best, best_err = 0, None
    for i, k in enumerate(config.octave_shifts):
        err = statistics.median(abs(100.0 * (sung - refs[i] - 12 * k)) for sung, refs in pairs)
        if best_err is None or err < best_err - 1e-9:
            best, best_err = k, err
    return best


def octave_pairs(
    frames: Sequence[PitchFrame], voiced: Sequence[bool], reference: Reference, song_times: Sequence[float | None],
    config: ScoreConfig = DEFAULT_SCORE_CONFIG,
) -> list[OctavePair]:
    """An octave_pair for every voiced frame the Reference scores."""
    pairs = []
    for f, v, song_t in zip(frames, voiced, song_times):
        if v and song_t is not None:
            pair = octave_pair(reference, song_t, hz_to_midi(f.hz), config)
            if pair is not None:
                pairs.append(pair)
    return pairs


class PhraseTracker:
    """Phrases of a stream of (time, voiced) frames. spans[i] = [first frame index, last voiced frame index + 1].
    A voiced run counts once it is min_voiced_run_frames long (the full gate drops shorter ones); it then belongs to
    the open Phrase when it starts less than phrase_gap_s after the last counted voiced frame, else it opens a new
    one."""

    def __init__(self, config: ScoreConfig = DEFAULT_SCORE_CONFIG):
        self._min, self._gap = config.min_voiced_run_frames, config.phrase_gap_s
        self.spans: list[list[int]] = []
        self._i = 0
        self._run_start = 0
        self._run_start_t = 0.0
        self._run_len = 0
        self._last_t = 0.0

    @property
    def counted(self) -> bool:
        """The voiced run in progress is long enough to count."""
        return self._run_len >= self._min

    def step(self, t: float, voiced: bool) -> bool:
        """Feed the next frame. True when this frame opened a new Phrase."""
        i, self._i = self._i, self._i + 1
        if not voiced:
            self._run_len = 0
            return False
        if self._run_len == 0:
            self._run_start, self._run_start_t = i, t
        self._run_len += 1
        opened = False
        if self._run_len == self._min:
            if self.spans and self._run_start_t - self._last_t < self._gap - PHRASE_GAP_EPS:
                self.spans[-1][1] = i + 1
            else:
                self.spans.append([self._run_start, i + 1])
                opened = True
        elif self._run_len > self._min:
            self.spans[-1][1] = i + 1
        if self._run_len >= self._min:
            self._last_t = t
        return opened


def split_phrases(
    frames: Sequence[PitchFrame], voiced: Sequence[bool], config: ScoreConfig = DEFAULT_SCORE_CONFIG
) -> list[tuple[int, int]]:
    """The Phrases of a Take as frame index ranges [start, end): first voiced frame to last voiced frame + 1."""
    tracker = PhraseTracker(config)
    for f, v in zip(frames, voiced):
        tracker.step(f.t, v)
    return [(a, b) for a, b in tracker.spans]


@dataclass(frozen=True)
class TakeOctaves:
    per_frame: list[int]            # the shift each frame is scored (and its Reference drawn) at
    shift: int                      # the Take's one number: pinned k, legacy k, else the most common Phrase shift
    mode: str                       # "auto" | "pinned" | "" (legacy, or not a Melody)
    phrases: tuple[tuple[int, int], ...] = ()   # Auto: (shift, number of Phrases) for each shift used, smallest first


def take_octaves(
    frames: Sequence[PitchFrame], voiced: Sequence[bool], reference: Reference, song_times: Sequence[float | None],
    config: ScoreConfig = DEFAULT_SCORE_CONFIG, octave: OctaveSetting = AUTO,
) -> TakeOctaves:
    """The final Octave shift of every frame of a whole Take. Only a Melody has an original singer to be an octave
    away from. Auto: each Phrase gets the shift that fits all of it; frames between Phrases keep the previous
    Phrase's (the first Phrase's before the first one). The Take's number is the most common Phrase shift, ties to
    the Phrase set with more scored frames, then the smaller |k|."""
    n = len(frames)
    if reference.kind != "melody":
        return TakeOctaves([0] * n, 0, "")
    if octave.mode == "pinned":
        return TakeOctaves([octave.k] * n, octave.k, "pinned")
    if octave.mode == "legacy":
        k = best_octave_shift(octave_pairs(frames, voiced, reference, song_times, config), config)
        return TakeOctaves([k] * n, k, "")
    spans = split_phrases(frames, voiced, config)
    shifts: list[tuple[int, int] | None] = []          # per Phrase: (k, pairs) or None when nothing in it is scored
    for a, b in spans:
        pairs = octave_pairs(frames[a:b], voiced[a:b], reference, song_times[a:b], config)
        shifts.append((best_octave_shift(pairs, config), len(pairs)) if pairs else None)
    known = [s[0] for s in shifts if s]
    per_frame, k_now, at = [], known[0] if known else 0, 0
    for (a, b), s in zip(spans, shifts):
        per_frame += [k_now] * (a - at)                # the gap before this Phrase: the previous Phrase's k
        k_now = s[0] if s else k_now
        per_frame += [k_now] * (b - a)
        at = b
    per_frame += [k_now] * (n - at)
    votes: Counter = Counter()
    weight: Counter = Counter()
    for s in shifts:
        if s:
            votes[s[0]] += 1
            weight[s[0]] += s[1]
    shift = min(votes, key=lambda k: (-votes[k], -weight[k], abs(k))) if votes else 0
    return TakeOctaves(per_frame, shift, "auto", tuple(sorted(votes.items())))

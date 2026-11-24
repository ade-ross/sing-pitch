"""Octave shift: the whole-octave offset between Simone's voice and the original singer, and the Phrase it is chosen for.

A Phrase is a run of singing between pauses of about half a second or more (config.phrase_gap_s). Auto picks one
shift per Phrase (the whole octave k in -2..+2 whose shifted Reference is nearest, by median absolute error, ties to
the smaller |k|). A pinned shift applies to the whole Take. LEGACY is how Takes saved before Octave shift per Phrase
were scored: one shift for the whole Take; their rows have no octave_mode and re-score this way.

Phrase boundaries are a pure function of the frames and their voiced mask (split_phrases). The live view runs the
same PhraseTracker one frame at a time, so a Phrase starts and ends at the same frames live and on re-score; only the
two frames of a voiced run before it is long enough to count (min_voiced_run_frames) are scored live under the
previous Phrase.

Timing shift (QA #44, "forgive timing, not pitch"): a Melody Phrase also gets ONE timing shift, how late (+) or early
(-) the voice is, from -timing_window_s to +timing_window_s in timing_step_s steps. The Octave shift k and the timing
shift are chosen TOGETHER, in one search: the pair (k, shift) whose Reference (the line at song time - shift, plus 12 k)
gives the smallest median absolute error over the Phrase's scored frames. Ties go to the earlier k in
config.octave_shifts (the smaller |k|), then to the smaller |shift|. A pinned Octave searches the shift only, at its k.
Every frame of the Phrase is then scored strictly at that shift: no per-frame choice, so a constant pitch error stays
in the numbers. Takes saved by #41 (timing_per_phrase False) keep their per-frame closest-in-window rule."""
from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from ..config import DEFAULT_SCORE_CONFIG, ScoreConfig
from ..music import hz_to_midi
from ..pitch import PitchFrame
from ..reference import Reference
from .note_scores import weighted_median

PHRASE_GAP_EPS = 1e-9
TIMING_TIE_CENTS = 0.5      # medians this close are a tie between two (Octave, timing) shifts
TIMING_CAP_CENTS = 300.0    # the tie-break's mean caps each frame's error here: a blip an octave off does not decide


@dataclass(frozen=True)
class OctaveSetting:
    mode: Literal["auto", "pinned", "legacy"]
    k: int = 0                       # the pinned shift


AUTO = OctaveSetting("auto")
LEGACY = OctaveSetting("legacy")


def pinned(k: int) -> OctaveSetting:
    return OctaveSetting("pinned", k)


OctavePair = tuple[float, tuple[float, ...]]     # (sung MIDI, Reference MIDI scored at each of config.octave_shifts)


def timing_lags(config: ScoreConfig) -> tuple[float, ...]:
    """The timing shifts a Phrase may get, in s (+ = the voice late), smallest |shift| first, late before early.
    Only 0 when the Take is not scored with a timing shift per Phrase (#41 Takes, or no Timing window)."""
    if not config.timing_per_phrase or config.timing_window_s <= 0 or config.timing_step_s <= 0:
        return (0.0,)
    n = int(round(config.timing_window_s / config.timing_step_s))
    lags = [0.0]
    for i in range(1, n + 1):
        lags += [round(i * config.timing_step_s, 9), -round(i * config.timing_step_s, 9)]
    return tuple(lags)


def shifts_timing(reference: Reference, config: ScoreConfig) -> bool:
    """This Take's Phrases each get a timing shift: a Melody scored per Phrase with a Timing window."""
    return reference.kind == "melody" and len(timing_lags(config)) > 1


def scoring_target(reference: Reference, song_t: float, midi: float, config: ScoreConfig, lag_s: float = 0.0):
    """What a sung frame is scored against: a Melody at its Phrase's timing shift lag_s (or, a #41 Take, the closest
    reference within the Timing window; midi: the voice moved to the original singer's octave); the others have no
    timing."""
    if reference.kind == "melody" and config.timing_window_s > 0:
        if shifts_timing(reference, config):
            return reference.target(song_t, midi, lag_s=lag_s)
        return reference.target(song_t, midi, config.timing_window_s)
    return reference.target(song_t, midi)


def octave_pair(reference: Reference, song_t: float, midi: float, config: ScoreConfig) -> OctavePair | None:
    """The frame's Reference at every candidate shift k, each picked as scoring picks it (the Timing window's target
    for the voice moved by -12 k); None where the Reference scores nothing (that does not depend on k).
    Timing shift per Phrase: (sung MIDI, the Reference at each of timing_lags), the same for every k."""
    if shifts_timing(reference, config):
        row = reference.shifted_midi([song_t], timing_lags(config))[0]
        return None if np.isnan(row).all() else (midi, tuple(float(x) for x in row))
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


def best_timing(pairs: Sequence[OctavePair], config: ScoreConfig, ks: Sequence[int]) -> tuple[int, float]:
    """(k, timing shift) jointly: of the Octave shifts ks and every timing shift, the pair whose Reference gives the
    smallest median absolute error over the pairs (octave_pair with a timing shift per Phrase). Medians within
    TIMING_TIE_CENTS of each other are a tie (on held notes more than half the frames match at several shifts, all with
    median 0), broken by the mean error with each frame capped at TIMING_CAP_CENTS (so an octave blip does not decide),
    then the earlier k, then the smaller |shift|. No usable pairs: (ks[0], 0)."""
    lags = timing_lags(config)
    if pairs:
        sung = np.array([p[0] for p in pairs], dtype=np.float64)
        refs = np.array([p[1] for p in pairs], dtype=np.float64)
        keep = np.isfinite(refs).all(axis=1)     # Notes mode: a frame whose shifted frame has no bar is left out
    if not pairs or not keep.any():
        return ks[0], 0.0
    diff = sung[keep, None] - refs[keep]
    best, best_err, best_mean = (ks[0], 0.0), None, None
    for k in ks:
        err = np.abs(100.0 * (diff - 12 * k))
        medians, means = np.median(err, axis=0), np.minimum(err, TIMING_CAP_CENTS).mean(axis=0)
        for lag, med, mean in zip(lags, medians, means):
            if (best_err is None or med < best_err - TIMING_TIE_CENTS
                    or (med <= best_err + TIMING_TIE_CENTS and mean < best_mean - 1e-9)):
                best, best_err, best_mean = (k, lag), float(med), float(mean)
    return best


def octave_pairs(
    frames: Sequence[PitchFrame], voiced: Sequence[bool], reference: Reference, song_times: Sequence[float | None],
    config: ScoreConfig = DEFAULT_SCORE_CONFIG,
) -> list[OctavePair]:
    """An octave_pair for every voiced frame the Reference scores."""
    if shifts_timing(reference, config):              # one lookup for all the frames
        idx = [i for i, (v, st) in enumerate(zip(voiced, song_times)) if v and st is not None]
        if not idx:
            return []
        rows = reference.shifted_midi([song_times[i] for i in idx], timing_lags(config))
        return [(hz_to_midi(frames[i].hz), tuple(float(x) for x in row))
                for i, row in zip(idx, rows) if not np.isnan(row).all()]
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
    per_frame_lag: tuple[float, ...] = ()       # timing shift per Phrase: each frame's, in s (empty = all 0)
    timing_shift_s: float | None = None         # the Take's median timing shift over its scored frames (+ = late)


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
    if shifts_timing(reference, config):
        return _take_timing(frames, voiced, reference, song_times, config, octave)
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


def _take_timing(
    frames: Sequence[PitchFrame], voiced: Sequence[bool], reference: Reference, song_times: Sequence[float | None],
    config: ScoreConfig, octave: OctaveSetting,
) -> TakeOctaves:
    """take_octaves for a Melody scored with a timing shift per Phrase: each Phrase's (k, shift) chosen together
    (best_timing); frames between Phrases keep the previous Phrase's (the first Phrase's before the first one)."""
    n = len(frames)
    if octave.mode == "legacy":                       # one shift for the whole Take (not saved since #17; for safety)
        ks: tuple[int, ...] = (best_timing(octave_pairs(frames, voiced, reference, song_times, config), config,
                                           config.octave_shifts)[0],)
    else:
        ks = (octave.k,) if octave.mode == "pinned" else tuple(config.octave_shifts)
    spans = split_phrases(frames, voiced, config)
    fits: list[tuple[int, float, int] | None] = []    # per Phrase: (k, shift, pairs) or None when nothing is scored
    for a, b in spans:
        pairs = octave_pairs(frames[a:b], voiced[a:b], reference, song_times[a:b], config)
        fits.append((*best_timing(pairs, config, ks), len(pairs)) if pairs else None)
    known = [f for f in fits if f]
    per_k, per_lag = [], []
    k_now, lag_now = (known[0][0], known[0][1]) if known else (ks[0], 0.0)
    at = 0
    for (a, b), f in zip(spans, fits):
        per_k += [k_now] * (a - at)
        per_lag += [lag_now] * (a - at)
        if f:
            k_now, lag_now = f[0], f[1]
        per_k += [k_now] * (b - a)
        per_lag += [lag_now] * (b - a)
        at = b
    per_k += [k_now] * (n - at)
    per_lag += [lag_now] * (n - at)
    timing = weighted_median([f[1] for f in known], [f[2] for f in known]) if known else None
    if octave.mode != "auto":
        return TakeOctaves(per_k, ks[0], "pinned" if octave.mode == "pinned" else "", (), tuple(per_lag), timing)
    votes: Counter = Counter()
    weight: Counter = Counter()
    for f in known:
        votes[f[0]] += 1
        weight[f[0]] += f[2]
    shift = min(votes, key=lambda k: (-votes[k], -weight[k], abs(k))) if votes else 0
    return TakeOctaves(per_k, shift, "auto", tuple(sorted(votes.items())), tuple(per_lag), timing)

"""Notes mode: the reference melody as flat notes on exact semitones.

segment_notes is a pure function of (frame times, reference MIDI, scored mask, Minimum note length). The scored frames
are split into reference Phrases (scored runs separated by a pause of PHRASE_GAP_S or more); inside a Phrase each scored
frame is rounded to its semitone, runs of the same semitone are the first notes, and then the shortest note under the
Minimum note length is merged into a neighbour (the one nearer in pitch; ties: the longer, then the earlier) until none
is left that short, or the Phrase is one note; two neighbours left on the same semitone become one note. A note's semitone is the one held longest over its frames (time-weighted
mode, ties to the one sung first), never the mean: a slide E-F-F#-G held on G is a G. Notes never cross a Phrase, and an
Unscored stretch inside a note stays unscored (only scored frames take the note's semitone). PER_PHRASE (the far end of
the slider) gives one note per Phrase."""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

PER_PHRASE = math.inf      # Minimum note length at the far end of the slider: one note per reference Phrase
PHRASE_GAP_S = 0.5         # a pause this long in the reference ends a reference Phrase (as config.phrase_gap_s)
MIN_NOTE_RANGE = (0.1, 4.0)
_EPS = 1e-9


@dataclass(frozen=True)
class Note:
    start: int      # first frame of the note
    end: int        # last frame + 1 (frames in between that are unscored stay unscored)
    midi: int       # the semitone


def _phrases(t: np.ndarray, ok: np.ndarray, gap_s: float) -> list[np.ndarray]:
    """The frame indices of each reference Phrase: scored frames, split where the time from one scored frame to the next
    is gap_s or more (as PhraseTracker splits the voice)."""
    idx = np.flatnonzero(ok)
    if not len(idx):
        return []
    dt = np.diff(t[idx])
    cuts = np.flatnonzero(dt >= gap_s - _EPS) + 1
    return np.split(idx, cuts)


def _mode(hist: dict[int, int]) -> int:
    """The semitone with the most frames; ties to the one that came first (dicts keep first-seen order)."""
    return max(hist.items(), key=lambda kv: kv[1])[0]


def _merge(a: dict[int, int], b: dict[int, int]) -> dict[int, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out


def _phrase_notes(semis: np.ndarray, min_frames: float) -> list[tuple[int, int, int]]:
    """(first, last + 1, semitone) in positions of one Phrase's frame list. Segments sit in a linked list and the
    shortest is found with a heap (stale entries skipped), so a long Phrase of many short runs costs n log n."""
    edges = np.flatnonzero(np.diff(semis)) + 1
    starts = np.concatenate([[0], edges]).tolist()
    ends = np.concatenate([edges, [len(semis)]]).tolist()
    n = len(starts)
    hist: list[dict[int, int]] = [{int(semis[a]): b - a} for a, b in zip(starts, ends)]
    if math.isinf(min_frames):
        total: dict[int, int] = {}
        for h in hist:
            total = _merge(total, h)
        return [(0, len(semis), _mode(total))]
    mode = [_mode(h) for h in hist]
    prev = list(range(-1, n - 1))
    nxt = list(range(1, n + 1))
    stamp = [0] * n
    alive = n
    heap = [(ends[i] - starts[i], starts[i], i, 0) for i in range(n)]
    heapq.heapify(heap)

    def join(lo: int, hi: int) -> None:           # hi (the next segment after lo) is absorbed into lo
        nonlocal alive
        ends[lo] = ends[hi]
        hist[lo] = _merge(hist[lo], hist[hi])
        mode[lo] = _mode(hist[lo])
        stamp[hi] = -1
        stamp[lo] += 1
        nxt[lo] = nxt[hi]
        if nxt[hi] < n:
            prev[nxt[hi]] = lo
        alive -= 1

    while alive > 1 and heap:
        length, _, i, version = heap[0]
        if stamp[i] != version:
            heapq.heappop(heap)
            continue
        if length >= min_frames - _EPS:
            break
        heapq.heappop(heap)
        left, right = prev[i], nxt[i]
        options = [j for j in (left, right) if 0 <= j < n]
        j = min(options, key=lambda j: (abs(mode[j] - mode[i]), -(ends[j] - starts[j]), j))   # left first on a tie
        lo, hi = min(i, j), max(i, j)
        join(lo, hi)
        if nxt[lo] < n and mode[lo] == mode[nxt[lo]]:   # a blip merged away: the held note on either side of it is
            join(lo, nxt[lo])                           # one note again
        if prev[lo] >= 0 and mode[prev[lo]] == mode[lo]:
            lo = prev[lo]
            join(lo, nxt[lo])
        heapq.heappush(heap, (ends[lo] - starts[lo], starts[lo], lo, stamp[lo]))
    out, k = [], 0
    while k < n:
        out.append((starts[k], ends[k], mode[k]))
        k = nxt[k]
    return out


def segment_notes(t, midi, scored, min_note_s: float, phrase_gap_s: float = PHRASE_GAP_S) -> list[Note]:
    """The notes of a reference line. t: frame times (s); midi: reference MIDI per frame; scored: the scored mask (the
    rest are Unscored stretches); min_note_s: the Minimum note length, PER_PHRASE for one note per Phrase."""
    t = np.asarray(t, dtype=np.float64)
    midi = np.asarray(midi, dtype=np.float64)
    ok = np.asarray(scored, dtype=bool) & np.isfinite(midi)
    hop = float(np.median(np.diff(t))) if len(t) > 1 else 0.01
    min_frames = round(min_note_s / hop) if math.isfinite(min_note_s) else math.inf     # whole frames, as scored_mask
    notes = []
    for idx in _phrases(t, ok, phrase_gap_s):
        semis = np.round(midi[idx]).astype(np.int64)
        for a, b, semi in _phrase_notes(semis, min_frames):
            notes.append(Note(int(idx[a]), int(idx[b - 1]) + 1, semi))
    return notes


def note_line(n_frames: int, scored, notes: list[Note]) -> tuple[np.ndarray, np.ndarray]:
    """(MIDI, note index) per frame: the note's semitone on its scored frames, NaN and -1 elsewhere."""
    midi = np.full(n_frames, np.nan)
    index = np.full(n_frames, -1, dtype=np.int64)
    scored = np.asarray(scored, dtype=bool)
    for k, n in enumerate(notes):
        on = scored[n.start:n.end]
        midi[n.start:n.end][on] = n.midi
        index[n.start:n.end][on] = k
    return midi, index


def format_length(min_note_s: float) -> str:
    """How a Minimum note length is written: '0.3 s', or 'per Phrase'."""
    return "per Phrase" if math.isinf(min_note_s) else f"{min_note_s:.1f} s"

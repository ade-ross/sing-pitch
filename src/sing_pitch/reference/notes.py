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
    """(first, last + 1, semitone) in positions of one Phrase's frame list."""
    edges = np.flatnonzero(np.diff(semis)) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(semis)]])
    segs = [[int(a), int(b), {int(semis[a]): int(b - a)}] for a, b in zip(starts, ends)]
    if math.isinf(min_frames):
        hist: dict[int, int] = {}
        for s in segs:
            hist = _merge(hist, s[2])
        return [(0, len(semis), _mode(hist))]
    while len(segs) > 1:
        short = [i for i, s in enumerate(segs) if s[1] - s[0] < min_frames - _EPS]
        if not short:
            break
        i = min(short, key=lambda i: (segs[i][1] - segs[i][0], i))
        me = _mode(segs[i][2])
        options = [j for j in (i - 1, i + 1) if 0 <= j < len(segs)]
        j = min(options, key=lambda j: (abs(_mode(segs[j][2]) - me), -(segs[j][1] - segs[j][0]), j))
        lo, hi = min(i, j), max(i, j)
        segs[lo] = [segs[lo][0], segs[hi][1], _merge(segs[lo][2], segs[hi][2])]
        del segs[hi]
        for k in (lo + 1, lo):      # a blip merged away: the held note on either side of it is one note again
            if 0 < k < len(segs) and _mode(segs[k - 1][2]) == _mode(segs[k][2]):
                segs[k - 1] = [segs[k - 1][0], segs[k][1], _merge(segs[k - 1][2], segs[k][2])]
                del segs[k]
    return [(a, b, _mode(h)) for a, b, h in segs]


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

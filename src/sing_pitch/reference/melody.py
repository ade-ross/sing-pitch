"""The Melody Reference: the pitch line RMVPE extracted from a song's vocal stem, looked up at song time t."""
from __future__ import annotations

import hashlib
from typing import ClassVar

import numpy as np

from ..library.ports import ReferenceLine
from ..music import A4_MIDI
from ..config import A4_HZ
from .base import Target
from .notes import Note, note_line, segment_notes

_NOTES_CACHE: dict[tuple, list[Note]] = {}     # (song, line digest, Minimum note length) -> its notes
_NOTES_CACHE_MAX = 64

MIN_CONFIDENCE = 0.3      # RMVPE activation below this = reference unreliable (04-rmvpe notes)
FILL_GAP_S = 0.12         # a dip shorter than this between two reliable frames is a consonant, not a stretch
MIN_SCORED_S = 0.12       # a reliable run shorter than this is a blip: too short to judge a voice against


def _runs(flags: np.ndarray) -> list[tuple[int, int, bool]]:
    """(start, end, value) of each run of equal values, end exclusive."""
    edges = np.flatnonzero(np.diff(flags.astype(np.int8))) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(flags)]])
    return [(int(a), int(b), bool(flags[a])) for a, b in zip(starts, ends)]


def scored_mask(
    line: ReferenceLine, min_confidence: float = MIN_CONFIDENCE, fill_gap_s: float = FILL_GAP_S,
    min_scored_s: float = MIN_SCORED_S,
) -> np.ndarray:
    """Which reference frames are scored; the rest are Unscored stretches. Confidence >= min_confidence, then
    smoothed so the mask does not flicker: low-confidence gaps shorter than fill_gap_s between two reliable
    frames count as reliable, then reliable runs shorter than min_scored_s count as unreliable."""
    mask = _reliable(line, min_confidence)
    if len(mask) < 2:
        return mask
    hop = float(np.median(np.diff(np.asarray(line.t, dtype=np.float64))))
    # compare whole frames: the float32 hop (0.0099999998) would put a 12-frame run just under 0.12 s
    fill_frames, min_frames = round(fill_gap_s / hop), round(min_scored_s / hop)
    for start, end, value in _runs(mask):
        if not value and start > 0 and end < len(mask) and end - start < fill_frames:
            mask[start:end] = True
    for start, end, value in _runs(mask):
        if value and end - start < min_frames:
            mask[start:end] = False
    return mask


def _reliable(line: ReferenceLine, min_confidence: float) -> np.ndarray:
    """Frames whose own Hz can be trusted: confident, and a real positive frequency."""
    hz = np.asarray(line.hz, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return (np.asarray(line.confidence) >= min_confidence) & np.isfinite(hz) & (hz > 0)


def _midi_line(line: ReferenceLine, min_confidence: float = MIN_CONFIDENCE) -> np.ndarray:
    """MIDI per frame from the reliable frames only. RMVPE's Hz at low confidence is arbitrary (99 to 1316 Hz on
    silence plus noise; 04-rmvpe notes), so frames the smoothing filled are interpolated between their reliable
    neighbours and held at the edges, never read from their own Hz. NaN when no frame is reliable."""
    ok = _reliable(line, min_confidence)
    if not ok.any():
        return np.full(len(ok), np.nan)
    hz = np.asarray(line.hz, dtype=np.float64)
    midi = A4_MIDI + 12.0 * np.log2(hz[ok] / A4_HZ)
    return np.interp(np.arange(len(ok)), np.flatnonzero(ok), midi)


class Melody:
    """Scores the voice against the original singer's line. t is song time (the engine has already removed
    the latency). Frames where the reference is unreliable, or outside the song, give an unscored Target."""

    kind: ClassVar[str] = "melody"

    def __init__(self, line: ReferenceLine, song_id: str, scored: np.ndarray | None = None):
        self.line = line
        self.song_id = song_id
        self._t = np.asarray(line.t, dtype=np.float64)
        self._midi = _midi_line(line)
        self._line_midi = self._midi          # a NotesMelody replaces _midi with its bars; the line stays
        self.scored = scored_mask(line) if scored is None else np.asarray(scored, dtype=bool)
        hop = float(np.median(np.diff(self._t))) if len(self._t) > 1 else 0.01
        self._tolerance = 1.5 * hop

    @classmethod
    def from_song(cls, song) -> Melody:
        return cls(song.load_reference(), song.id)

    @property
    def end_s(self) -> float | None:
        """Where the line (the vocal stem, as long as the song) ends in song time; None for an empty line."""
        return float(self._t[-1] + self._tolerance / 1.5) if len(self._t) else None

    def unscored_spans(self) -> list[tuple[float, float]]:
        """(start, end) in song time of every Unscored stretch: what the live and replay views grey out."""
        if not len(self._t):
            return []
        hop = self._tolerance / 1.5
        return [(float(self._t[a] - hop / 2), float(self._t[b - 1] + hop / 2))
                for a, b, scored in _runs(self.scored) if not scored]

    def midi_line(self) -> tuple[np.ndarray, np.ndarray]:
        """(song time, MIDI) of the line, NaN where unscored: what the live view draws."""
        return self._t, np.where(self.scored, self._midi, np.nan)

    def _frames_at(self, song_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(nearest frame, inside) for each song time: inside = a frame lies within the tolerance of it."""
        with np.errstate(invalid="ignore"):
            j = np.clip(np.searchsorted(self._t, song_t), 1, len(self._t) - 1) if len(self._t) > 1 else np.zeros(
                song_t.shape, dtype=int)
            if len(self._t) > 1:
                j = np.where(song_t - self._t[j - 1] < self._t[j] - song_t, j - 1, j)
            inside = np.isfinite(song_t) & (np.abs(self._t[j] - song_t) <= self._tolerance)
        return j, inside

    def sample(self, song_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(MIDI, unscored) of the line at each song time, for drawing it on any clock. MIDI is NaN where the line is
        unscored or the time is outside it (or NaN); unscored is True only inside an Unscored stretch."""
        song_t = np.asarray(song_t, dtype=np.float64)
        midi = np.full(song_t.shape, np.nan)
        unscored = np.zeros(song_t.shape, dtype=bool)
        if not len(self._t):
            return midi, unscored
        j, inside = self._frames_at(song_t)
        ok = inside & self.scored[j] & np.isfinite(self._midi[j])
        midi[ok] = self._midi[j][ok]
        unscored[inside] = ~self.scored[j][inside]
        return midi, unscored

    def notes(self, min_note_s: float) -> NotesMelody:
        """This Melody in Notes mode at the given Minimum note length (notes.PER_PHRASE: one note per Phrase). The
        segmentation is cached per song, line and length."""
        digest = hashlib.sha1(np.ascontiguousarray(self._t).tobytes() + np.ascontiguousarray(self._midi).tobytes()
                              + np.packbits(self.scored).tobytes()).hexdigest()
        key = (self.song_id, digest, float(min_note_s))
        notes = _NOTES_CACHE.get(key)
        if notes is None:
            if len(_NOTES_CACHE) >= _NOTES_CACHE_MAX:
                _NOTES_CACHE.pop(next(iter(_NOTES_CACHE)))
            notes = _NOTES_CACHE[key] = segment_notes(self._t, self._midi, self.scored, min_note_s)
        return NotesMelody(self, float(min_note_s), notes)

    def target(self, t: float, sung_midi: float) -> Target:
        if not len(self._t):
            return Target(None)
        j = int(np.searchsorted(self._t, t))
        if j == len(self._t) or (j > 0 and t - self._t[j - 1] < self._t[j] - t):
            j -= 1
        if abs(self._t[j] - t) > self._tolerance or not self.scored[j] or not np.isfinite(self._midi[j]):
            return Target(None)
        return Target(float(self._midi[j]), line_midi=float(self._line_midi[j]))


class NotesMelody(Melody):
    """A Melody in Notes mode: its line is the notes' exact semitones (see notes.py), so the target, the drawing, the
    replay and the re-score all use the bars. Unscored stretches are the Melody's own."""

    def __init__(self, melody: Melody, min_note_s: float, notes: list[Note]):
        super().__init__(melody.line, melody.song_id, melody.scored)
        self.min_note_s = min_note_s
        self.note_list = notes
        self._midi, self._note_index = note_line(len(self._t), self.scored, notes)

    def notes(self, min_note_s: float) -> NotesMelody:
        return Melody(self.line, self.song_id, self.scored).notes(min_note_s)

    def note_index_at(self, song_t: np.ndarray) -> np.ndarray:
        """Which note sounds at each song time (its index in note_list), -1 where none does."""
        song_t = np.asarray(song_t, dtype=np.float64)
        index = np.full(song_t.shape, -1, dtype=np.int64)
        if len(self._t):
            j, inside = self._frames_at(song_t)
            index[inside] = self._note_index[j][inside]
        return index

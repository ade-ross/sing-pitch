"""Notes mode, per note (#41): each note of a NotesMelody is scored once, on the voice's median over it.

A frame belongs to the note it was scored against (FrameScore.note, after the Timing window). A note's verdict is the
median of its frames' cents (the voice's median pitch minus the bar) and its Band under the Tolerance. Only notes with
at least one scored frame count ("sung notes"); each weighs its length (its scored reference frames). The live view
shows a verdict once the song is past the note's end plus the Timing window (NoteTally); the summary uses the same
verdicts for the whole Take (note_verdicts), so both agree."""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..config import ScoreConfig
from ..reference import Band, Reference


@dataclass(frozen=True)
class NoteVerdict:
    note: int            # index in the NotesMelody's note_list
    cents: float         # median of the voice's cents from the bar over the note
    band: Band
    length_s: float      # the note's length: its weight in % in tune


def scored_per_note(reference: Reference, config: ScoreConfig) -> bool:
    return config.notes_per_note and hasattr(reference, "note_list")


def _band(cents: float, config: ScoreConfig) -> Band:
    a = abs(cents)
    return Band.GREEN if a <= config.green_cents else Band.YELLOW if a <= config.yellow_cents else Band.RED


class NoteTally:
    """The notes' verdicts as a Take goes on: add() frame scores, close_until(song time) the notes that have ended."""

    def __init__(self, reference, config: ScoreConfig):
        self._ends, self._lengths = reference.note_spans_s()
        self._config = config
        self._cents: dict[int, list[float]] = {}

    def add(self, scores: Iterable) -> None:
        for s in scores:
            if s.note is not None and s.cents is not None and s.note >= 0:
                self._cents.setdefault(s.note, []).append(s.cents)

    def close_until(self, song_s: float) -> list[NoteVerdict]:
        """Verdicts of the notes sung so far that end (plus the Timing window) at or before song_s, each once, in note
        order."""
        window = self._config.timing_window_s
        done = sorted(k for k in self._cents if self._ends[k] + window <= song_s)
        out = []
        for k in done:
            cents = statistics.median(self._cents.pop(k))
            out.append(NoteVerdict(k, cents, _band(cents, self._config), float(self._lengths[k])))
        return out


def note_verdicts(scores: Sequence, reference, config: ScoreConfig) -> list[NoteVerdict]:
    """Every sung note's verdict over a whole Take, in note order."""
    tally = NoteTally(reference, config)
    tally.add(scores)
    return tally.close_until(float("inf"))


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    """The value where half the total weight lies at or below it."""
    pairs = sorted(zip(values, weights))
    half, run = sum(weights) / 2, 0.0
    for v, w in pairs:
        run += w
        if run >= half - 1e-12:
            return v
    return pairs[-1][0]

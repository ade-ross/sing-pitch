"""Steadiness: SD in cents around each held note's own median, attack and release left out."""
from __future__ import annotations

import math
import statistics
from typing import Sequence

from ..config import ScoreConfig


def _held_note(s) -> str | None:
    """The note a frame belongs to: the unmerged line's, so the Notes slider does not change what is held."""
    return s.line_note or s.note_name


def _notes(scores: Sequence) -> list[list]:
    """Held notes = runs of consecutive voiced, scored frames measured against the same note."""
    notes: list[list] = []
    run: list = []
    for s in scores:
        if s.midi is None or s.note_name is None:
            if run:
                notes.append(run)
            run = []
        elif run and _held_note(run[-1]) != _held_note(s):
            notes.append(run)
            run = [s]
        else:
            run.append(s)
    if run:
        notes.append(run)
    return notes


def steadiness_cents(scores: Sequence, config: ScoreConfig) -> float | None:
    """Root-mean-square spread (cents) of the sung pitch around each note's own median, over the stable middle of
    every note held longer than the minimum. None when no note was held long enough."""
    squares: list[float] = []
    for note in _notes(scores):
        start, end = note[0].t, note[-1].t
        if end - start < config.steadiness_min_note_s:
            continue
        trim = max(config.steadiness_trim_s, config.steadiness_trim_fraction * (end - start))
        middle = [100.0 * s.midi for s in note if start + trim <= s.t <= end - trim]
        if not middle:
            continue
        centre = statistics.median(middle)
        squares += [(c - centre) ** 2 for c in middle]
    if not squares:
        return None
    return math.sqrt(sum(squares) / len(squares))

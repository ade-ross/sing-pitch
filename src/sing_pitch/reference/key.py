from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from .base import Target

MAJOR = (0, 2, 4, 5, 7, 9, 11)
NATURAL_MINOR = (0, 2, 3, 5, 7, 8, 10)

# Spelling used in the picker and the Take table, one per pitch class. Both spellings are accepted by parse().
_MAJOR_NAMES = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
_MINOR_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B")
_SPELLINGS = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7,
    "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11,
}


@dataclass(frozen=True)
class Key:
    """The Key Reference: a major or natural-minor scale. A sung note is in key when the equal-tempered note
    closest to it is in the scale; in-key notes are scored against that note, out-of-key notes against the
    closest in-key note (so they land in the red Band) and flagged."""

    tonic: int                      # pitch class 0..11, C = 0
    mode: str                       # "major" | "minor"

    kind: ClassVar[str] = "key"

    @property
    def scale(self) -> frozenset[int]:
        steps = MAJOR if self.mode == "major" else NATURAL_MINOR
        return frozenset((self.tonic + s) % 12 for s in steps)

    @property
    def name(self) -> str:
        names = _MAJOR_NAMES if self.mode == "major" else _MINOR_NAMES
        return f"{names[self.tonic]} {self.mode}"

    @classmethod
    def all(cls) -> list[Key]:
        """The picker list: 12 major keys then 12 natural-minor keys."""
        return [cls(t, m) for m in ("major", "minor") for t in range(12)]

    @classmethod
    def parse(cls, name: str) -> Key:
        note, mode = name.split()
        return cls(_SPELLINGS[note.upper()], mode.lower())

    def target(self, t: float, sung_midi: float) -> Target:
        nearest = math.floor(sung_midi + 0.5)
        scale = self.scale
        if nearest % 12 in scale:
            return Target(midi=float(nearest), in_key=True)
        in_key = [m for m in (nearest - 1, nearest + 1) if m % 12 in scale]
        best = min(in_key, key=lambda m: (abs(sung_midi - m), m))
        return Target(midi=float(best), in_key=False)

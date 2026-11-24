from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol


class Band(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass(frozen=True)
class Target:
    """What a sung moment is measured against."""

    midi: float | None            # target pitch as float MIDI; None = unscored (Melody only)
    in_key: bool | None = None    # Key Reference only: False = outside the key
    line_midi: float | None = None   # Melody only: the line itself where midi is a Notes-mode bar (None = same as midi)


class Reference(Protocol):
    """What "in tune" is measured against: Nearest note, Key or Melody."""

    kind: Literal["nearest", "key", "melody"]

    def target(self, t: float, sung_midi: float) -> Target:
        """t = time on the Take clock (already latency-corrected by the engine). A Melody also takes window_s, the
        Timing window (see Melody.target)."""
        ...

from __future__ import annotations

import math

from .base import Target


class NearestNote:
    """The closest equal-tempered note, like a tuner."""

    kind = "nearest"

    def target(self, t: float, sung_midi: float) -> Target:
        return Target(midi=float(math.floor(sung_midi + 0.5)))

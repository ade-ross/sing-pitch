from __future__ import annotations

from typing import Protocol

import numpy as np

from .frame import PitchFrame


class PitchTracker(Protocol):
    """Chunks in, frames out. One tracker instance serves many Takes: call reset() at each Start."""

    def reset(self) -> None: ...

    def push(self, chunk: np.ndarray, sample_rate: int) -> list[PitchFrame]: ...

    def flush(self) -> list[PitchFrame]: ...

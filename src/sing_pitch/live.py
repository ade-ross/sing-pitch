"""Glue without a UI: audio chunks -> pitch tracker -> Take engine. Qt-free and device-free, so it is testable."""
from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np

from .audio_input import AudioInput
from .pitch import PitchTracker
from .reference import Reference
from .take_engine import FrameScore, TakeEngine, TakeRecord
from .take_store import save_take


class LiveTake:
    """One Take, fed chunk by chunk. Drive from ONE thread."""

    def __init__(self, tracker: PitchTracker, reference: Reference, sample_rate: int):
        self._tracker = tracker
        self._engine = TakeEngine(reference)
        self._sample_rate = sample_rate
        tracker.reset()

    def feed(self, chunk: np.ndarray) -> list[FrameScore]:
        return self._engine.push(self._tracker.push(chunk, self._sample_rate))

    def finish(self) -> TakeRecord:
        self._engine.push(self._tracker.flush())
        return self._engine.stop()

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


class LiveRunner:
    """Start/Stop for the UI. The audio callback only queues chunks; a worker thread runs tracker + engine;
    the UI calls drain() from its timer. Stop() ends the Take and saves it."""

    def __init__(self, source: AudioInput, tracker: PitchTracker, reference: Reference, home: Path):
        self._source = source
        self._tracker = tracker
        self._reference = reference
        self._home = home
        self._chunks: queue.SimpleQueue = queue.SimpleQueue()
        self._scores: queue.SimpleQueue = queue.SimpleQueue()
        self._take: LiveTake | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._take = LiveTake(self._tracker, self._reference, self._source.sample_rate)
        self._thread = threading.Thread(target=self._work, name="sing-pitch-tracker", daemon=True)
        self._thread.start()
        self._source.start(self._chunks.put)

    def _work(self) -> None:
        while (chunk := self._chunks.get()) is not None:
            self._scores.put(self._take.feed(chunk))

    def drain(self) -> list[FrameScore]:
        out: list[FrameScore] = []
        while True:
            try:
                out += self._scores.get_nowait()
            except queue.Empty:
                return out

    def stop(self) -> TakeRecord:
        self._source.stop()
        self._chunks.put(None)
        self._thread.join()
        return save_take(self._home, self._take.finish())

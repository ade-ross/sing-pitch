"""Glue without a UI: audio chunks -> pitch tracker -> Take engine. Qt-free and device-free, so it is testable."""
from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from .audio_input import AudioInput
from .pitch import PitchTracker
from .reference import Reference
from .take_engine import FrameScore, TakeEngine, TakeRecord
from .take_store import save_take


class Playback(Protocol):
    """What the runner needs from the Backing player (playback.BackingPlayer)."""

    def start(self) -> None: ...

    def position_s(self) -> float: ...

    def stop(self) -> None: ...

    def set_gains(self, backing: float, vocal: float) -> None: ...


class LiveTake:
    """One Take, fed chunk by chunk. Drive from ONE thread. latency_s: see TakeEngine."""

    def __init__(self, tracker: PitchTracker, reference: Reference, sample_rate: int, latency_s: float = 0.0):
        self._tracker = tracker
        self._engine = TakeEngine(reference, latency_s=latency_s)
        self._sample_rate = sample_rate
        tracker.reset()

    @property
    def latency_s(self) -> float:
        return self._engine.latency_s

    @latency_s.setter
    def latency_s(self, value: float) -> None:
        self._engine.latency_s = value

    def feed(self, chunk: np.ndarray) -> list[FrameScore]:
        return self._engine.push(self._tracker.push(chunk, self._sample_rate))

    def finish(self) -> TakeRecord:
        self._engine.push(self._tracker.flush())
        return self._engine.stop()


class LiveRunner:
    """Start/Stop for the UI. The audio callback only queues chunks; a worker thread runs tracker + engine;
    the UI calls drain() from its timer. Stop() ends the Take and saves it.

    Melody Takes pass the Backing player and the calibrated loop latency. The Backing starts first; at the first
    mic chunk the runner reads how far the song already is, so the Take's latency offset becomes
    latency_s - (song position when the input clock was at 0)."""

    def __init__(self, source: AudioInput, tracker: PitchTracker, reference: Reference, home: Path,
                 backing: Playback | None = None, latency_s: float = 0.0):
        self._source = source
        self._tracker = tracker
        self._reference = reference
        self._home = home
        self._backing = backing
        self._latency_s = latency_s
        self._chunks: queue.SimpleQueue = queue.SimpleQueue()
        self._scores: queue.SimpleQueue = queue.SimpleQueue()
        self._take: LiveTake | None = None
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None      # what killed the worker, if anything

    def start(self) -> None:
        """Open the Backing and the input, then start the worker: a Start that fails leaves no thread behind."""
        self._error = None
        self._take = LiveTake(self._tracker, self._reference, self._source.sample_rate, self._latency_s)
        if self._backing is None:
            on_chunk = self._chunks.put
        else:
            try:
                self._backing.start()
            except Exception:
                self._quietly(self._backing.stop)
                raise
            first = True

            def on_chunk(chunk: np.ndarray) -> None:      # PortAudio thread: one read, one assignment
                nonlocal first
                if first:
                    first = False
                    song_at_zero = self._backing.position_s() - len(chunk) / self._source.sample_rate
                    self._chunks.put(("align", self._latency_s - song_at_zero))
                self._chunks.put(chunk)

        try:
            self._source.start(on_chunk)
        except Exception:
            self._quietly(self._source.stop)
            if self._backing is not None:
                self._quietly(self._backing.stop)
            raise
        self._thread = threading.Thread(target=self._work, name="sing-pitch-tracker", daemon=True)
        self._thread.start()          # chunks that arrived meanwhile wait in the queue

    @staticmethod
    def _quietly(cleanup) -> None:
        try:
            cleanup()
        except Exception:
            pass

    def _work(self) -> None:
        while (chunk := self._chunks.get()) is not None:
            if self._error is not None:       # crashed earlier: keep emptying the queue until stop() ends us
                continue
            try:
                if isinstance(chunk, tuple):            # ("align", latency): set before the first frame is scored
                    self._take.latency_s = chunk[1]
                else:
                    self._scores.put(self._take.feed(chunk))
            except Exception as exc:
                self._error = exc

    @property
    def latency_s(self) -> float | None:
        """The Take's latency offset (after alignment to the Backing), for drawing the reference line."""
        return None if self._take is None else self._take.latency_s

    def drain(self) -> list[FrameScore]:
        out: list[FrameScore] = []
        while True:
            try:
                out += self._scores.get_nowait()
            except queue.Empty:
                break
        if not out and self._error is not None:       # the scores made before the crash are shown first
            raise self._error
        return out

    def stop(self) -> TakeRecord:
        self._source.stop()
        if self._backing is not None:
            self._backing.stop()
        self._chunks.put(None)
        self._thread.join()
        if self._error is not None:
            raise self._error
        return save_take(self._home, self._take.finish())

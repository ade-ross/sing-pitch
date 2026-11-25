"""Getting the stretched stems of a song for a Speed without freezing the window: a worker thread loads them from the
song's cache or renders them (playback/stretch.py), then a Qt signal hands the result back to the UI thread."""
from __future__ import annotations

import itertools
import threading
from typing import Callable

from PySide6 import QtCore

from ..playback.stretch import SpeedCache, Stems


class SpeedPreparer(QtCore.QObject):
    """start() returns a job number at once; `done(job, stems, error)` fires on the UI thread when it is finished
    (stems is None when it failed). A job for a Speed already in preparation waits for that one; a superseded job still
    finishes and fills the cache."""

    done = QtCore.Signal(int, object, object)
    progress = QtCore.Signal(int, float)           # (job, fraction 0..1 of its rendering), whole percents only

    def __init__(self, cache_for: Callable[[str], SpeedCache], parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._cache_for = cache_for
        self._numbers = itertools.count(1)
        self._lock = threading.Lock()
        self._inflight: dict[tuple[str, int], list[int]] = {}     # (song, percent) -> the jobs waiting for it

    def is_cached(self, song_id: str, percent: int) -> bool:
        return self._cache_for(song_id).has(percent)

    def load_now(self, song_id: str, percent: int) -> Stems:
        """Blocking: only for a Speed that is already cached (a read from disk)."""
        return self._cache_for(song_id).get(percent)

    def start(self, song_id: str, percent: int) -> int:
        job, key = next(self._numbers), (song_id, percent)
        with self._lock:
            if key in self._inflight:                   # the same Speed is already being made: wait for that one
                self._inflight[key].append(job)
                return job
            self._inflight[key] = [job]
        cache = self._cache_for(song_id)

        shown = [-1]

        def on_progress(fraction: float) -> None:
            if round(100 * fraction) == shown[0]:
                return
            shown[0] = round(100 * fraction)
            with self._lock:
                waiting = list(self._inflight.get(key, ()))
            for waiting_job in waiting:
                self.progress.emit(waiting_job, fraction)

        def work() -> None:
            try:
                stems, error = cache.get(percent, progress=on_progress), None
            except Exception as exc:
                stems, error = None, exc
            with self._lock:
                jobs = self._inflight.pop(key)
            for waiting in jobs:
                try:
                    self.done.emit(waiting, stems, error)   # queued to the UI thread: the receiver lives there
                except RuntimeError:                        # the window was closed while this was stretching
                    return

        threading.Thread(target=work, name=f"speed-{percent}", daemon=True).start()
        return job

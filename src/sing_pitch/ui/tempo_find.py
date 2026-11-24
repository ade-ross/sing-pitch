"""Finding a song's tempo without freezing the window: the first Count-in of a song imported before tempo.json existed
beat-tracks its Backing (library/tempo.py, ~20 s the first time) on a worker thread; a Qt signal hands the result back to
the UI thread."""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6 import QtCore

from ..library.tempo import TempoDetector, song_tempo


class TempoFinder(QtCore.QObject):
    """find(song_id, song_dir) returns at once; `done(song_id, tempo)` fires on the UI thread (tempo None = unknown).
    A song already being worked on is not started twice."""

    done = QtCore.Signal(str, object)

    def __init__(self, detector: TempoDetector, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._detector = detector
        self._lock = threading.Lock()
        self._busy: set[str] = set()

    def busy(self, song_id: str) -> bool:
        with self._lock:
            return song_id in self._busy

    def find(self, song_id: str, song_dir: Path) -> None:
        with self._lock:
            if song_id in self._busy:
                return
            self._busy.add(song_id)
        detector, song_dir = self._detector, Path(song_dir)

        def work() -> None:
            tempo = song_tempo(song_dir, detector)
            with self._lock:
                self._busy.discard(song_id)
            try:
                self.done.emit(song_id, tempo)      # queued to the UI thread: the receiver lives there
            except RuntimeError:                    # the window was closed meanwhile; tempo.json is written anyway
                pass

        threading.Thread(target=work, name="tempo", daemon=True).start()

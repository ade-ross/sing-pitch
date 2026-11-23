"""SwiftF0 streaming tracker (MIT, 16 kHz, 16 ms hop, ~160 ms look-ahead). The only file that imports swift_f0."""
from __future__ import annotations

import numpy as np

from .frame import PitchFrame

FMIN_HZ = 65.0     # below a low male voice
FMAX_HZ = 1100.0   # above a high soprano


class SwiftF0Tracker:
    def __init__(self) -> None:
        from swift_f0 import SwiftF0

        self._detector = SwiftF0(threads=2, spin=False)   # build once; the ONNX session is the expensive part
        self._stream = None
        self.reset()
        # the first call is ~190 ms of ONNX warm-up: pay it now, not at the first Start
        self.push(np.zeros(4800, dtype=np.float32), 48000)
        self.reset()

    def reset(self) -> None:
        """Start a fresh stream: frame times count from the first sample pushed after this call."""
        self._stream = self._detector.stream(fmin=FMIN_HZ, fmax=FMAX_HZ)

    def push(self, chunk: np.ndarray, sample_rate: int) -> list[PitchFrame]:
        return _frames(self._stream.push(chunk, sample_rate))

    def flush(self) -> list[PitchFrame]:
        return _frames(self._stream.flush())


def _frames(result) -> list[PitchFrame]:
    return [
        PitchFrame(float(t), float(hz), float(c), float(db))
        for t, hz, c, db in zip(result.timestamps, result.pitch_hz, result.confidence, result.loudness_db)
    ]

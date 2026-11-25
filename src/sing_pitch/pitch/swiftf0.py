"""SwiftF0 streaming tracker (MIT, 16 kHz, 16 ms hop). The only file that imports swift_f0.

The library's stream releases a frame once 10 later frames exist (~197 ms after its moment), the point where it equals
the whole-file result. This one releases it after LOOKAHEAD_FRAMES: measured 2026-11-25 on a real vocal and on it one
and two octaves down, 6 frames releases at ~133 ms with 0.7 % of frames more than 50 cents from the 10-frame result on
the lowest clip, while 4 frames gives 3.1 % (tests/test_tracker_lookahead.py)."""
from __future__ import annotations

import numpy as np

from .frame import PitchFrame

FMIN_HZ = 65.0     # below a low male voice
FMAX_HZ = 1100.0   # above a high soprano
LOOKAHEAD_FRAMES = 6   # later 16 ms frames a frame waits for before it is released (the library: 10)


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
        self._stream = _stream(self._detector)

    def push(self, chunk: np.ndarray, sample_rate: int) -> list[PitchFrame]:
        return _frames(self._stream.push(chunk, sample_rate))

    def flush(self) -> list[PitchFrame]:
        return _frames(self._stream.flush())


def _frames(result) -> list[PitchFrame]:
    return [
        PitchFrame(float(t), float(hz), float(c), float(db))
        for t, hz, c, db in zip(result.timestamps, result.pitch_hz, result.confidence, result.loudness_db)
    ]


def _stream(detector):
    """The library's PitchStream with its release step taking LOOKAHEAD_FRAMES (read at each call) instead of its
    fixed 10. A copy of swift_f0 0.3.0's PitchStream._emit with that one change; the 10-frame test checks the copy
    still matches the library."""
    from swift_f0 import core

    class _Stream(core.PitchStream):
        def _emit(self, final: bool):
            first = self._emitted - self._base
            available = len(self._buffer) // core.HOP
            if final and self._base == 0 and available == 0 and len(self._buffer):
                available = 1
            last = available if final else available - LOOKAHEAD_FRAMES
            if last > first:
                pitch, confidence, loudness = self._detector._run(self._buffer, self._fmin, self._fmax)
                pitch, confidence, loudness = (pitch[first:last].copy(), confidence[first:last].copy(),
                                               loudness[first:last].copy())
            else:
                pitch, confidence, loudness = np.zeros(0), np.zeros(0), np.zeros(0)
            result = core.PitchResult(core._timestamps(self._emitted, len(pitch)), pitch, confidence, loudness)
            self._emitted += len(pitch)
            if final:
                self._buffer = np.zeros(0, dtype=np.float32)
            else:
                keep = max(0, self._emitted - core.LEFT_FRAMES) - self._base
                self._buffer = self._buffer[keep * core.HOP:]
                self._base += keep
            return result

    return _Stream(detector, FMIN_HZ, FMAX_HZ)

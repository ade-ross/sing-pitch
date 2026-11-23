"""The Backing in the headphones during a Melody Take. The frame counter of its output callback is the song clock."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from ..config import SAMPLE_RATE
from .drone import _open_output


def load_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Any wav -> float32 (frames, 2) at the stream rate. Resampled once here, never in the callback."""
    import soundfile as sf
    import soxr

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    audio = audio[:, :2]
    if sr != sample_rate:
        audio = soxr.resample(audio, sr, sample_rate).astype(np.float32)
    return np.ascontiguousarray(audio)


class BackingPlayer:
    """Plays a stereo array once from the start, silence after the end. position_s() = song time rendered so far
    (read from any thread). The constant gap between rendered and heard is part of the calibrated latency."""

    def __init__(self, audio: np.ndarray, audio_rate: int = SAMPLE_RATE,
                 stream_factory: Callable[..., object] = _open_output, sample_rate: int = SAMPLE_RATE,
                 device: int | None = None):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=1)
        if audio_rate != sample_rate:
            import soxr

            audio = soxr.resample(audio, audio_rate, sample_rate).astype(np.float32)
        self._audio = audio
        self._sr = sample_rate
        self._factory = stream_factory
        self._device = device
        self._n = 0
        self._stream = None

    @classmethod
    def from_file(cls, path: Path, **kwargs) -> BackingPlayer:
        rate = kwargs.get("sample_rate", SAMPLE_RATE)
        return cls(load_audio(Path(path), rate), rate, **kwargs)

    @property
    def duration_s(self) -> float:
        return len(self._audio) / self._sr

    @property
    def finished(self) -> bool:
        return self._n >= len(self._audio)

    def position_s(self) -> float:
        return self._n / self._sr

    def start(self) -> None:
        self.stop()
        self._n = 0

        def callback(outdata, frames, time_info, status):
            chunk = self._audio[self._n : self._n + frames]
            outdata[: len(chunk)] = chunk
            outdata[len(chunk):] = 0.0
            self._n += frames

        kwargs = dict(channels=2, samplerate=self._sr, blocksize=1024, dtype="float32", callback=callback)
        if self._device is not None:
            kwargs["device"] = self._device
        self._stream = self._factory(**kwargs)
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        stream, self._stream = self._stream, None
        stream.stop()
        stream.close()

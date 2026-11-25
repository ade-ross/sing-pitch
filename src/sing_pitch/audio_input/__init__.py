"""Audio input: device choice (Yeti preferred) and a chunk stream. sounddevice is imported lazily."""
from __future__ import annotations

from typing import Callable, Protocol, Sequence

import numpy as np

from ..config import BLOCKSIZE, SAMPLE_RATE

VIRTUAL_HINTS = ("blackhole", "background music", "aggregate", "multi-output", "zoom", "teams")


class AudioInput(Protocol):
    sample_rate: int

    def start(self, on_chunk: Callable[[np.ndarray], None]) -> None: ...

    def stop(self) -> None: ...


def pick_input(devices: Sequence[dict]) -> int | None:
    """Index of the input to use: the Blue Yeti if present, else the built-in mic, else the first
    non-virtual input. Pure function over sounddevice's device dicts, so it is tested without hardware."""
    inputs = [d for d in devices if d["max_input_channels"] > 0]
    for d in inputs:
        if "yeti" in d["name"].lower():
            return d["index"]
    for d in inputs:
        name = d["name"].lower()
        if "macbook" in name and "microphone" in name:
            return d["index"]
    for d in inputs:
        if not any(h in d["name"].lower() for h in VIRTUAL_HINTS):
            return d["index"]
    return None


def list_inputs() -> list[dict]:
    import sounddevice as sd

    return [dict(d) for d in sd.query_devices() if d["max_input_channels"] > 0]


class SoundDeviceInput:
    """Real microphone. The PortAudio callback only copies the chunk out; all work happens elsewhere."""

    def __init__(self, device_index: int, sample_rate: int = SAMPLE_RATE, blocksize: int = BLOCKSIZE):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self._blocksize = blocksize
        self._stream = None

    def start(self, on_chunk: Callable[[np.ndarray], None]) -> None:
        import sounddevice as sd

        def callback(indata, frames, time_info, status):
            on_chunk(indata[:, 0].copy())  # buffer is reused by PortAudio

        self._stream = sd.InputStream(
            device=self.device_index, channels=1, samplerate=self.sample_rate, blocksize=self._blocksize,
            dtype="float32", latency="low", callback=callback,
        )
        self._stream.start()

    @property
    def latency_s(self) -> float:
        """The input latency PortAudio reports (mic -> callback), 0 before start()."""
        return 0.0 if self._stream is None else float(self._stream.latency)

    def stop(self) -> None:
        if self._stream is not None:
            stream, self._stream = self._stream, None
            try:
                stream.stop()
            finally:
                stream.close()


class ArrayInput:
    """Test double: plays a numpy array as if it came from a mic, synchronously, on start()."""

    latency_s = 0.0

    def __init__(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE, chunk: int = BLOCKSIZE):
        self._samples = np.asarray(samples, dtype=np.float32)
        self.sample_rate = sample_rate
        self._chunk = chunk

    def start(self, on_chunk: Callable[[np.ndarray], None]) -> None:
        for i in range(0, len(self._samples), self._chunk):
            on_chunk(self._samples[i : i + self._chunk])

    def stop(self) -> None:
        pass

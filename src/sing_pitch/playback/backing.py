"""The Backing in the headphones during a Melody Take. The frame counter of its output callback is the song clock."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from ..config import SAMPLE_RATE
from .drone import _open_output
from .mix import Gains
from .transport import Loop, SongTimeLog, Transport, TransportEvent


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


def _stereo_at(audio: np.ndarray, rate: int, sample_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    if rate != sample_rate:
        import soxr

        audio = soxr.resample(audio, rate, sample_rate).astype(np.float32)
    return audio


class BackingPlayer:
    """The sounddevice shell around the Transport (transport.py): each output callback renders one Transport block.
    Plays the song once from the start, silence after the end. position_s() = song time rendered so far and
    song_log = the Transport's song-time log (both read from any thread). The constant gap between rendered and
    heard is part of the calibrated latency.

    With a `vocal` stem (the Guide vocal) the output is backing * g_b + vocal * g_v. set_gains(), play(), pause() and
    set_speed() may be called from any thread at any time and act from the next block (gains ramp within it).
    Defaults: Backing 1.0, Guide vocal 0.0 (off)."""

    def __init__(self, audio: np.ndarray, audio_rate: int = SAMPLE_RATE,
                 stream_factory: Callable[..., object] = _open_output, sample_rate: int = SAMPLE_RATE,
                 device: int | None = None, vocal: np.ndarray | None = None, vocal_rate: int | None = None):
        self._audio = _stereo_at(audio, audio_rate, sample_rate)
        self._vocal = None if vocal is None else _stereo_at(vocal, vocal_rate or audio_rate, sample_rate)
        self._gains: Gains = (1.0, 0.0)
        self._stretched: tuple[float, np.ndarray, np.ndarray | None] | None = None   # (speed, Backing, vocal) at Speed
        self._loop: Loop | None = None
        self._sr = sample_rate
        self._factory = stream_factory
        self._device = device
        self._transport = self._new_transport()
        self._stream = None
        self.on_event: Callable[[TransportEvent], None] | None = None   # runs in the audio callback: only hand over

    def _new_transport(self) -> Transport:
        transport = Transport(self._audio, self._vocal, self._sr, gains=self._gains)
        if self._stretched is not None:
            transport.set_speed(*self._stretched)
        if self._loop is not None:
            transport.set_loop(self._loop)
        return transport

    @classmethod
    def from_file(cls, path: Path, vocals: Path | None = None, **kwargs) -> BackingPlayer:
        rate = kwargs.get("sample_rate", SAMPLE_RATE)
        vocal = None if vocals is None else load_audio(Path(vocals), rate)
        return cls(load_audio(Path(path), rate), rate, vocal=vocal, vocal_rate=rate, **kwargs)

    def set_gains(self, backing: float, vocal: float) -> None:
        """Linear amplitude gains. Heard from the next block on, ramped within it (no click)."""
        self._gains = (float(backing), float(vocal))
        self._transport.set_gains(*self._gains)

    def set_speed(self, speed: float, backing: np.ndarray | None = None, vocal: np.ndarray | None = None) -> None:
        """Speed (1.0 = original) with the Backing and the vocal stem already stretched to it (playback/stretch.py;
        not needed at 1.0). Applies from the next block, crossfaded, and stays for the next start(). Any thread."""
        if speed == 1.0:
            self._stretched = None
            self._transport.set_speed(1.0, self._audio, self._vocal)
            return
        self._stretched = (float(speed), _stereo_at(backing, self._sr, self._sr),
                           None if vocal is None else _stereo_at(vocal, self._sr, self._sr))
        self._transport.set_speed(*self._stretched)

    @property
    def speed(self) -> float:
        return 1.0 if self._stretched is None else self._stretched[0]

    def set_loop(self, loop: Loop | None) -> None:
        """Loop a Section (transport.Loop) from the next block, starting at its Lead-in; None stops looping. Stays for
        the next start(). Pass events go to on_event. Any thread."""
        self._loop = loop
        self._transport.set_loop(loop)

    @property
    def loop(self) -> Loop | None:
        """The loop asked for (shown from the next block)."""
        return self._loop

    def play(self) -> None:
        self._transport.play()

    def pause(self) -> None:
        self._transport.pause()

    def seek(self, song_s: float) -> None:
        """Go on from this song time, from the next block (a "jumped" event goes to on_event). Ends a loop."""
        self._loop = None
        self._transport.seek(song_s)

    @property
    def transport(self) -> Transport:
        return self._transport

    @property
    def song_log(self) -> SongTimeLog:
        return self._transport.log

    @property
    def duration_s(self) -> float:
        return self._transport.duration_s

    @property
    def finished(self) -> bool:
        return self._transport.finished

    def position_s(self) -> float:
        return self._transport.position_s

    def start(self) -> None:
        """From the song start, with a fresh song-time log; the gains as they are now (no ramp up from silence)."""
        self.stop()
        transport = self._transport = self._new_transport()

        def callback(outdata, frames, time_info, status):
            block = transport.render(frames)
            outdata[:] = block.audio
            if block.events and self.on_event is not None:
                for event in block.events:
                    self.on_event(event)

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

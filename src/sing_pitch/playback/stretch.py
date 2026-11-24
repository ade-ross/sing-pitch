"""Speed: pitch-preserving time-stretch with Rubber Band (R3 "finer" engine, offline mode), and the per-song cache
of stretched stems. The only file that imports pylibrb.

Rubber Band is GPLv2 (commercial licence otherwise): fine for a personal app, flag it if sing-pitch is distributed."""
from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

BLOCK = 16384          # Rubber Band's process limit is 524288 frames; retrieving after every block keeps buffers small


def time_stretch(audio: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
    """(frames, channels) float32 -> (frames / speed, channels) to the frame. speed 0.5 = half as fast, pitch unchanged.
    Speed 1.0 returns the input itself. The GIL is released while Rubber Band works, so this can run in a thread."""
    if speed == 1.0:
        return audio
    import pylibrb as rb

    x = np.ascontiguousarray(audio.T, dtype=np.float32)               # Rubber Band wants (channels, samples)
    n = x.shape[1]
    stretcher = rb.RubberBandStretcher(sample_rate, x.shape[0], rb.Option.PROCESS_OFFLINE | rb.Option.ENGINE_FINER,
                                       initial_time_ratio=1.0 / speed)
    stretcher.set_expected_input_duration(n)                          # makes the output length exact
    chunks = []
    for start in range(0, n, BLOCK):
        stretcher.process(np.ascontiguousarray(x[:, start : start + BLOCK]), final=start + BLOCK >= n)
        if (ready := stretcher.available()) > 0:
            chunks.append(stretcher.retrieve(ready))
    while (ready := stretcher.available()) > 0:                       # drain what the final block released
        chunks.append(stretcher.retrieve(ready))
    return np.ascontiguousarray(np.concatenate(chunks, axis=1).T)


Stretcher = Callable[[np.ndarray, int, float], np.ndarray]
Stems = tuple[np.ndarray, "np.ndarray | None"]      # (Backing, Guide vocal or None), each (frames, 2) at the stream rate


class SpeedCache:
    """Stretched stems of ONE song, kept in <song folder>/speed/<percent>/{backing,vocals}.wav (32-bit float wav, at
    the stream rate). `get` reads a cached Speed from disk and otherwise renders it (Backing and vocal in parallel
    threads), writes it, then returns it. The folder is written complete before it gets its name, so a Speed that
    shows up in `has` is whole. Deleting or replacing the song takes its cache with it."""

    def __init__(self, song_dir: Path, sample_rate: int, stretcher: Stretcher = time_stretch):
        self._dir = Path(song_dir)
        self._sr = sample_rate
        self._stretcher = stretcher

    def folder(self, percent: int) -> Path:
        return self._dir / "speed" / str(percent)

    def has(self, percent: int) -> bool:
        return percent == 100 or (self.folder(percent) / "backing.wav").exists()

    def get(self, percent: int) -> Stems:
        from .backing import load_audio

        vocals = self._dir / "vocals.wav"
        if percent == 100:
            return load_audio(self._dir / "backing.wav", self._sr), (load_audio(vocals, self._sr) if vocals.exists() else None)
        folder = self.folder(percent)
        if not self.has(percent):
            self._render(percent, folder)
        return load_audio(folder / "backing.wav", self._sr), (load_audio(folder / "vocals.wav", self._sr)
                                                             if (folder / "vocals.wav").exists() else None)

    def _render(self, percent: int, folder: Path) -> None:
        import soundfile as sf

        from .backing import load_audio

        sources = {"backing.wav": self._dir / "backing.wav"}
        if (self._dir / "vocals.wav").exists():
            sources["vocals.wav"] = self._dir / "vocals.wav"
        results: dict[str, np.ndarray] = {}
        errors: list[BaseException] = []

        def work(name: str, path: Path) -> None:
            try:
                results[name] = self._stretcher(load_audio(path, self._sr), self._sr, percent / 100)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=work, args=item, name=f"stretch-{item[0]}") for item in sources.items()]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        tmp = folder.parent / f".tmp-{uuid.uuid4().hex[:8]}"
        tmp.mkdir(parents=True)
        try:
            for name, audio in results.items():
                sf.write(str(tmp / name), audio, self._sr, subtype="FLOAT")
            if folder.exists():
                shutil.rmtree(folder)
            tmp.rename(folder)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

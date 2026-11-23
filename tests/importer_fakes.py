"""Test doubles for the three heavy parts of an import: separation, melody extraction, download."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from audio import harmonic

from sing_pitch.library import ReferenceLine, SourceInfo

SR = 44100


def write_wav(path: Path, audio: np.ndarray, sr: int = SR) -> Path:
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return path


def song_file(path: Path, seconds: float = 2.0) -> Path:
    return write_wav(path, harmonic(220.0, seconds, sr=SR))


class FakeSeparator:
    """Splits the input into 'vocals' (the input) and 'backing' (half of it), as real files."""
    model_name = "fake-roformer"

    def __init__(self):
        self.calls: list[Path] = []

    def separate(self, audio: Path, out_dir: Path) -> None:
        self.calls.append(Path(audio))
        x, sr = sf.read(str(audio), dtype="float32")
        write_wav(out_dir / "vocals.wav", x, sr)
        write_wav(out_dir / "backing.wav", x * 0.5, sr)


class FakeExtractor:
    """Reads the vocals file and returns a 10 ms line: 220 Hz, confident while the file has sound."""
    model_name = "fake-rmvpe"

    def __init__(self):
        self.calls: list[Path] = []

    def extract(self, vocals: Path) -> ReferenceLine:
        self.calls.append(Path(vocals))
        x, sr = sf.read(str(vocals), dtype="float32")
        n = int(len(x) / sr / 0.01)
        t = np.arange(n) * 0.01
        return ReferenceLine(t=t, hz=np.full(n, 220.0), confidence=np.full(n, 0.9))


class FakeDownloader:
    def __init__(self, video_id="abc12345678", title="Some Song / Live", duration_s=120.0, is_playlist=False):
        self.info = SourceInfo(id=video_id, title=title, duration_s=duration_s, is_playlist=is_playlist)
        self.probes: list[str] = []
        self.downloads: list[str] = []

    def probe(self, url: str) -> SourceInfo:
        self.probes.append(url)
        return self.info

    def download(self, url: str, dest_dir: Path) -> Path:
        self.downloads.append(url)
        return song_file(Path(dest_dir) / f"{self.info.id}.wav")

"""A folder of stems exported by Logic's Stem Splitter, used instead of automatic separation."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

AUDIO_SUFFIXES = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg"}
READY_BACKING = ("backing", "instrumental", "karaoke", "no vocal")
BACKING_VOICE_WORDS = ("back", "bgv", "harmon", "choir", "double")


class StemsError(Exception):
    pass


def find_stems(folder: Path) -> tuple[Path, list[Path]]:
    """Return (lead vocal file, the other stem files)."""
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in AUDIO_SUFFIXES and not p.name.startswith("."))
    voices = [p for p in files if any(w in p.stem.lower() for w in ("vocal", "vox", "voice"))]
    leads = [p for p in voices if not any(w in p.stem.lower() for w in BACKING_VOICE_WORDS)]
    if not voices:
        raise StemsError(f"no vocal stem found in {folder} (looking for a file name containing 'vocal')")
    if len(leads) != 1:
        names = ", ".join(p.name for p in (leads or voices))
        raise StemsError(f"cannot tell which file is the lead vocal in {folder}: {names}")
    lead = leads[0]
    others = [p for p in files if p != lead]
    if not others:
        raise StemsError(f"only the vocal stem is in {folder}: nothing to make a Backing from")
    return lead, others


def stems_key(files: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(files):
        h.update(p.name.encode())
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return "stems:" + h.hexdigest()


def _read(path: Path, sr: int, channels: int) -> np.ndarray:
    x, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    if file_sr != sr:
        x = soxr.resample(x, file_sr, sr)
    if x.shape[1] == 1 and channels == 2:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > channels:
        x = x[:, :channels]
    return x


def write_stems(lead: Path, others: list[Path], vocals_out: Path, backing_out: Path) -> None:
    """Write vocals.wav (the lead as it is) and backing.wav (a ready-made backing file, else every other stem summed)."""
    info = sf.info(str(lead))
    sr, channels = info.samplerate, min(info.channels, 2)
    vocals = _read(lead, sr, channels)
    ready = [p for p in others if any(w in p.stem.lower() for w in READY_BACKING) and "vocal" not in p.stem.lower()]
    parts = [_read(p, sr, channels) for p in (ready[:1] or others)]
    n = max(len(p) for p in parts)
    mix = np.zeros((n, channels), dtype=np.float32)
    for p in parts:
        mix[: len(p)] += p
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.99:                      # summing can clip; scale the whole mix, never clip it
        mix *= 0.99 / peak
    sf.write(str(vocals_out), vocals, sr, subtype="PCM_24")
    sf.write(str(backing_out), mix, sr, subtype="PCM_24")

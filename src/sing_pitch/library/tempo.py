"""The song's tempo, for the Count-in: beats per minute plus the beat times, detected once from the Backing by a beat
tracker and kept in the song folder as tempo.json. Songs imported before this are done on first use.

The beat tracker is librosa's (librosa.beat.beat_track). librosa comes with the `import` extra (audio-separator needs
it), so it is imported only when a tempo is detected; without it the tempo is unknown and Simone sets the BPM himself.
Prog songs fool every detector, so his own BPM (player.json) always wins over the detected one."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .library import BACKING

TEMPO = "tempo.json"
DETECT_SR = 22050        # the beat tracker's own rate: tempo needs no more


@dataclass(frozen=True)
class Tempo:
    bpm: float
    beats_s: tuple[float, ...] = ()      # beat times in song seconds (original tempo)


class TempoUnavailable(Exception):
    """No tempo can be found: no beat tracker is installed (the `import` extra brings one), or it found no beat."""


TempoDetector = Callable[[Path], Tempo]


def detect_tempo(backing: Path) -> Tempo:
    """Beat-track the Backing (mono, 22.05 kHz). Raises TempoUnavailable without librosa or when it finds no beat
    (0 BPM)."""
    try:
        import librosa
    except ImportError as exc:
        raise TempoUnavailable("no beat tracker installed (uv sync --extra import)") from exc
    y, sr = librosa.load(str(backing), sr=DETECT_SR, mono=True)
    bpm, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    bpm = float(bpm[0] if hasattr(bpm, "__len__") else bpm)
    if not math.isfinite(bpm) or bpm <= 0:
        raise TempoUnavailable(f"no beat found ({bpm} BPM)")
    return Tempo(round(bpm, 2), tuple(round(float(b), 4) for b in beats))


def load_tempo(song_dir: Path) -> Tempo | None:
    """The stored tempo; None when there is none or it cannot be read."""
    try:
        raw = json.loads((Path(song_dir) / TEMPO).read_text())
        bpm = float(raw["bpm"])
        beats = tuple(float(b) for b in raw.get("beats_s", ()))
    except (OSError, ValueError, TypeError, KeyError):
        return None
    if not math.isfinite(bpm) or bpm <= 0:
        return None
    return Tempo(bpm, beats)


def save_tempo(song_dir: Path, tempo: Tempo) -> None:
    path = Path(song_dir) / TEMPO
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"bpm": tempo.bpm, "beats_s": list(tempo.beats_s)}) + "\n")
    os.replace(tmp, path)


def song_tempo(song_dir: Path, detector: TempoDetector = detect_tempo) -> Tempo | None:
    """The song's tempo: stored, else detected from its Backing now and stored. None = unknown (no beat tracker, or
    it failed or found no beat): the Count-in then needs the BPM typed in. A tempo that cannot be stored is still
    returned."""
    if (tempo := load_tempo(song_dir)) is not None:
        return tempo
    try:
        tempo = detector(Path(song_dir) / BACKING)
    except Exception:
        return None
    if not math.isfinite(tempo.bpm) or tempo.bpm <= 0:
        return None
    try:
        save_tempo(song_dir, tempo)
    except OSError:         # not stored: it is found again next time, but known now
        pass
    return tempo

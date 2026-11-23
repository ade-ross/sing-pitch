"""Builders for Melody tests: a reference line from notes, a voice singing those notes late, a song in a library."""
from pathlib import Path

import numpy as np
import soundfile as sf
from frames import HOP, hz_at

from sing_pitch.library import Library
from sing_pitch.library.library import BACKING, REFERENCE, VOCALS, save_reference
from sing_pitch.library.ports import ReferenceLine
from sing_pitch.pitch import PitchFrame

REF_HOP = 0.010    # RMVPE frames


def reference_line(notes, gap=0.0, confidence=0.9):
    """notes = [(cents from A4, seconds), ...] laid end to end, like an RMVPE line of a sung melody."""
    t, hz, conf, start = [], [], [], 0.0
    for cents, seconds in notes:
        n = round(seconds / REF_HOP)
        t += [start + i * REF_HOP for i in range(n)]
        hz += [hz_at(cents)] * n
        conf += [confidence] * n
        start += seconds + gap
    return ReferenceLine(np.array(t, np.float32), np.array(hz, np.float32), np.array(conf, np.float32))


def voice_following(notes, delay_s: float, cents_off: float = 0.0, confidence: float = 0.98):
    """Tracker frames of a voice singing exactly the notes of reference_line(notes), heard delay_s late on the
    Take clock. Built from the notes, not from the line, so it is an independent source of truth."""
    out, end = [], sum(s for _, s in notes)
    for i in range(round((end + delay_s) / HOP)):
        t = i * HOP
        song_t, start, hz = t - delay_s, 0.0, None
        for cents, seconds in notes:
            if start <= song_t < start + seconds:
                hz = hz_at(cents + cents_off)
            start += seconds
        out.append(PitchFrame(t, 0.0, 0.0, -140.0) if hz is None else PitchFrame(t, hz, confidence, -20.0))
    return out


def library_song(home: Path, line: ReferenceLine, title: str = "Do Re Mi", seconds: float = 1.0, sr: int = 44100):
    """A complete song in <home>/songs without running the importer: silent stems + the given line."""
    lib = Library(home)
    staging = lib.new_staging()
    quiet = np.zeros((round(seconds * sr), 2), np.float32)
    sf.write(str(staging / VOCALS), quiet, sr)
    sf.write(str(staging / BACKING), quiet, sr)
    save_reference(staging / REFERENCE, line)
    return lib.commit(staging, dict(source_kind="file", source_key=f"sha256:{title}", imported_at="2026-11-23T20:00:00",
                                    stems_from="separated", separator_model="fake", extractor_model="fake"), title)

"""A synthetic song whose vocals.wav really sings its reference line: the perfect-singer oracle's input."""
from pathlib import Path

import numpy as np
import soundfile as sf
from audio import SR, harmonic, quiet
from frames import hz_at

from sing_pitch.library import Library
from sing_pitch.library.library import BACKING, REFERENCE, VOCALS, save_reference
from sing_pitch.library.ports import ReferenceLine
from sing_pitch.settings import Settings
from sing_pitch.testmode import FAKE_OUTPUT

REF_HOP = 0.010
LEAD_S = 0.5           # silence before the first note
# far-apart notes, so a voice that is late by a few hundred ms lands on the wrong one
NOTES = [(-900, 0.7), (-400, 0.7), (-800, 0.7), (-200, 0.7), (-700, 0.7)]


def singing_song(home: Path, notes=NOTES, title: str = "Oracle Song", tail_s: float = 0.5):
    """A song in <home>/songs: vocals.wav = harmonic tones of the notes, the reference line = the same notes written
    down (an independent truth: from the note list, not extracted from the wav), a silent backing."""
    vocals = np.concatenate([quiet(LEAD_S)] + [harmonic(hz_at(c), s) for c, s in notes] + [quiet(tail_s)])
    t, hz, start = [], [], LEAD_S
    for cents, seconds in notes:
        n = round(seconds / REF_HOP)
        t += [start + i * REF_HOP for i in range(n)]
        hz += [hz_at(cents)] * n
        start += seconds
    line = ReferenceLine(np.array(t, np.float32), np.array(hz, np.float32), np.full(len(t), 0.9, np.float32))
    lib = Library(home)
    staging = lib.new_staging()
    sf.write(str(staging / VOCALS), np.stack([vocals, vocals], axis=1), SR)
    sf.write(str(staging / BACKING), np.zeros((len(vocals), 2), np.float32), SR)
    save_reference(staging / REFERENCE, line)
    return lib.commit(staging, dict(source_kind="file", source_key=f"sha256:{title}", imported_at="2026-11-24T10:00:00",
                                    stems_from="separated", separator_model="fake", extractor_model="fake"), title)


def configure_latency(home: Path, latency_s: float) -> None:
    s = Settings.load(home)
    s.set_latency(FAKE_OUTPUT, latency_s)
    s.save()

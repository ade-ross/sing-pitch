"""Pure pitch arithmetic: Hz <-> MIDI (float), note names. A4 = 440 Hz, equal temperament."""
from __future__ import annotations

import math

from .config import A4_HZ

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


A4_MIDI = 69


def midi_to_hz(midi: float) -> float:
    return A4_HZ * 2 ** ((midi - A4_MIDI) / 12)


def hz_to_midi(hz: float) -> float:
    return A4_MIDI + 12.0 * math.log2(hz / A4_HZ)


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"

"""Helpers that build PitchFrames the way the pitch tracker would emit them."""
from sing_pitch.pitch import PitchFrame

HOP = 0.016


def hz_at(cents_from_a4: float) -> float:
    return 440.0 * 2 ** (cents_from_a4 / 1200)


def sung(hz: float, seconds: float, start: float = 0.0, confidence: float = 0.98, loudness_db: float = -20.0):
    """Frames of a steadily held pitch."""
    n = round(seconds / HOP)
    return [PitchFrame(t=start + i * HOP, hz=hz, confidence=confidence, loudness_db=loudness_db) for i in range(n)]


def silence(seconds: float, start: float = 0.0):
    """Frames as SwiftF0 emits them for silence: confidence 0, junk Hz."""
    n = round(seconds / HOP)
    return [PitchFrame(t=start + i * HOP, hz=0.0, confidence=0.0, loudness_db=-140.0) for i in range(n)]

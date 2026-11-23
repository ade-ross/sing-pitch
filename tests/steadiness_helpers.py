import math

from frames import HOP, hz_at

from sing_pitch.pitch import PitchFrame


def contour(cents_at, seconds, start=0.0):
    """Frames whose pitch (cents from A4) is cents_at(seconds since the note began)."""
    n = round(seconds / HOP)
    return [PitchFrame(t=start + i * HOP, hz=hz_at(cents_at(i * HOP)), confidence=0.98, loudness_db=-20.0)
            for i in range(n)]


def vibrato(depth_cents, rate_hz, seconds, start=0.0, centre=0.0):
    return contour(lambda s: centre + depth_cents * math.sin(2 * math.pi * rate_hz * s), seconds, start)

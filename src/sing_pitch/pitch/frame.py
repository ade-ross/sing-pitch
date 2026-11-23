from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PitchFrame:
    """One raw tracker frame. Not yet judged voiced or unvoiced: the gate lives in the Take engine
    so saved Takes can be re-scored with a different gate."""

    t: float             # seconds on the Take clock (0 = first sample after Start), frame centre
    hz: float            # meaningless when the frame is unvoiced
    confidence: float    # 0..1
    loudness_db: float   # dBFS RMS around the frame

"""Voicing gate: decides which raw tracker frames are voice. Hysteresis streaming, short runs dropped on re-scoring."""
from __future__ import annotations

from typing import Sequence

from ..config import ScoreConfig
from ..pitch import PitchFrame


class VoicingGate:
    """Streaming hysteresis: enter voiced at high confidence, leave only below a lower one."""

    def __init__(self, config: ScoreConfig):
        self._config = config
        self._voiced = False

    def step(self, frame: PitchFrame) -> bool:
        c = self._config
        loud_enough = frame.loudness_db >= c.gate_min_loudness_db
        threshold = c.gate_leave_confidence if self._voiced else c.gate_enter_confidence
        self._voiced = loud_enough and frame.confidence >= threshold
        return self._voiced


def voiced_mask(frames: Sequence[PitchFrame], config: ScoreConfig) -> list[bool]:
    """Full gate over a whole Take: hysteresis, then voiced runs shorter than the minimum are dropped."""
    gate = VoicingGate(config)
    mask = [gate.step(f) for f in frames]
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        if j - i < config.min_voiced_run_frames:
            mask[i:j] = [False] * (j - i)
        i = j
    return mask

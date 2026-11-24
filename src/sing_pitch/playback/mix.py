"""Mixing of the Backing and the Guide vocal for one output block. Pure numpy: the player and its successor share it."""
from __future__ import annotations

import numpy as np

Gains = tuple[float, float]     # (backing, vocal), linear amplitude


def mix_block(backing: np.ndarray, vocal: np.ndarray | None, start: Gains, end: Gains,
              frames: int | None = None) -> np.ndarray:
    """output = backing * g_b + vocal * g_v, (frames, 2) float32. Each gain moves linearly from its `start` value
    (the last block's) to its `end` value (the target) over the block, ending exactly on the target, so a gain change
    never steps. Chunks shorter than `frames` (song end) are padded with silence; vocal may be None."""
    frames = len(backing) if frames is None else frames
    ramp = (np.arange(1, frames + 1, dtype=np.float32) / frames)[:, None] if frames else np.zeros((0, 1), np.float32)
    out = np.zeros((frames, 2), np.float32)
    for source, g0, g1 in ((backing, start[0], end[0]), (vocal, start[1], end[1])):
        if source is None or not len(source):
            continue
        n = min(len(source), frames)
        out[:n] += source[:n] * (g0 + (g1 - g0) * ramp[:n])
    return out

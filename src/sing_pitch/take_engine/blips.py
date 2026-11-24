"""Tracker octave blips: a few frames the pitch tracker puts a whole octave (or more) away from the voice around them.

A blip is a short run of scored frames at least `blip_cents` away from the median of the scored frames about a second
either side of it, all on the same side. A note really sung an octave away lasts longer than `blip_max_s`, so it is
not one. Steadiness and Range leave blips out; % in tune and the median cents off do not (they are judged against
the target, where a blip is simply a wrong frame)."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import ScoreConfig

_GAP_S = 0.1          # two scored frames further apart than this are not one run


def blip_flags(scores: Sequence, config: ScoreConfig) -> list[bool]:
    """True for each of `scores` (frames that were scored) that sits in a blip."""
    n = len(scores)
    flags = np.zeros(n, dtype=bool)
    if n < 3:
        return flags.tolist()
    t = np.array([s.t for s in scores])
    cents = 100.0 * np.array([s.midi for s in scores])
    lo = np.searchsorted(t, t - config.blip_window_s, side="left")
    hi = np.searchsorted(t, t + config.blip_window_s, side="right")
    far = np.zeros(n, dtype=np.int8)
    for i in range(n):
        around = np.median(cents[lo[i]:hi[i]])
        gap = cents[i] - around
        if abs(gap) >= config.blip_cents:
            far[i] = 1 if gap > 0 else -1
    hop = float(np.median(np.diff(t))) if n > 1 else 0.0
    i = 0
    while i < n:
        if far[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and far[j + 1] == far[i] and t[j + 1] - t[j] <= _GAP_S:
            j += 1
        if t[j] - t[i] + hop <= config.blip_max_s:
            flags[i:j + 1] = True
        i = j + 1
    return flags.tolist()


def without_blips(scores: Sequence, config: ScoreConfig) -> list:
    """`scores` with the blip frames among the scored ones removed. Unvoiced and unscored frames are kept."""
    scored = [s for s in scores if s.midi is not None and s.cents is not None]
    out = {id(s) for s, blip in zip(scored, blip_flags(scored, config)) if blip}
    return [s for s in scores if id(s) not in out]

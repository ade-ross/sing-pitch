"""Where the plot's vertical range sits (#34). Think of a camera operator who only pans when the action is about to
leave the frame: the voice (its median over the last seconds, never one sample) and the Reference coming in the next
seconds at the Octave shift being drawn stay in view, and the view moves, once, to centre on them when either would
cross the edge. Between moves it stands still, so the picture does not jerk mid-phrase."""
from __future__ import annotations

from typing import Callable

import numpy as np

MIN_VOICED = 10          # voiced frames needed before the voice may decide anything
EDGE_MARGIN = 1.0        # semitones: content closer than this to an edge counts as leaving the view
VOICE_TRIGGER = 4.0      # voice alone (no Reference): its median must be this far from the centre to move the view
WIDE_TRIGGER = 3.0       # a Reference wider than the view cannot be fitted: move only when its middle is this far off
VOICE_NEAR = 5.0         # a voice further than this from the Reference (an octave off) is not worth showing with it
OUTLIER_PCT = 0.25       # percent of points the Reference range ignores at each end (a wild frame or two)


class YWindow:
    """The chosen centre (whole semitones) of a view `half` semitones either side. follow() is called every redraw
    with the recent voiced MIDI values and the coming Reference's MIDI values (both may be empty or hold NaN); it
    returns True when the centre moved. `further` (called only when a move is due) gives the Reference of the seconds
    after those, so a move centres on where the melody is going and not only on the edge it just reached."""

    def __init__(self, centre: float, half: int):
        self.centre = int(round(centre))
        self.half = half

    def set(self, centre: float) -> None:
        self.centre = int(round(centre))

    def follow(self, voice: np.ndarray, ahead: np.ndarray,
               further: Callable[[], np.ndarray] | None = None) -> bool:
        voice = np.asarray(voice, dtype=float)
        ahead = np.asarray(ahead, dtype=float)
        voice, ahead = voice[np.isfinite(voice)], ahead[np.isfinite(ahead)]
        voice_med = float(np.median(voice)) if len(voice) >= MIN_VOICED else None
        if not len(ahead):
            return self._move_to(voice_med) if voice_med is not None and \
                abs(voice_med - self.centre) > VOICE_TRIGGER else False
        lo_r, hi_r = (float(v) for v in np.percentile(ahead, (OUTLIER_PCT, 100 - OUTLIER_PCT)))
        lo, hi = self._content(lo_r, hi_r, voice_med)
        if hi - lo > self._room:                         # wider than the view: the coming melody cannot all be shown
            return self._move_to((lo + hi) / 2) if abs((lo + hi) / 2 - self.centre) > WIDE_TRIGGER else False
        if self.centre - self.half + EDGE_MARGIN <= lo and hi <= self.centre + self.half - EDGE_MARGIN:
            return False
        if further is not None:                          # a move is due: centre on this and the seconds after it
            more = np.asarray(further(), dtype=float)
            more = more[np.isfinite(more)]
            if len(more):
                lo_f, hi_f = (float(v) for v in np.percentile(np.concatenate([ahead[np.isfinite(ahead)], more]),
                                                              (OUTLIER_PCT, 100 - OUTLIER_PCT)))
                lo_f, hi_f = self._content(lo_f, hi_f, voice_med)
                if hi_f - lo_f <= self._room:
                    lo, hi = lo_f, hi_f
        return self._move_to((lo + hi) / 2)

    @property
    def _room(self) -> float:
        return 2 * (self.half - EDGE_MARGIN)             # the span that can sit inside the margins

    def _content(self, lo: float, hi: float, voice_med: float | None) -> tuple[float, float]:
        """The range to show: the Reference's, with the voice's median added when both still fit together."""
        if voice_med is not None and lo - VOICE_NEAR <= voice_med <= hi + VOICE_NEAR \
                and max(hi, voice_med) - min(lo, voice_med) <= self._room:
            return min(lo, voice_med), max(hi, voice_med)
        return lo, hi

    def _move_to(self, centre: float) -> bool:
        new = int(round(centre))
        moved = new != self.centre
        self.centre = new
        return moved

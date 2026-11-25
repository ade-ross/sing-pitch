"""Deliberately broken Transports, to ask whether the pause hunt (pause_fuzz.py) has teeth. Each one leaves the paused
flag the window reads alone and breaks only what is HEARD or where the song resumes, the faults a weak check misses:

- tail: after Space the song's sound goes on for 0.8 s (a fade-out that is far too long);
- blip: while paused, one block of tone leaks every 2 s;
- skip_resume: resuming jumps the song 1.5 s ahead.

apply(name) patches Transport.render and returns the function that undoes it."""
from __future__ import annotations

import numpy as np

from sing_pitch.playback import transport as T

SR = 48000
NAMES = ("tail", "blip", "skip_resume")


def _tone(frames: int, start: int) -> np.ndarray:
    t = (start + np.arange(frames)) / SR
    x = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return np.stack([x, x], axis=1)


def apply(name: str):
    original = T.Transport.render

    if name == "tail":
        def render(self, frames):
            block = original(self, frames)
            if not self._playing:
                began = self.__dict__.setdefault("_pause_began", self.log.rendered_s - frames / SR)
                if self.log.rendered_s - began < 0.8:
                    return T.Block(_tone(frames, round(self.log.rendered_s * SR)), block.song_t, block.events)
            else:
                self.__dict__.pop("_pause_began", None)
            return block
    elif name == "blip":
        def render(self, frames):
            block = original(self, frames)
            if not self._playing and int(self.log.rendered_s * 1000) % 2000 < 6:
                return T.Block(_tone(frames, round(self.log.rendered_s * SR)), block.song_t, block.events)
            return block
    elif name == "skip_resume":
        def render(self, frames):
            if self._want_playing and not self._playing:
                self._n += int(1.5 * self._sr)
            return original(self, frames)
    else:
        raise ValueError(name)

    T.Transport.render = render

    def undo() -> None:
        T.Transport.render = original

    return undo

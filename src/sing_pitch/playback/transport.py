"""The Transport: song playback for Melody Takes as pure logic over arrays. The sounddevice stream (BackingPlayer)
is a thin shell that asks it for one block per callback.

Two clocks meet here. The OUTPUT clock counts frames handed to the sound card (output_s = frames / rate); it runs
on as long as the stream runs, paused or not. SONG time is the position in the song in original-tempo seconds. Each
block reports the song time of every frame it rendered, and the SongTimeLog keeps the whole mapping so the Take
engine can ask "which song time was playing at output time x?" after any pause (later: seek, loop, Speed).

Control calls (play, pause, set_gains) may come from any thread: each is one attribute assignment, read by
render() at the start of the next block."""
from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

from ..config import SAMPLE_RATE
from .mix import Gains, mix_block


@dataclass(frozen=True)
class TransportEvent:
    """Something that changed the song-time mapping, at the first output frame it applies to.
    Kinds now: "paused", "resumed". Later tickets add "jumped", "pass_started", "pass_ended", "speed_changed"."""

    kind: str
    output_s: float
    song_t: float


@dataclass(frozen=True)
class Block:
    audio: np.ndarray            # (frames, 2) float32
    song_t: np.ndarray           # (frames,) song time of each frame in seconds; NaN = no song (paused)
    events: tuple[TransportEvent, ...] = ()


class SongTimeLog:
    """Output time -> song time, for everything rendered so far. Append-only stretches; each is either playing (song
    time advances `rate` song seconds per output second) or paused (no song time). Written by the audio thread,
    read by the tracker thread: a stretch's fields are appended before its start, so a reader that sees a start
    sees the whole stretch."""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self._song0: list[float | None] = []    # song frame at the stretch start; None = paused
        self._rate: list[float] = []
        self._start: list[int] = []             # output frame where the stretch starts
        self._frames = 0

    @property
    def rendered_s(self) -> float:
        return self._frames / self.sample_rate

    def append(self, frames: int, song0: float | None, rate: float = 1.0) -> None:
        """Log `frames` output frames, starting at song frame `song0` (None = paused). Continues the last stretch
        when it simply goes on, so a long linear run stays one stretch."""
        if self._start:
            last0, last_rate, last_start = self._song0[-1], self._rate[-1], self._start[-1]
            if song0 is None and last0 is None:
                self._frames += frames
                return
            if song0 is not None and last0 is not None and last_rate == rate \
                    and abs(last0 + (self._frames - last_start) * last_rate - song0) < 1e-6:
                self._frames += frames
                return
        self._song0.append(song0)
        self._rate.append(rate)
        self._start.append(self._frames)
        self._frames += frames

    def song_time(self, output_s: float) -> float | None:
        """Song time (s) rendered at this output time; None while paused. Before the first stretch it extends the
        first one back, after the last rendered frame the last one forward (a linear song keeps its old meaning)."""
        n = len(self._start)
        if n == 0:
            return None
        frame = output_s * self.sample_rate
        i = max(0, bisect.bisect_right(self._start, frame, 0, n) - 1)
        song0 = self._song0[i]
        if song0 is None:
            return None
        return (song0 + (frame - self._start[i]) * self._rate[i]) / self.sample_rate


class Transport:
    """Backing (+ optional Guide vocal) as stereo float32 arrays already at `sample_rate`. Starts playing at the
    song start. Gains as in mix_block: the target set by set_gains is reached by the end of the next block."""

    def __init__(self, backing: np.ndarray, vocal: np.ndarray | None = None, sample_rate: int = SAMPLE_RATE,
                 gains: Gains = (1.0, 0.0)):
        self._audio = backing
        if vocal is not None:
            vocal = np.ascontiguousarray(np.pad(vocal, ((0, max(0, len(backing) - len(vocal))), (0, 0)))[: len(backing)])
        self._vocal = vocal
        self._sr = sample_rate
        self._gains: Gains = gains          # target (any thread)
        self._applied: Gains = gains        # what the last block ended on (render thread only)
        self._want_playing = True           # requested (any thread)
        self._playing = True                # in effect (render thread only)
        self._n = 0                         # song frame of the next frame to render
        self.log = SongTimeLog(sample_rate)

    # --- control, any thread -------------------------------------------------------------------------------
    def play(self) -> None:
        self._want_playing = True

    def pause(self) -> None:
        self._want_playing = False

    def set_gains(self, backing: float, vocal: float) -> None:
        self._gains = (float(backing), float(vocal))

    # --- state -----------------------------------------------------------------------------------------------
    @property
    def playing(self) -> bool:
        return self._want_playing

    @property
    def position_s(self) -> float:
        """Song time of the next frame to render."""
        return self._n / self._sr

    @property
    def duration_s(self) -> float:
        return len(self._audio) / self._sr

    @property
    def finished(self) -> bool:
        return self._n >= len(self._audio)

    # --- the core --------------------------------------------------------------------------------------------
    def render(self, frames: int) -> Block:
        events: tuple[TransportEvent, ...] = ()
        if self._want_playing != self._playing:
            self._playing = self._want_playing
            kind = "resumed" if self._playing else "paused"
            events = (TransportEvent(kind, self.log.rendered_s, self.position_s),)
        end = self._gains
        if not self._playing:
            self.log.append(frames, None)
            self._applied = end
            return Block(np.zeros((frames, 2), np.float32), np.full(frames, np.nan), events)
        n = self._n
        vocal = None if self._vocal is None else self._vocal[n : n + frames]
        audio = mix_block(self._audio[n : n + frames], vocal, self._applied, end, frames)
        self._applied = end
        self.log.append(frames, float(n))
        self._n = n + frames
        return Block(audio, np.arange(n, n + frames) / self._sr, events)

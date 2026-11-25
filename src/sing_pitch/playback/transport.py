"""The Transport: song playback for Melody Takes as pure logic over arrays. The sounddevice stream (BackingPlayer)
is a thin shell that asks it for one block per callback.

Two clocks meet here. The OUTPUT clock counts frames handed to the sound card (output_s = frames / rate); it runs
on as long as the stream runs, paused or not. SONG time is the position in the song in original-tempo seconds. Each
block reports the song time of every frame it rendered, and the SongTimeLog keeps the whole mapping so the Take
engine can ask "which song time was playing at output time x?" after any pause, seek or Speed change or loop wrap.

Speed: the Transport plays arrays that were stretched beforehand (playback/stretch.py). Song time stays in original
seconds: at Speed s each output frame advances the song by s frames, and the log records rate = s.

Control calls (play, pause, set_gains, seek, set_loop, set_speed) may come from any thread: each is one attribute assignment, read by
render() at the start of the next block."""
from __future__ import annotations

import bisect
import dataclasses
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..config import SAMPLE_RATE
from .mix import Gains, mix_block


LEAD_IN_S = 2.0      # song played before a looped Section's start on every Pass (never scored)
MIN_LOOP_S = 0.2     # a shorter (or empty) Section is widened to this, so every Pass renders some song
CLICK_S = 0.025      # one metronome click of the Count-in
CLICK_HZ = (1600.0, 1000.0)     # (first beat of the bar, the others)
CLICK_LEVEL = (0.5, 0.3)        # the first beat is accented


@dataclass(frozen=True)
class CountIn:
    """One bar of metronome clicks before every Pass, in place of the Lead-in: the song's tempo (beats per minute of
    the original song, so the clicks scale with Speed) and the beats in a bar."""

    bpm: float
    beats_per_bar: int = 4

    @property
    def beat_s(self) -> float:
        """One beat in song seconds."""
        return 60.0 / self.bpm

    @property
    def bar_s(self) -> float:
        return self.beats_per_bar * self.beat_s


@dataclass(frozen=True)
class Loop:
    """A looped Section: song times in original seconds. name None = an unnamed mark. With a count_in, every Pass
    starts with one bar of clicks (the song silent) instead of the Lead-in of song."""

    start_s: float
    end_s: float
    name: str | None = None
    lead_in_s: float = LEAD_IN_S
    count_in: CountIn | None = None

    @property
    def pass_start_s(self) -> float:
        """Where every Pass starts: start - Lead-in, clamped at the song start; with a Count-in, one bar of song time
        before the start (it may lie before the song start: the song is silent then anyway)."""
        if self.count_in is not None:
            return self.start_s - self.count_in.bar_s
        return max(0.0, self.start_s - self.lead_in_s)


def click_bar(count_in: CountIn, speed: float, sample_rate: int) -> np.ndarray:
    """The Count-in as output audio (mono) at this Speed: one click per beat, the first accented, each beat
    beat_s / speed output seconds after the last."""
    beat = count_in.beat_s / speed * sample_rate
    out = np.zeros(math.ceil(count_in.beats_per_bar * beat) + 1, np.float32)
    t = np.arange(round(CLICK_S * sample_rate)) / sample_rate
    for k in range(count_in.beats_per_bar):
        hz, level = (CLICK_HZ[0], CLICK_LEVEL[0]) if k == 0 else (CLICK_HZ[1], CLICK_LEVEL[1])
        click = (level * np.exp(-t / 0.006) * np.cos(2 * np.pi * hz * t)).astype(np.float32)
        at = round(k * beat)
        out[at: at + len(click)] = click[: len(out) - at]
    return out


@dataclass(frozen=True)
class TransportEvent:
    """Something that changed the song-time mapping, at the first output frame it applies to.
    Kinds: "paused", "resumed", "speed_changed", "jumped" (a seek: song_t is where the song goes on), and for a loop
    "pass_started" (pass_index 1, 2, ...; song_t = start - Lead-in), "pass_ended" (the Pass that just ended) and
    "loop_ended" (looping switched off, or ended by a seek outside the Section; a seek inside it is a "pass_ended" +
    "pass_started" at the song time sought); all three carry their `loop`. "speed_changed" carries the new `speed` and, when it
    applied at a Pass boundary, the `loop` (outside a loop, loop is None). "gains_changed" carries the target `gains`
    (Backing, Guide vocal): the first block the new gains apply to."""

    kind: str
    output_s: float
    song_t: float
    pass_index: int | None = None
    loop: Loop | None = None
    speed: float | None = None
    gains: Gains | None = None


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

    def tail(self) -> tuple[float, float | None, float] | None:
        """(output time, song time, rate) at the end of what has been rendered: where the song is going on from, and
        how fast (song seconds per output second). Song time None = paused there; None overall = nothing logged yet."""
        frames = self._frames                   # read first: a stretch appended after this starts at or after it
        n = len(self._start)
        if n == 0:
            return None
        song0, rate, start = self._song0[n - 1], self._rate[n - 1], self._start[n - 1]      # one stretch, read once
        frames = max(frames, start)
        song = None if song0 is None else (song0 + (frames - start) * rate) / self.sample_rate
        return frames / self.sample_rate, song, rate

    def paused_spans(self) -> list[tuple[float, float]]:
        """(start, end) output seconds of every paused stretch, oldest first; end inf = still paused at the end."""
        n = len(self._start)                    # read first: the fields of the first n stretches are complete
        starts, song0 = self._start[:n], self._song0[:n]
        return [(starts[i] / self.sample_rate, starts[i + 1] / self.sample_rate if i + 1 < n else math.inf)
                for i in range(n) if song0[i] is None]

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

    def song_times(self, output_s: np.ndarray) -> np.ndarray:
        """song_time() over an array of output times in one pass (NaN = paused): what the plot draws every tick."""
        output_s = np.asarray(output_s, dtype=np.float64)
        n = len(self._start)
        if n == 0:
            return np.full(output_s.shape, np.nan)
        start = np.array(self._start[:n], dtype=np.float64)
        song0 = np.array([np.nan if s is None else s for s in self._song0[:n]], dtype=np.float64)
        rate = np.array(self._rate[:n], dtype=np.float64)
        frame = output_s * self.sample_rate
        i = np.maximum(np.searchsorted(start, frame, side="right") - 1, 0)
        return (song0[i] + (frame - start[i]) * rate[i]) / self.sample_rate


class Transport:
    """Backing (+ optional Guide vocal) as stereo float32 arrays already at `sample_rate`. Starts playing at the
    song start. Gains as in mix_block: the target set by set_gains is reached by the end of the next block."""

    def __init__(self, backing: np.ndarray, vocal: np.ndarray | None = None, sample_rate: int = SAMPLE_RATE,
                 gains: Gains = (1.0, 0.0)):
        self._audio = backing                # the array being played: the original, or the stretched one at `speed`
        self._vocal = self._fit(vocal, backing)
        self._song_frames = len(backing)     # the song's own length, whatever the Speed
        self._speed = 1.0                    # song frames per output frame (render thread only)
        self._speed_asked = (1.0, backing, vocal)   # (speed, backing, vocal) as last asked for (control threads)
        self._speed_requests: deque[tuple[float, np.ndarray, np.ndarray | None]] = deque()   # set_speed calls
                                                                    # (any thread); the last one wins
        self._sr = sample_rate
        self._gains: Gains = gains          # target (any thread)
        self._applied: Gains = gains        # what the last block ended on (render thread only)
        self._want_playing = True           # requested (any thread)
        self._playing = True                # in effect (render thread only)
        self._n = 0                         # song frame of the next frame to render
        self._seeks: deque[tuple[float, bool]] = deque()  # (song time, the loop goes on) asked for (any thread);
                                                            # the last one wins
        self._loop_requests: deque[Loop | None] = deque()   # set_loop calls (any thread); the last one wins
        self._loop: Loop | None = None      # in effect (render thread only)
        self._loop_next: Loop | None = None  # the same Section with a new Lead-in / Count-in, from the next wrap
        self._pass = 0                      # index of the Pass playing (render thread only)
        self._block0 = 0                    # output frame of the block being rendered (render thread only)
        self._wrap_out_s = 0.0              # output time of the Pass boundary being applied (render thread only)
        self._clicks: tuple[CountIn, float, np.ndarray] | None = None   # the Count-in bar last built (render thread)
        self.log = SongTimeLog(sample_rate)

    @staticmethod
    def _fit(vocal: np.ndarray | None, backing: np.ndarray) -> np.ndarray | None:
        if vocal is None:
            return None
        return np.ascontiguousarray(np.pad(vocal, ((0, max(0, len(backing) - len(vocal))), (0, 0)))[: len(backing)])

    # --- control, any thread -------------------------------------------------------------------------------
    def play(self) -> None:
        self._want_playing = True

    def pause(self) -> None:
        self._want_playing = False

    def set_gains(self, backing: float, vocal: float) -> None:
        self._gains = (float(backing), float(vocal))

    def seek(self, song_s: float) -> bool:
        """Go on from this song time (original seconds, clamped to the song), from the start of the next block,
        playing or paused, at whatever Speed is playing. Returns whether the loop goes on: song_s lies inside the
        loop it will play (the fitted one, see set_loop; a loop asked for and not yet applied counts); the next block
        does exactly that (the next Pass starts at song_s), else the seek ends the loop. The caller shows this
        answer."""
        song_s = float(song_s)
        loop = self._loop_after_requests()
        keep = loop is not None and loop.start_s <= song_s < loop.end_s
        self._seeks.append((song_s, keep))
        return keep

    def set_loop(self, loop: Loop | None) -> None:
        """Loop this Section from the next block (None = stop looping, the song runs on). Turning it on goes straight
        to start - Lead-in (or the Count-in) as Pass 1; at the Section end the song wraps back there and the next Pass
        starts. A seek outside the Section ends the loop; one inside it starts the next Pass there (Pass 1, when the
        seek comes in the same block as the set_loop). The same Section (start, end, name) with only another Lead-in or
        Count-in does not restart: the Pass playing goes on and the next one starts with the new Lead-in or Count-in. A Section shorter than MIN_LOOP_S or past the song end is first fitted into the song
        (loop shows the fitted one), so every Pass renders some song."""
        self._loop_requests.append(None if loop is None else self._fit_loop(loop))

    def _loop_after_requests(self) -> Loop | None:
        """The loop the next block plays: the last one asked for, else the one in effect."""
        try:
            return self._loop_requests[-1]
        except IndexError:
            return self._loop

    @staticmethod
    def _same_section(a: Loop | None, b: Loop | None) -> bool:
        """Both loop the same stretch of song: what plays between Passes and the name (saving the unsaved mark only
        names it) may differ, and apply from the next Pass."""
        return a is not None and b is not None and (a.start_s, a.end_s) == (b.start_s, b.end_s)

    def _fit_loop(self, loop: Loop) -> Loop | None:
        """The loop clamped to the song and at least MIN_LOOP_S long (the whole song if it is shorter); None for a
        song with no frames."""
        song_s = self._song_frames / self._sr
        if self._song_frames <= 0:
            return None
        width = min(MIN_LOOP_S, song_s)
        end = min(max(loop.end_s, loop.start_s + width), song_s)
        start = min(max(0.0, loop.start_s), end - width)
        if (start, end) == (loop.start_s, loop.end_s):
            return loop
        return dataclasses.replace(loop, start_s=start, end_s=end)

    def set_speed(self, speed: float, backing: np.ndarray, vocal: np.ndarray | None = None) -> None:
        """Play from the next block at `speed` (1.0 = original), with the Backing and Guide vocal already stretched
        to it. The song position is kept. The first block after the change crossfades from the old Speed's audio
        and a "speed_changed" event is reported; before the first block it just sets the starting Speed. Asking again
        for the Speed last asked for, with the same arrays, changes nothing (no event)."""
        asked = self._speed_asked
        if float(speed) == asked[0] and backing is asked[1] and vocal is asked[2]:
            return
        self._speed_asked = (float(speed), backing, vocal)
        self._speed_requests.append((float(speed), backing, self._fit(vocal, backing)))

    # --- state -----------------------------------------------------------------------------------------------
    @property
    def playing(self) -> bool:
        return self._want_playing

    @property
    def paused(self) -> bool:
        """Pause was asked for (and play not since): from the next block the song is silent and stands still. The one
        answer to "is the song paused": the window, the Pause button and the plot all read it, none keeps its own."""
        return not self._want_playing

    @property
    def position_s(self) -> float:
        """Song time of the next frame to render."""
        return self._n / self._sr

    @property
    def pending_seek_s(self) -> float | None:
        """Where the last seek not yet applied (it applies at the start of the next block) will go, clamped to the
        song; None when none is waiting. A second arrow key pressed before that block starts from here, not from the
        stale position_s."""
        try:
            target = self._seeks[-1][0]
        except IndexError:
            return None
        return min(max(0.0, target), self._song_frames / self._sr)

    @property
    def loop(self) -> Loop | None:
        """The loop in effect after the last block (a set_loop shows from the next block)."""
        return self._loop

    @property
    def pass_index(self) -> int:
        """The Pass playing in the last block (1, 2, ... counting every wrap); 0 without a loop."""
        return self._pass if self._loop is not None else 0

    @property
    def speed(self) -> float:
        """Speed of the block rendered last (1.0 = original); a set_speed shows from the next block."""
        return self._speed

    @property
    def duration_s(self) -> float:
        """The song's length in original seconds."""
        return self._song_frames / self._sr

    @property
    def finished(self) -> bool:
        return self._n >= self._song_frames

    # --- the core --------------------------------------------------------------------------------------------
    def render(self, frames: int) -> Block:
        self._block0 = round(self.log.rendered_s * self._sr)    # output frame of this block's first frame
        events: tuple[TransportEvent, ...] = ()
        if self._want_playing != self._playing:
            self._playing = self._want_playing
            kind = "resumed" if self._playing else "paused"
            events = (TransportEvent(kind, self.log.rendered_s, self.position_s),)
        fade_from = None
        if self._speed_requests and self._loop is None:    # in a loop it waits for the next Pass (_wrap)
            if self.log.rendered_s > 0 and self._playing:
                fade_from = (self._audio, self._vocal, self._speed)
            events += self._apply_speed()
        if self._seeks:   # after a Speed swap, so a crossfade in this block fades both Speeds at the new position
            target, keep = None, False
            while self._seeks:
                target, keep = self._seeks.popleft()
            asked = self._loop_after_requests() if self._loop_requests else None
            if keep and asked is not None and asked.start_s <= target < asked.end_s \
                    and not self._same_section(asked, self._loop):
                self._loop_requests.clear()      # a loop asked for, and a seek inside it: its Pass 1 starts there
                events += self._end_loop(0)
                self._loop, self._pass = asked, 0
                events += self._jump_in_loop(target)
            elif keep and self._loop is not None and self._loop.start_s <= target < self._loop.end_s:
                events += self._jump_in_loop(target)     # inside the looping Section: the next Pass starts there
                                                         # (a loop asked for after the seek applies below)
            else:
                self._loop_requests.clear()      # a seek ends the loop (and wins over a loop asked in the same block)
                events += self._end_loop(0)
                self._n = min(max(0, round(target * self._sr)), self._song_frames)
                events += (TransportEvent("jumped", self.log.rendered_s, self.position_s),)
        if self._loop_requests:
            loop = None
            while self._loop_requests:
                loop = self._loop_requests.popleft()
            if self._same_section(loop, self._loop):
                self._loop_next = loop           # only Between Passes changed: from the next wrap, the Passes go on
            else:
                events += self._end_loop(0)
                if loop is not None:
                    self._loop, self._pass = loop, 0
                    events += self._wrap(0)
        end = self._gains
        if end != self._applied:
            events += (TransportEvent("gains_changed", self.log.rendered_s, self.position_s, gains=end),)
        if not self._playing:
            self.log.append(frames, None)
            self._applied = end
            return Block(np.zeros((frames, 2), np.float32), np.full(frames, np.nan), events)
        start = self._applied
        parts, times, pos = [], [], 0
        while pos < frames:                  # one segment per stretch of song; a loop wrap starts a new one
            speed = self._speed              # a wrap may change it
            count = frames - pos
            if self._loop is not None:
                left = math.ceil((min(self._loop.end_s * self._sr, self._song_frames) - self._n) / speed - 1e-9)
                if left <= 0:
                    events += self._wrap(pos)
                    continue
                count = min(count, left)
            n = self._n
            g0, g1 = (tuple(a + (b - a) * x / frames for a, b in zip(start, end)) for x in (pos, pos + count))
            loop = self._loop
            counting = loop is not None and loop.count_in is not None and n < loop.start_s * self._sr - 1e-6
            if counting:
                count = min(count, math.ceil((loop.start_s * self._sr - n) / speed - 1e-9))
                audio = self._count_in(loop, n, speed, count)
            else:
                audio = self._mix(self._audio, self._vocal, n, speed, count, g0, g1)
            if fade_from is not None and pos == 0:   # a Speed change: crossfade (only up to a wrap in this block)
                # the song is silent in a Count-in at either Speed: fade the clicks in from silence
                old = np.zeros_like(audio) if counting else self._mix(*fade_from[:2], n, fade_from[2], count, g0, g1)
                weight = (np.arange(1, count + 1, dtype=np.float32) / frames)[:, None]
                audio = old * (1.0 - weight) + audio * weight
            parts.append(audio)
            times.append((n + np.arange(count) * speed) / self._sr)
            self.log.append(count, float(n), speed)
            self._n = n + count * speed
            pos += count
        self._applied = end
        return Block(np.concatenate(parts), np.concatenate(times), events)

    def _apply_speed(self) -> tuple[TransportEvent, ...]:
        """Swap in the Speed asked for by set_speed (before the first block it is only the starting Speed)."""
        request = None
        while self._speed_requests:       # popleft is atomic: a request appended meanwhile is kept for the next block
            request = self._speed_requests.popleft()
        speed, audio, vocal = request
        self._speed, self._audio, self._vocal = speed, audio, vocal
        if self.log.rendered_s <= 0:
            return ()
        return (TransportEvent("speed_changed", self.log.rendered_s if self._loop is None else self._wrap_out_s,
                               self.position_s, loop=self._loop, speed=speed),)

    def _end_loop(self, pos: int) -> tuple[TransportEvent, ...]:
        if self._loop is None:
            return ()
        out_s, song_s = (self._block0 + pos) / self._sr, self.position_s
        index, loop, self._loop, self._pass, self._loop_next = self._pass, self._loop, None, 0, None
        return (TransportEvent("pass_ended", out_s, song_s, index, loop),
                TransportEvent("loop_ended", out_s, song_s, loop=loop))

    def _jump_in_loop(self, song_s: float) -> tuple[TransportEvent, ...]:
        """A seek inside the looping Section: the Pass playing ends here and the next one starts at song_s (no Lead-in
        or Count-in: he chose where to go on), so each Take stays one unbroken stretch and the loop goes on."""
        out_s, loop = self.log.rendered_s, self._loop
        ended = (TransportEvent("pass_ended", out_s, self.position_s, self._pass, loop),) if self._pass else ()
        self._pass += 1
        self._n = round(song_s * self._sr)
        return ended + (TransportEvent("pass_started", out_s, self.position_s, self._pass, loop),)

    def _wrap(self, pos: int) -> tuple[TransportEvent, ...]:
        """A Pass boundary at frame `pos` of the block being rendered: the next Pass starts at start - Lead-in.
        A Speed change asked for during the loop is applied here, so a Pass is never half at one Speed."""
        out_s = self._wrap_out_s = (self._block0 + pos) / self._sr
        events: tuple[TransportEvent, ...] = ()
        if self._pass:
            events = (TransportEvent("pass_ended", out_s, self.position_s, self._pass, self._loop),)
        self._pass += 1
        if self._loop_next is not None:
            self._loop, self._loop_next = self._loop_next, None
        self._n = round(self._loop.pass_start_s * self._sr)
        if self._speed_requests:
            events += self._apply_speed()
        return events + (TransportEvent("pass_started", out_s, self.position_s, self._pass, self._loop),)

    def _count_in(self, loop: Loop, n: float, speed: float, frames: int) -> np.ndarray:
        """`frames` frames of the Count-in from song frame n (it runs from pass_start_s to the Section start): the
        clicks only, the song silent, whatever the gains."""
        if self._clicks is None or self._clicks[:2] != (loop.count_in, speed):
            self._clicks = (loop.count_in, speed, click_bar(loop.count_in, speed, self._sr))
        bar = self._clicks[2]
        i = round((n - loop.pass_start_s * self._sr) / speed)
        mono = np.zeros(frames, np.float32)
        part = bar[max(0, i): max(0, i + frames)]
        mono[: len(part)] = part
        return np.repeat(mono[:, None], 2, axis=1)

    @staticmethod
    def _mix(backing, vocal, n, speed, frames, start, end) -> np.ndarray:
        """`frames` frames of the array pair from song frame n (index n / speed in the stretched arrays), the gains
        ramping from start to end."""
        i = round(n / speed)
        if i < 0:       # before the song start: silence, never a slice wrapped round to the song end
            out = np.zeros((frames, 2), np.float32)
            skip = min(-i, frames)
            if skip < frames:
                out[skip:] = Transport._mix(backing, vocal, 0.0, speed, frames - skip, start, end)
            return out
        return mix_block(backing[i : i + frames], None if vocal is None else vocal[i : i + frames], start, end, frames)

"""Song clocks: which song time a voice frame is scored against. A voice frame at input time t answers what the
singer heard one calibrated latency earlier; the clock turns t into that song time (None = no song was playing)."""
from __future__ import annotations

import bisect
import threading
import time
from typing import Protocol, Sequence

import numpy as np

from ..pitch import PitchFrame


class SongLog(Protocol):
    """What LogClock reads (playback.transport.SongTimeLog)."""

    def song_time(self, output_s: float) -> float | None: ...

    def tail(self) -> tuple[float, float | None, float] | None: ...

    def paused_spans(self) -> list[tuple[float, float]]: ...


class SongClock(Protocol):
    offset_s: float      # t - song time at the Take start: drawn on the plot and kept in takes.csv (latency_offset_s)

    def __call__(self, t: float) -> float | None: ...


class LinearClock:
    """Song time = t - offset_s. Nearest note and Key (offset 0), and Melody Takes saved before the song-time log
    (their row's latency_offset_s), which were played linearly from start to end."""

    def __init__(self, offset_s: float = 0.0):
        self.offset_s = offset_s

    def __call__(self, t: float) -> float | None:
        return t - self.offset_s


class HostClock:
    """The clock PortAudio stamps every callback with (time_info) and reads in Pa_GetStreamTime: on macOS the host
    time (mach_absolute_time), the same clock as time.monotonic() (measured 2026-11-25 on CoreAudio: Pa_GetStreamTime -
    time.monotonic() = -0.4 us; a callback's currentTime - time.monotonic() read in it = -0.09 ms median). Input and
    output streams of one process share it, so a mic sample and a frame in the headphones can be compared on it."""

    @staticmethod
    def now() -> float:
        return time.monotonic()


PORTAUDIO = HostClock()


class StreamTimeline:
    """When each frame of one audio stream met the hardware, on a host clock: the DAC time an output frame left the
    device, or the ADC time a mic sample entered it, as the device reports it (PortAudio time_info).

    A stream's frame count alone is a clock that slips (#49): a block the callback rendered too late is a silence the
    count never sees (output underflow: everything after is heard later than counted), a chunk lost before its callback
    ran is a gap the count closes (input overflow: everything after was sung later than counted), and two devices'
    crystals drift apart. The timeline keeps anchors (stream seconds by frame count, host seconds); from an anchor on,
    the stream runs one second per second. add() is called once per block with that block's own timestamp: a block
    within TOLERANCE_S of where the last anchor puts it adds nothing (a steady stream stays one anchor); a lost block or
    slow drift adds one. A jump of more than GLITCH_S is a lost block or chunk: counted in `glitches`, its length in
    `lost_s`. No anchors: host time = stream time (frame counting, how a stream without timestamps keeps time).

    One audio thread adds, any thread reads: the host list is appended before the stream list, and readers use only the
    first len(stream list) entries."""

    TOLERANCE_S = 0.0005
    GLITCH_S = 0.002

    def __init__(self, clock: object = PORTAUDIO):
        self.clock = clock          # what the host times are measured on (has now()); timelines on different clocks
        self._stream: list[float] = []  # cannot be compared
        self._host: list[float] = []
        self.glitches = 0
        self.lost_s = 0.0
        self.flagged = 0            # blocks the driver itself flagged (PortAudio status: underflow / overflow)

    def __len__(self) -> int:
        return len(self._stream)

    def add(self, stream_s: float, host_s: float) -> None:
        """The block starting at stream time stream_s met the hardware at host time host_s."""
        n = len(self._stream)
        if n:
            jump = host_s - (self._host[n - 1] + stream_s - self._stream[n - 1])
            if abs(jump) <= self.TOLERANCE_S:
                return
            if jump > self.GLITCH_S:
                self.glitches += 1
                self.lost_s += jump
        self._host.append(float(host_s))
        self._stream.append(float(stream_s))

    def host(self, stream_s):
        """Host time of stream time stream_s (scalar or array): from the last anchor at or before it."""
        n = len(self._stream)
        if n == 0:
            return stream_s
        if np.ndim(stream_s):
            s = np.asarray(stream_s, dtype=np.float64)
            stream, host = np.array(self._stream[:n]), np.array(self._host[:n])
            i = np.maximum(np.searchsorted(stream, s, side="right") - 1, 0)
            return host[i] + (s - stream[i])
        i = max(0, bisect.bisect_right(self._stream, stream_s, 0, n) - 1)
        return self._host[i] + (stream_s - self._stream[i])

    def stream(self, host_s):
        """Stream time at host time host_s (scalar or array). Between a lost block's end and the next anchor nothing
        met the hardware: the stream stands at the next anchor's frame (an output plays silence, an input hears
        nothing) until then."""
        n = len(self._stream)
        if n == 0:
            return host_s
        if np.ndim(host_s):
            h = np.asarray(host_s, dtype=np.float64)
            stream, host = np.array(self._stream[:n]), np.array(self._host[:n])
            i = np.maximum(np.searchsorted(host, h, side="right") - 1, 0)
            s = stream[i] + (h - host[i])
            nxt = np.append(stream[1:], np.inf)
            return np.minimum(s, nxt[i])
        i = max(0, bisect.bisect_right(self._host, host_s, 0, n) - 1)
        s = self._stream[i] + (host_s - self._host[i])
        return min(s, self._stream[i + 1]) if i + 1 < n else s


_COUNTING = StreamTimeline(clock=None)      # no anchors: host time = frame count


def device_times(output: StreamTimeline | None, input: StreamTimeline | None):
    """(output, input) timelines when they can be used together (both exist, on one clock), else None: then the song
    clock counts frames as before #49. Either may still be waiting for its first timestamp (see LogClock)."""
    if output is None or input is None or output.clock is None or output.clock is not input.clock:
        return None
    return output, input


class LogClock:
    """The song time the Transport rendered at output time t - latency_s, where the output clock read
    output_at_input_zero_s when the input clock was at 0. Correct across pauses (later: seeks, loops, Speed).

    devices = (output, input) StreamTimelines (see device_times): then voice time and output time meet on the host
    clock instead of by counting frames, so a lost output block, a dropped mic chunk or the two devices drifting apart
    no longer shift the song against the voice for the rest of the Take (#49). The calibrated latency keeps its meaning:
    at the alignment (the first mic chunk, aligned_at_s = its end in voice time) the gap between the output's next frame
    and the input's newest sample is taken in host time instead of in counted frames, so with no glitch both ways give
    the same mapping; loop_s is the latency left once that gap is taken out (host seconds from a frame leaving the
    output to the voice answering it entering the input). A timeline without a timestamp yet (the output's first block
    lost, or not played before the first mic chunk) leaves the clock counting until both have one; the gap is then
    read from the timestamps, which place the frame rendered next at the alignment where it was really heard."""

    def __init__(self, log: SongLog, latency_s: float, output_at_input_zero_s: float,
                 devices: tuple[StreamTimeline, StreamTimeline] | None = None, aligned_at_s: float = 0.0):
        self._log = log
        self.latency_s = latency_s
        self._lock = threading.RLock()                        # alignment and settling may race (audio vs tracker)
        self._alignment = (output_at_input_zero_s, aligned_at_s)
        self._devices = devices
        self._map = self._mapping(_COUNTING, _COUNTING)      # (output, input, loop_s): one attribute, swapped whole
        self.offset_s = aligned_at_s - self.to_output(aligned_at_s)    # = latency_s - output_at_input_zero_s untimed
        self._settle()

    @property
    def output_at_input_zero_s(self) -> float:
        return self._alignment[0]

    @property
    def aligned(self) -> bool:
        """The output had handed something to the device at the alignment. When not (the Backing's first block came
        after the first mic chunk) there was no render frontier to align on, and the runner calls realign() at the
        first chunk after one exists."""
        output_at_zero, at = self._alignment
        return output_at_zero + at > 0

    def realign(self, output_at_input_zero_s: float, aligned_at_s: float) -> None:
        """Align again at a later mic chunk (see aligned); any thread."""
        with self._lock:
            self._alignment = (output_at_input_zero_s, aligned_at_s)
            devices = self._devices
            timed = devices is None and self._map[0] is not _COUNTING
            out, inp = (self._map[0], self._map[1]) if timed else (_COUNTING, _COUNTING)
            self._map = self._mapping(out, inp)
            self.offset_s = aligned_at_s - self.to_output(aligned_at_s)
        self._settle()

    def _mapping(self, out: StreamTimeline, inp: StreamTimeline) -> tuple[StreamTimeline, StreamTimeline, float]:
        output_at_zero, at = self._alignment
        gap = out.host(output_at_zero + at) - inp.host(at)
        return out, inp, self.latency_s - gap

    def _settle(self) -> None:
        """Switch to the device timestamps once both timelines have one (any thread; idempotent)."""
        devices = self._devices
        if devices is None or not (len(devices[0]) and len(devices[1])):
            return
        with self._lock:
            if self._devices is None:
                return
            self._map = self._mapping(*devices)
            self._devices = None
            self.offset_s = self._alignment[1] - self.to_output(self._alignment[1])

    @property
    def timed(self) -> bool:
        """The song is read through the device timestamps (not by counting frames)."""
        if self._devices is not None:
            self._settle()
        return self._map[0] is not _COUNTING

    @property
    def loop_s(self) -> float:
        return self._map[2]

    def to_output(self, t):
        """Output time (the Transport's frame count) answered by voice time t (scalar or array). Timed, never past what
        has been rendered: a block whose callback has not run yet cannot be playing, so while one is late (the GIL held
        elsewhere) the song waits at the last frame handed over, as the silence in the headphones does, instead of
        running on until the late block's timestamp shows the gap."""
        if self._devices is not None:
            self._settle()
        out, inp, loop_s = self._map
        o = out.stream(inp.host(t) - loop_s)
        if out is _COUNTING:
            return o
        rendered = getattr(self._log, "rendered_s", None)
        if rendered is None:
            return o
        return np.minimum(o, rendered) if np.ndim(o) else float(min(o, rendered))

    def to_voice(self, output_s):
        """Voice time that answers output time output_s (a Transport event, the end of what is rendered)."""
        if output_s == np.inf:
            return np.inf
        if self._devices is not None:
            self._settle()
        out, inp, loop_s = self._map
        return inp.stream(out.host(output_s) + loop_s)

    def voice_now(self) -> float | None:
        """The voice time being sung now (its sample enters the mic now): the input stream read on the host clock now.
        None untimed (the caller counts received samples)."""
        if not self.timed:
            return None
        inp = self._map[1]
        return float(inp.stream(inp.clock.now()))

    def __call__(self, t: float) -> float | None:
        return self._log.song_time(self.to_output(t))

    def song_times(self, t: np.ndarray) -> np.ndarray:
        """__call__ over an array of voice times, NaN where paused (for drawing: one pass, not a call per point)."""
        many = getattr(self._log, "song_times", None)
        out = self.to_output(np.asarray(t, dtype=np.float64))
        if many is not None:
            return many(out)
        return np.array([np.nan if (s := self._log.song_time(o)) is None else s for o in out], dtype=np.float64)

    def tail(self) -> tuple[float, float | None, float] | None:
        """(voice time, song time, rate) where the rendered song ends, for projecting the song that is still to be
        heard: the voice time is ahead of the live voice by about the latency. Song time None = paused there. Timed,
        never before the voice now: while a block is late the song cannot go on before it arrives (see to_output)."""
        tail = self._log.tail()
        if tail is None:
            return None
        voice = self.to_voice(tail[0])
        now = self.voice_now()
        return (voice if now is None else max(voice, now), tail[1], tail[2])

    def paused_spans(self) -> list[tuple[float, float]]:
        """(start, end) voice times the song was paused (end inf = still paused): the voice there answers nothing."""
        return [(self.to_voice(start), self.to_voice(end)) for start, end in self._log.paused_spans()]


class SampledClock:
    """A saved Take's own clock: the song time of each of its frames (None = paused), read between and just beyond
    them. Draws the Reference on a replay of a Take that was jumped into or paused. Beyond the last frame the song goes
    on at the rate it had there (the median step of its last frames, so a Take at Speed 50 % goes on at 0.5, not at
    1.0), and ends at song_end_s when that is given."""

    TAIL_FRAMES = 20     # how many of the last frames set the rate beyond the end

    def __init__(self, t: Sequence[float], song_times: Sequence[float | None], song_end_s: float | None = None):
        self._t = np.asarray(t, dtype=np.float64)
        self._song = np.array([np.nan if s is None else s for s in song_times], dtype=np.float64)
        self._song_end_s = song_end_s
        valid = np.flatnonzero(np.isfinite(self._song))
        self.offset_s = float(self._t[valid[0]] - self._song[valid[0]]) if len(valid) else 0.0
        self._rate = self._tail_rate()

    def _tail_rate(self) -> float:
        t, song = self._t[-self.TAIL_FRAMES:], self._song[-self.TAIL_FRAMES:]
        with np.errstate(invalid="ignore", divide="ignore"):
            steps = np.diff(song) / np.diff(t)
        steps = steps[np.isfinite(steps)]
        rate = float(np.median(steps)) if len(steps) else 1.0
        return rate if rate > 0 else 1.0

    def __call__(self, t: float) -> float | None:
        if not len(self._t):
            return None
        if t <= self._t[0]:
            edge, at, rate = self._song[0], self._t[0], 1.0
        elif t >= self._t[-1]:
            edge, at, rate = self._song[-1], self._t[-1], self._rate
        else:
            value = float(np.interp(t, self._t, self._song))
            return None if np.isnan(value) else value
        if np.isnan(edge):
            return None
        value = float(edge + (t - at) * rate)
        return None if self._song_end_s is not None and value > self._song_end_s else value

    def paused_spans(self) -> list[tuple[float, float]]:
        """(start, end) voice times of each run of frames without a song time (paused): from its first frame to the
        next frame with one (inf when the Take ended paused)."""
        paused = np.isnan(self._song).astype(np.int8)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], paused, [0]])))
        return [(float(self._t[a]), float(self._t[b]) if b < len(self._t) else np.inf)
                for a, b in zip(edges[::2], edges[1::2])]


class PauseFold:
    """Voice time <-> view time, the plot's x axis: voice time with every pause of the song folded away. While the
    song is paused the view time stands still, and on resume it goes on from the same place, so the plot freezes and
    then continues seamlessly. spans: the clock's paused_spans() (voice times, sorted, apart; end inf = paused now).
    No spans (Nearest note, Key, a Take never paused): view time = voice time."""

    def __init__(self, spans: Sequence[tuple[float, float]] = ()):
        self._start = np.array([s for s, _ in spans], dtype=np.float64)
        self._end = np.array([e for _, e in spans], dtype=np.float64)
        length = self._end - self._start
        self._before = np.concatenate([[0.0], np.cumsum(length)[:-1]])     # folded away before each span
        self._at = self._start - self._before                              # the view time each span folds to

    def paused(self, t: float) -> bool:
        """The voice time t lies in a pause: the voice there is neither drawn nor scored."""
        return bool(np.any((self._start <= t) & (t < self._end)))

    def to_view(self, t):
        t = np.asarray(t, dtype=np.float64)
        view = t.copy()
        for start, end in zip(self._start, self._end):
            view = view - np.clip(t - start, 0.0, end - start)
        return float(view) if view.ndim == 0 else view

    def to_voice(self, x):
        """The voice time shown at view time x. At a fold the voice time just after the pause (where the song went
        on); past the fold of a pause still going on, its start (the song waits there)."""
        x = np.asarray(x, dtype=np.float64)
        voice = x.copy()
        for start, end, before, at in zip(self._start, self._end, self._before, self._at):
            after = x >= at
            voice = np.where(after, start if np.isinf(end) else x + before + (end - start), voice)
        return float(voice) if voice.ndim == 0 else voice


def project_song_time(dt: np.ndarray, song_s: float, rate: float, loop_end_s: float | None = None,
                      loop_pass_start_s: float = 0.0, song_end_s: float | None = None) -> np.ndarray:
    """The song time that will be heard `dt` output seconds after the song was at `song_s` playing at `rate` (song
    seconds per output second). Inside a loop the song runs to loop_end_s and wraps to loop_pass_start_s (the Lead-in),
    again and again; outside one it is NaN (no song) past song_end_s. Assumes what is playing now keeps playing: a
    seek, pause or Speed change that has not been rendered yet is not foreseen."""
    dt = np.asarray(dt, dtype=np.float64)
    if loop_end_s is not None and rate > 0 and song_s >= loop_end_s:
        song_s = loop_pass_start_s          # already past the Section end: the wrap is due, the song goes on from the Lead-in
    song = song_s + dt * rate
    if loop_end_s is not None and rate > 0:
        pass_s = max(loop_end_s - loop_pass_start_s, 1e-6)
        over = song - loop_end_s                        # song seconds past the Section end, as if it ran on
        wrapped = loop_pass_start_s + np.mod(over, pass_s)
        return np.where(over >= 0, wrapped, song)
    if song_end_s is not None:
        song = np.where(song > song_end_s, np.nan, song)
    return song


def linear_song_times(frames: Sequence[PitchFrame], offset_s: float = 0.0) -> list[float]:
    """Per-frame song times of a linearly played Take (t - offset_s)."""
    return [f.t - offset_s for f in frames]

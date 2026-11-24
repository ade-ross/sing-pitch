"""Glue without a UI: audio chunks -> pitch tracker -> Take engine. Qt-free and device-free, so it is testable."""
from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from .audio_input import AudioInput
from .pitch import PitchFrame, PitchTracker
from .reference import Reference
from .take_engine import FrameScore, LogClock, PassTag, SongClock, TakeEngine, TakeRecord
from .take_store import save_take

FRAME_S = 0.016              # one tracker frame
MIN_TAKE_SCORED_S = 3.0      # a Take closed by a jump or a loop wrap with less scored singing than this is discarded


def pass_tag(loop, pass_index: int) -> PassTag:
    """The tag of Pass `pass_index` of a playback.transport.Loop; an unnamed mark is named by its span."""
    name = loop.name or f"{loop.start_s:.1f}-{loop.end_s:.1f} s"
    return PassTag(name, loop.start_s, loop.end_s, pass_index)


def _too_short(record: TakeRecord) -> bool:
    return record.summary.n_scored * FRAME_S < MIN_TAKE_SCORED_S


class Playback(Protocol):
    """What the runner needs from the Backing player (playback.BackingPlayer)."""

    def start(self) -> None: ...

    def position_s(self) -> float: ...

    @property
    def song_log(self): ...         # playback.SongTimeLog: output time -> song time

    def stop(self) -> None: ...

    on_event: object             # settable: called from the audio thread with each TransportEvent

    def set_gains(self, backing: float, vocal: float) -> None: ...

    def set_speed(self, speed: float, backing=None, vocal=None) -> None: ...


class LiveTake:
    """One Take, fed chunk by chunk. Drive from ONE thread. song_clock: see TakeEngine."""

    def __init__(self, tracker: PitchTracker, reference: Reference, sample_rate: int,
                 song_clock: SongClock | None = None, reset_tracker: bool = True, pass_tag: PassTag | None = None):
        """reset_tracker False: a Take that goes on from the previous one's audio (after a jump or a loop wrap), so
        the tracker keeps its state and its frame times. pass_tag: see TakeEngine."""
        self._tracker = tracker
        self._engine = TakeEngine(reference, song_clock=song_clock, pass_tag=pass_tag)
        self._sample_rate = sample_rate
        if reset_tracker:
            tracker.reset()

    @property
    def song_clock(self) -> SongClock:
        return self._engine.song_clock

    @song_clock.setter
    def song_clock(self, value: SongClock) -> None:
        self._engine.song_clock = value

    @property
    def pass_tag(self) -> PassTag | None:
        return self._engine.pass_tag

    @pass_tag.setter
    def pass_tag(self, value: PassTag | None) -> None:
        self._engine.pass_tag = value

    def feed(self, chunk: np.ndarray) -> list[FrameScore]:
        return self._engine.push(self._tracker.push(chunk, self._sample_rate))

    def push_frames(self, frames: list[PitchFrame]) -> list[FrameScore]:
        """Frames the tracker already made (the runner splits them between Takes)."""
        return self._engine.push(frames)

    def close(self) -> TakeRecord:
        """End this Take here, leaving the tracker alone."""
        return self._engine.stop()

    def finish(self) -> TakeRecord:
        self._engine.push(self._tracker.flush())
        return self._engine.stop()


class LiveRunner:
    """Start/Stop for the UI. The audio callback only queues chunks; a worker thread runs tracker + engine;
    the UI calls drain() from its timer. Stop() ends the Take and saves it.

    Melody Takes pass the Backing player and the calibrated loop latency. The Backing starts first; at the first
    mic chunk the runner reads how much output the Backing had rendered when the input clock was at 0, and from then
    on a voice frame at input time t is scored against the song time the Backing's song-time log shows at output
    time t - latency_s (a LogClock), so pauses and seeks (later loops, Speed) keep the sync.

    A jump (the Backing's "jumped" event) closes the current Take and opens the next one: voice frames before the
    jump's input time (event output time + the clock's offset) belong to the old Take, the rest to the new one. A Take
    closed that way with less than MIN_TAKE_SCORED_S of scored singing is discarded; the others are saved at once
    and listed by drain_saved(). On stop() the last Take is saved too, unless it is that short and an earlier Take of
    the run was saved (a run always leaves at least one Take).

    A Section loop works the same way: each "pass_started" event closes the Take there and opens the next one, tagged
    with the Section and the pass index (PassTag), so every Pass is its own Take and its Lead-in is never scored.
    "loop_ended" closes the last Pass Take and opens an untagged one. A Pass that starts before the song clock exists
    (the loop was on at Start) tags the first Take."""

    def __init__(self, source: AudioInput, tracker: PitchTracker, reference: Reference, home: Path,
                 backing: Playback | None = None, latency_s: float = 0.0):
        self._source = source
        self._tracker = tracker
        self._reference = reference
        self._home = home
        self._backing = backing
        self._latency_s = latency_s
        self._chunks: queue.SimpleQueue = queue.SimpleQueue()
        self._scores: queue.SimpleQueue = queue.SimpleQueue()
        self._take: LiveTake | None = None
        self._cuts: list[tuple[float, PassTag | None]] = []   # (input time, tag of the next Take) of boundaries not
                                                              # yet reached by the frames (worker only)
        self._saved: list[TakeRecord] = []        # Takes saved so far in this run (worker, then stop())
        self._saved_news: queue.SimpleQueue = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None      # what killed the worker, if anything

    def start(self) -> None:
        """Open the Backing and the input, then start the worker: a Start that fails leaves no thread behind."""
        self._error = None
        self._cuts, self._saved = [], []
        self._take = LiveTake(self._tracker, self._reference, self._source.sample_rate)
        if self._backing is None:
            on_chunk = self._chunks.put
        else:
            self._backing.on_event = self._on_event
            try:
                self._backing.start()
            except Exception:
                self._quietly(self._backing.stop)
                raise
            first = True

            def on_chunk(chunk: np.ndarray) -> None:      # PortAudio thread: one read, one assignment
                nonlocal first
                if first:
                    first = False
                    log = self._backing.song_log
                    output_at_zero = log.rendered_s - len(chunk) / self._source.sample_rate
                    self._chunks.put(("align", LogClock(log, self._latency_s, output_at_zero)))
                self._chunks.put(chunk)

        try:
            self._source.start(on_chunk)
        except Exception:
            self._quietly(self._source.stop)
            if self._backing is not None:
                self._quietly(self._backing.stop)
            raise
        self._thread = threading.Thread(target=self._work, name="sing-pitch-tracker", daemon=True)
        self._thread.start()          # chunks that arrived meanwhile wait in the queue

    def _on_event(self, event) -> None:       # audio thread: only hand over, in order with the chunks
        if event.kind in ("jumped", "loop_ended"):
            self._chunks.put(("jump", (event.output_s, None)))
        elif event.kind == "pass_started":
            self._chunks.put(("jump", (event.output_s, pass_tag(event.loop, event.pass_index))))

    @staticmethod
    def _quietly(cleanup) -> None:
        try:
            cleanup()
        except Exception:
            pass

    def _work(self) -> None:
        while (chunk := self._chunks.get()) is not None:
            if self._error is not None:       # crashed earlier: keep emptying the queue until stop() ends us
                continue
            try:
                if isinstance(chunk, tuple):
                    self._control(*chunk)
                else:
                    self._scores.put(self._route(self._tracker.push(chunk, self._source.sample_rate)))
            except Exception as exc:
                self._error = exc

    def _control(self, kind: str, value) -> None:
        if kind == "align":             # ("align", clock): set before the first frame is scored
            self._take.song_clock = value
        elif kind == "jump":            # ("jump", (output_s, tag)): a boundary before the clock exists has no Take
            output_s, tag = value       # to close, so the first Take simply takes its tag
            if isinstance(self._take.song_clock, LogClock):
                self._cuts.append((output_s + self._take.song_clock.offset_s, tag))
            else:
                self._take.pass_tag = tag

    def _route(self, frames: list[PitchFrame]) -> list[FrameScore]:
        """Tracker frames to the current Take; at each pending jump the Take is closed and the next one opened."""
        scores: list[FrameScore] = []
        while self._cuts and frames:
            first_after = next((i for i, f in enumerate(frames) if f.t >= self._cuts[0][0]), None)
            if first_after is None:
                break
            scores += self._take.push_frames(frames[:first_after])
            frames = frames[first_after:]
            self._next_take(self._cuts.pop(0)[1])
        return scores + self._take.push_frames(frames)

    def _next_take(self, tag: PassTag | None = None) -> None:
        old, clock = self._take, self._take.song_clock
        record = old.close()
        if not _too_short(record):
            saved = save_take(self._home, record)
            self._saved.append(saved)
            self._saved_news.put(saved)
        self._take = LiveTake(self._tracker, self._reference, self._source.sample_rate, song_clock=clock,
                              reset_tracker=False, pass_tag=tag)

    def drain_saved(self) -> list[TakeRecord]:
        """Takes saved at a jump or a loop wrap since the last call (the last Take of the run comes from stop())."""
        out: list[TakeRecord] = []
        while True:
            try:
                out.append(self._saved_news.get_nowait())
            except queue.Empty:
                return out

    @property
    def song_clock(self) -> SongClock | None:
        """Voice time -> song time of the running Take (the same clock across jumps)."""
        return None if self._take is None else self._take.song_clock

    @property
    def latency_s(self) -> float | None:
        """The song clock's offset at the Take start (voice time - song time while the song plays linearly), for
        drawing the reference line on the voice clock."""
        return None if self._take is None else self._take.song_clock.offset_s

    def drain(self) -> list[FrameScore]:
        out: list[FrameScore] = []
        while True:
            try:
                out += self._scores.get_nowait()
            except queue.Empty:
                break
        if not out and self._error is not None:       # the scores made before the crash are shown first
            raise self._error
        return out

    def stop(self) -> TakeRecord:
        self._source.stop()
        if self._backing is not None:
            self._backing.stop()
        self._chunks.put(None)
        self._thread.join()
        if self._error is not None:
            raise self._error
        record = self._take.finish()
        if self._saved and _too_short(record):
            return self._saved[-1]
        return save_take(self._home, record)

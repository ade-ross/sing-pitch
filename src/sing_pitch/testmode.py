"""Test mode: a fake mic and a null output so the real app runs end to end without a microphone or speakers.

One FakeAudioHub thread is the audio hardware. Every step (BLOCKSIZE frames of app time) it first lets each open
output stream play a step (calling its callback whenever its block runs out, like PortAudio), then gives each fake
mic one chunk. Steps are paced by the wall clock times `time_scale`, so the Transport's song clock advances exactly
as it would with a device. Because one thread drives both sides, output and input stay in a fixed relation:

- FakeMic(wav) reads the wav from `offset_s`, starting at the moment the open output stream started (or the mic's
  own start when nothing plays) and heard `delay_s` later: a singer who sings the file in time with the song,
  as late as the output->input latency. Pauses and seeks after the start are not followed: it is a file.
- FakeMic(ECHO) hears the output mix itself, `delay_s` late: with the Guide vocal on and the Backing at 0 it sings
  exactly what plays, through pauses, seeks, loops and Speed; it also answers the calibration clicks.

Selected only by flags (`sing-pitch --fake-mic ...`, see parse_flags) or by the QA driver (sing_pitch.qa)."""
from __future__ import annotations

import argparse
import functools
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .config import BLOCKSIZE, SAMPLE_RATE

FAKE_OUTPUT = "Fake output"        # the output's name: settings.json keys its latency by it
FAKE_MIC = "Fake mic"
ECHO = "echo"                      # --fake-mic echo: the mic hears the output mix
ECHO_KEEP_S = 5.0                  # the longest echo delay


class FakeOutputStream:
    """A null sounddevice.OutputStream: the hub consumes its blocks at real-time pace. Same kwargs as the real one."""

    def __init__(self, hub: FakeAudioHub, callback: Callable, channels: int = 2, blocksize: int = 1024,
                 samplerate: int = SAMPLE_RATE, dtype: str = "float32", device=None, **_):
        if samplerate != hub.sample_rate:
            raise ValueError(f"fake output runs at {hub.sample_rate} Hz, not {samplerate}")
        self._hub, self._callback = hub, callback
        self.channels, self.blocksize, self.device = channels, blocksize, device
        self.started_frame: int | None = None     # hub frame of the first frame played
        self._block = np.zeros((0, channels), np.float32)
        self._at = 0                              # frames of _block already played
        self.failed: Exception | None = None

    def start(self) -> None:
        self._hub._add_output(self)

    def stop(self) -> None:
        self._hub._remove(self)

    def close(self) -> None:
        pass

    def _play(self, frames: int) -> np.ndarray:
        """The next `frames` frames as they reach the speaker (hub thread)."""
        out = np.empty((frames, self.channels), np.float32)
        done = 0
        while done < frames:
            if self._at >= len(self._block):
                block = np.zeros((self.blocksize, self.channels), np.float32)
                if self.failed is None:
                    try:
                        self._callback(block, self.blocksize, None, None)
                    except Exception as exc:    # PortAudio would abort the stream: stay silent, say why once
                        self.failed = exc
                        print(f"fake output: callback failed: {exc!r}", file=sys.stderr)
                        block[:] = 0
                self._block, self._at = block, 0
            n = min(frames - done, len(self._block) - self._at)
            out[done : done + n] = self._block[self._at : self._at + n]
            done, self._at = done + n, self._at + n
        return out


class FakeMic:
    """An AudioInput fed by the hub: chunks of BLOCKSIZE at real-time pace. `source` is a mono float32 array at
    the hub rate, or ECHO. See the module docstring for what it hears."""

    def __init__(self, hub: FakeAudioHub, source, offset_s: float = 0.0, delay_s: float = 0.0, loop: bool = False):
        self._hub = hub
        self._source = source
        self.sample_rate = hub.sample_rate
        self.offset_s, self.delay_s, self.loop = float(offset_s), float(delay_s), loop
        self._on_chunk: Callable[[np.ndarray], None] | None = None
        self._origin = 0          # hub frame where the source's offset_s is heard, before the delay

    def start(self, on_chunk: Callable[[np.ndarray], None]) -> None:
        self._on_chunk = on_chunk
        self._hub._add_input(self)

    def stop(self) -> None:
        self._hub._remove(self)

    def _chunk(self, frame: int, n: int) -> np.ndarray:
        """Samples heard at hub frames [frame, frame + n) (hub thread)."""
        delay = round(self.delay_s * self.sample_rate)
        if isinstance(self._source, str):
            return self._hub._mix_at(frame - delay, n)
        rel = (frame - self._origin - delay) + np.arange(n)      # frames since the voice started singing
        idx = round(self.offset_s * self.sample_rate) + rel
        src = self._source
        if self.loop and len(src):
            idx = idx % len(src)
        ok = (rel >= 0) & (idx >= 0) & (idx < len(src))          # silent until then, and after the end
        out = np.zeros(n, np.float32)
        out[ok] = src[idx[ok]]
        return out


class FakeAudioHub:
    """The fake audio hardware: one thread steps every fake output and mic, paced at time_scale x real time."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, step: int = BLOCKSIZE, time_scale: float = 1.0,
                 record: bool = False):
        self.sample_rate, self.step, self.time_scale = sample_rate, step, float(time_scale)
        self._cond = threading.Condition()   # guards everything below; held while a step runs
        self._frame = 0
        self._outputs: list[FakeOutputStream] = []
        self._inputs: list[FakeMic] = []
        self._held = 0
        self._closed = False
        self._echo = np.zeros((round(ECHO_KEEP_S * sample_rate) // step + 1) * step, np.float32)  # ring, mono mix
        self._recording: list[np.ndarray] | None = [] if record else None
        self._thread = threading.Thread(target=self._run, name="sing-pitch-fake-audio", daemon=True)
        self._thread.start()

    # --- what the app opens ------------------------------------------------------------------------------------
    def open_output(self, **kwargs) -> FakeOutputStream:
        """A stream_factory for BackingPlayer, DronePlayer and the calibration player."""
        return FakeOutputStream(self, **kwargs)

    # --- clock ---------------------------------------------------------------------------------------------------
    @property
    def now_s(self) -> float:
        """App time: seconds of audio the hub has stepped since it was made."""
        return self._frame / self.sample_rate

    @contextmanager
    def held(self):
        """Stop the clock while the block runs (e.g. Start + seek land before the first block is played)."""
        with self._cond:
            self._held += 1
        try:
            yield
        finally:
            with self._cond:
                self._held -= 1
                self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._thread.join()

    def recording(self) -> np.ndarray:
        """Everything played, stereo, from the hub's first frame (only with record=True)."""
        if self._recording is None:
            raise RuntimeError("the hub was made without record=True")
        with self._cond:
            return np.concatenate(self._recording) if self._recording else np.zeros((0, 2), np.float32)

    def write_recording(self, path: Path) -> None:
        import soundfile as sf

        sf.write(str(path), self.recording(), self.sample_rate)

    # --- streams --------------------------------------------------------------------------------------------------
    def _add_output(self, stream: FakeOutputStream) -> None:
        with self._cond:
            stream.started_frame = self._frame
            self._outputs.append(stream)

    def _add_input(self, mic: FakeMic) -> None:
        with self._cond:
            mic._origin = self._outputs[-1].started_frame if self._outputs else self._frame
            self._inputs.append(mic)

    def _remove(self, stream) -> None:
        with self._cond:     # waits for a running step: no callback runs after stop() returns, as with PortAudio
            for streams in (self._outputs, self._inputs):
                if stream in streams:
                    streams.remove(stream)

    def _mix_at(self, frame: int, n: int) -> np.ndarray:
        """The mono output mix at hub frames [frame, frame + n); silence before the hub started or out of the ring."""
        out = np.zeros(n, np.float32)
        lo = max(frame, 0, self._frame + self.step - len(self._echo))
        hi = min(frame + n, self._frame + self.step)
        if hi > lo:
            idx = np.arange(lo, hi) % len(self._echo)
            out[lo - frame : hi - frame] = self._echo[idx]
        return out

    # --- the thread -----------------------------------------------------------------------------------------------
    def _step(self) -> None:
        n, frame = self.step, self._frame
        mix = np.zeros((n, 2), np.float32)
        for stream in list(self._outputs):
            played = stream._play(n)
            mix[:, : played.shape[1]] += played[:, :2]
        if self._recording is not None:
            self._recording.append(mix)
        self._echo[np.arange(frame, frame + n) % len(self._echo)] = mix.mean(axis=1)
        for mic in list(self._inputs):
            chunk = mic._chunk(frame, n)
            try:
                mic._on_chunk(chunk)
            except Exception as exc:
                print(f"fake mic: on_chunk failed: {exc!r}", file=sys.stderr)
        self._frame = frame + n

    def _run(self) -> None:
        t0, f0 = time.monotonic(), 0
        while True:
            with self._cond:
                if self._held and not self._closed:
                    while self._held and not self._closed:
                        self._cond.wait()
                    t0, f0 = time.monotonic(), self._frame       # the clock restarts where it was stopped
                if self._closed:
                    return
                self._step()
                due = t0 + (self._frame - f0) / self.sample_rate / self.time_scale
            wait = due - time.monotonic()
            if wait > 0:
                time.sleep(wait)


# --- window wiring ----------------------------------------------------------------------------------------------
def load_mono(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    from .playback.backing import load_audio

    return np.ascontiguousarray(load_audio(Path(path), sample_rate).mean(axis=1), dtype=np.float32)


class FakeMicFactory:
    """MainWindow's input_factory: every Start and calibration opens a new FakeMic with these settings. The
    attributes may be changed between Takes (the QA driver moves offset_s to where a Take starts)."""

    def __init__(self, hub: FakeAudioHub, source, offset_s: float = 0.0, delay_s: float = 0.0, loop: bool = False):
        self.hub, self.source, self.offset_s, self.delay_s, self.loop = hub, source, offset_s, delay_s, loop

    def __call__(self, _device_index=None) -> FakeMic:
        return FakeMic(self.hub, self.source, self.offset_s, self.delay_s, self.loop)


class _CalibrationPlayer:
    """CalibrationRun's player_factory: the click track through a BackingPlayer on the fake output."""

    def __init__(self, hub: FakeAudioHub):
        self._hub = hub

    def __call__(self, audio, rate):
        from .playback import BackingPlayer

        return BackingPlayer(audio, rate, sample_rate=rate, stream_factory=self._hub.open_output)


def _fake_output_info() -> tuple[str, str]:
    return FAKE_OUTPUT, "headphones"


def _no_devices() -> list[dict]:
    return []


def _fake_mic_devices() -> list[dict]:
    return [dict(index=0, name=FAKE_MIC, max_input_channels=1, max_output_channels=0)]


def _always_headphones(_name: str) -> str:
    return "headphones"


def window_kwargs(hub: FakeAudioHub, mic: FakeMicFactory | None, fake_output: bool) -> dict:
    """MainWindow keyword arguments that put the fakes in place (module-level callables only: nothing here holds
    the window, so no reference cycle through it, see handoff-segfault)."""
    from .playback import BackingPlayer, DronePlayer

    kwargs: dict = {}
    if mic is not None:
        kwargs.update(input_factory=mic, input_devices=_fake_mic_devices)
    if fake_output:
        kwargs.update(backing_factory=functools.partial(BackingPlayer.from_file, stream_factory=hub.open_output),
                      drone_factory=functools.partial(DronePlayer, stream_factory=hub.open_output),
                      calibration_player_factory=_CalibrationPlayer(hub), output_info=_fake_output_info,
                      output_devices=_no_devices, output_kind_of=_always_headphones)
    return kwargs


# --- flags --------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Flags:
    fake_mic: str | None = None           # a wav path, or ECHO
    fake_mic_offset: float = 0.0
    fake_mic_loop: bool = False
    fake_latency: float | None = None     # None = the latency configured for the fake output
    fake_output: bool = False
    record_output: str | None = None

    @property
    def any(self) -> bool:
        return self.fake_mic is not None or self.fake_output


def parse_flags(argv: list[str]) -> tuple[Flags, list[str]]:
    """The test-mode flags out of argv, and the rest. Without them the app is exactly the normal one."""
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--fake-mic")
    p.add_argument("--fake-mic-offset", type=float, default=0.0)
    p.add_argument("--fake-mic-loop", action="store_true")
    p.add_argument("--fake-latency", type=float)
    p.add_argument("--fake-output", action="store_true")
    p.add_argument("--record-output")
    ns, rest = p.parse_known_args(argv)
    if ns.record_output and not ns.fake_output:
        raise SystemExit("--record-output needs --fake-output")
    return Flags(ns.fake_mic, ns.fake_mic_offset, ns.fake_mic_loop, ns.fake_latency, ns.fake_output,
                 ns.record_output), rest


def build(flags: Flags, home: Path, time_scale: float = 1.0) -> tuple[FakeAudioHub, FakeMicFactory | None, dict]:
    """(hub, mic factory, MainWindow kwargs) for these flags. The mic's delay defaults to the latency the window
    will use for the fake output, so the file mic is a perfect singer out of the box."""
    from .settings import Settings

    hub = FakeAudioHub(time_scale=time_scale, record=flags.record_output is not None)
    mic = None
    if flags.fake_mic is not None:
        delay = flags.fake_latency
        if delay is None:
            delay = Settings.load(home).latency_for(FAKE_OUTPUT if flags.fake_output else None)
        source = ECHO if flags.fake_mic == ECHO else load_mono(Path(flags.fake_mic), hub.sample_rate)
        mic = FakeMicFactory(hub, source, flags.fake_mic_offset, delay, flags.fake_mic_loop)
    return hub, mic, window_kwargs(hub, mic, flags.fake_output)

"""Latency calibration: clicks play in the headphones, Simone claps or sings "ta" on each one, and the median delay
from click to answer (on the same clocks a Melody Take uses) becomes the Take's latency offset. Pure numpy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_CLICKS = 8
INTERVAL_S = 0.75     # 80 bpm: easy to anticipate, long enough for a 350 ms Bluetooth delay
FIRST_S = 1.5         # time to get ready
CLICK_S = 0.012
SEARCH_S = (-0.08, 0.5)   # an answer counts for a click from 80 ms early (anticipation) to 500 ms late
ENV_HOP_S = 0.001


def click_track(sample_rate: int, n: int = N_CLICKS, interval_s: float = INTERVAL_S, first_s: float = FIRST_S):
    """(stereo float32 audio, click times in s). Each click is a short decaying 2 kHz burst."""
    times = first_s + interval_s * np.arange(n)
    audio = np.zeros(round((times[-1] + 1.0) * sample_rate), np.float32)
    k = np.arange(round(CLICK_S * sample_rate))
    click = (0.6 * np.sin(2 * np.pi * 2000 * k / sample_rate) * np.exp(-k / (0.003 * sample_rate))).astype(np.float32)
    for c in times:
        i = round(c * sample_rate)
        audio[i : i + len(click)] = click
    return np.stack([audio, audio], axis=1), times


@dataclass(frozen=True)
class Calibration:
    latency_s: float | None     # None = not enough answers heard
    matched: int                # clicks with an answer
    spread_s: float             # median absolute deviation of the delays: how even the answers were


def _onsets(recording: np.ndarray, sample_rate: int) -> np.ndarray:
    """Times where the level jumps from the noise floor to well above it (first crossing, 1 ms resolution)."""
    x = np.asarray(recording, dtype=np.float64)
    if x.ndim == 2:
        x = x[:, 0]
    hop = max(1, round(ENV_HOP_S * sample_rate))
    n = len(x) // hop
    if n == 0:
        return np.array([])
    env = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1))
    floor = np.median(env)
    peak = np.max(env)
    if peak < max(8 * floor, 1e-3):
        return np.array([])
    threshold = floor + 0.2 * (peak - floor)
    above = env >= threshold
    rising = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    onsets, last = [], -1e9
    for i in rising:
        t = i * hop / sample_rate
        if t - last > 0.15:          # one answer = one onset
            onsets.append(t)
        last = t if t - last > 0.15 else last
    return np.array(onsets)


def measure_latency(recording: np.ndarray, sample_rate: int, click_times) -> Calibration:
    """recording is on the input clock, click_times on the song clock: both as a Melody Take sees them."""
    onsets = _onsets(recording, sample_rate)
    delays = []
    for c in click_times:
        hits = onsets[(onsets >= c + SEARCH_S[0]) & (onsets <= c + SEARCH_S[1])]
        if len(hits):
            delays.append(hits[0] - c)
    if len(delays) < max(3, len(click_times) // 2):
        return Calibration(None, len(delays), 0.0)
    d = np.array(delays)
    med = float(np.median(d))
    return Calibration(med, len(delays), float(np.median(np.abs(d - med))))


class CalibrationRun:
    """Plays the click track through the Backing player and records the mic, aligned exactly like LiveRunner
    aligns a Melody Take (song position read at the first mic chunk). start(), wait duration_s, finish()."""

    def __init__(self, source, player_factory=None, output_device: int | None = None):
        from .playback import BackingPlayer

        self._source = source
        self._sr = source.sample_rate
        self.audio, self.click_times = click_track(self._sr)
        factory = player_factory or (lambda audio, rate: BackingPlayer(audio, rate, sample_rate=rate,
                                                                        device=output_device))
        self._player = factory(self.audio, self._sr)
        self._chunks: list[np.ndarray] = []
        self._song_at_zero = 0.0

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self._sr

    def start(self) -> None:
        self._player.start()
        first = True

        def on_chunk(chunk):
            nonlocal first
            if first:
                first = False
                self._song_at_zero = self._player.position_s() - len(chunk) / self._sr
            self._chunks.append(chunk)

        try:
            self._source.start(on_chunk)
        except Exception:
            self._player.stop()
            raise

    def finish(self) -> Calibration:
        self._source.stop()
        self._player.stop()
        recording = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
        # click times on the input clock; the delay found is the Take's loop latency
        return measure_latency(recording, self._sr, np.asarray(self.click_times) - self._song_at_zero)

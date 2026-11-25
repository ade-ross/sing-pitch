"""The audio health record: one row per run (Start to Stop) in <home>/audio_health.csv, so a real session with
Bluetooth earbuds and a real mic shows what the tests could not (#62/#63 review): how often audio was lost on either
side, how jittery the device timestamps are, how many anchors the song clock needed, how slow the redraws were and how
often the window changed size. Written at Stop; never stops a Stop when it fails."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

FILE = "audio_health.csv"
FIELDS = ["stopped_at", "take_ids", "timed", "loop_latency", "latency_s", "error",
          "output_device", "output_block_min", "output_block_max", "output_latency_s",
          "lost_out_blocks", "lost_out_s", "out_steps_back", "out_steps_back_s", "out_spikes", "out_flagged",
          "out_anchors", "out_blocks", "out_jitter_p50_us", "out_jitter_p99_us",
          "input_device", "input_block_min", "input_block_max", "input_latency_s",
          "lost_in_chunks", "lost_in_s", "in_steps_back", "in_steps_back_s", "in_spikes", "in_flagged",
          "in_anchors", "in_blocks", "in_jitter_p50_us", "in_jitter_p99_us",
          "ticks", "tick_p50_ms", "tick_p99_ms", "window_size_changes", "window_min", "window_max"]


def dropout_note(n: int) -> str:
    """The Take summary's note when audio was lost (empty when none)."""
    return "" if n <= 0 else f"   ·   {n} audio dropout{'s' if n != 1 else ''}"


class UiStats:
    """What the window saw during a run: tick durations and its sizes."""

    def __init__(self):
        self.ticks: list[float] = []
        self.size_changes = 0
        self._size: tuple[int, int] | None = None
        self.min_size: tuple[int, int] | None = None
        self.max_size: tuple[int, int] | None = None

    def tick(self, seconds: float, size: tuple[int, int]) -> None:
        if len(self.ticks) < 200_000:
            self.ticks.append(seconds)
        if self._size is not None and size != self._size:
            self.size_changes += 1
        self._size = size
        area = size[0] * size[1]
        if self.min_size is None or area < self.min_size[0] * self.min_size[1]:
            self.min_size = size
        if self.max_size is None or area > self.max_size[0] * self.max_size[1]:
            self.max_size = size


def _side(prefix: str, times) -> dict:
    if times is None:
        return {}
    h = times.health()
    lost_key = "lost_out" if prefix == "out" else "lost_in"
    unit = "blocks" if prefix == "out" else "chunks"
    return {f"{lost_key}_{unit}": h["glitches"], f"{lost_key}_s": h["lost_s"], f"{prefix}_steps_back": h["steps_back"],
            f"{prefix}_steps_back_s": h["steps_back_s"], f"{prefix}_spikes": h["spikes"],
            f"{prefix}_flagged": h["flagged"], f"{prefix}_anchors": h["anchors"], f"{prefix}_blocks": h["blocks"],
            f"{prefix}_jitter_p50_us": h["jitter_p50_us"], f"{prefix}_jitter_p99_us": h["jitter_p99_us"],
            ("output" if prefix == "out" else "input") + "_block_min": h["frames_min"],
            ("output" if prefix == "out" else "input") + "_block_max": h["frames_max"]}


def row(runner, ui: UiStats, input_name: str, output_name: str, take_ids, error: str | None = None) -> dict:
    """The record of a run that just stopped (LiveRunner after stop())."""
    clock = runner.song_clock
    ticks = np.array(ui.ticks) * 1000 if ui.ticks else None
    r = dict(stopped_at=datetime.now().isoformat(timespec="seconds"), take_ids=" ".join(dict.fromkeys(take_ids)),
             timed=bool(getattr(clock, "timed", False)), loop_latency=runner.loop_latency,
             latency_s=runner.calibrated_latency_s, error=error or "",
             output_device=output_name, input_device=input_name,
             output_latency_s=runner.output_latency_s, input_latency_s=runner.input_latency_s,
             ticks=len(ui.ticks), tick_p50_ms=None if ticks is None else round(float(np.percentile(ticks, 50)), 2),
             tick_p99_ms=None if ticks is None else round(float(np.percentile(ticks, 99)), 2),
             window_size_changes=ui.size_changes,
             window_min="" if ui.min_size is None else f"{ui.min_size[0]}x{ui.min_size[1]}",
             window_max="" if ui.max_size is None else f"{ui.max_size[0]}x{ui.max_size[1]}")
    r.update(_side("out", runner.output_times))
    r.update(_side("in", runner.input_times))
    return {k: r.get(k) for k in FIELDS}


def append(home: Path, r: dict) -> None:
    path = Path(home) / FILE
    new = not path.exists()
    Path(home).mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: "" if v is None else v for k, v in r.items()})


def read(home: Path) -> list[dict]:
    path = Path(home) / FILE
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

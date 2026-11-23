"""Constants in one place: Bands, voicing gate, audio, data directory."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 48_000          # input stream rate; the tracker resamples to 16 kHz itself
BLOCKSIZE = 256               # small blocks keep display lag down (input latency ~36 ms)
A4_HZ = 440.0


@dataclass(frozen=True)
class ScoreConfig:
    green_cents: float = 25.0       # Band edges: |cents| <= 25 green, <= 50 yellow, beyond red
    yellow_cents: float = 50.0
    # voicing gate on raw tracker output (raw values are saved, so Takes can be re-scored with a new gate)
    gate_enter_confidence: float = 0.6
    gate_leave_confidence: float = 0.4
    gate_min_loudness_db: float = -50.0
    min_voiced_run_frames: int = 3  # voiced runs shorter than this (48 ms) are breaths/clicks: not scored
    tendency_deadband_cents: float = 5.0


DEFAULT_SCORE_CONFIG = ScoreConfig()


def data_dir() -> Path:
    """Where Takes live. Outside git. SING_PITCH_HOME overrides (tests, experiments)."""
    override = os.environ.get("SING_PITCH_HOME")
    if override:
        return Path(override)
    return Path.home() / "Library" / "Application Support" / "sing-pitch"


def models_dir() -> Path:
    """Where the downloaded models live (rmvpe.onnx, audio-separator/). SING_PITCH_MODELS overrides."""
    override = os.environ.get("SING_PITCH_MODELS")
    return Path(override) if override else data_dir() / "models"


# Vocal separator for song import: best open vocal RoFormer (MelBand, vocals SDR 12.6). Simone judges by listening.
DEFAULT_SEPARATOR_MODEL = "vocals_mel_band_roformer.ckpt"

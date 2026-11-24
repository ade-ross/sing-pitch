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
    # Steadiness: only held notes, with attack and release left out (Dai & Dixon 2019)
    steadiness_min_note_s: float = 0.5
    steadiness_trim_s: float = 0.12
    steadiness_trim_fraction: float = 0.15   # the larger of trim_s and this share of the note is cut each end
    # Steadiness and Range leave out tracker octave blips: a run of scored frames no longer than blip_max_s that sits
    # blip_cents or more from the median of the scored frames blip_window_s either side (see take_engine/blips.py);
    # Range is the middle of what is left (range_percentiles)
    blip_cents: float = 1000.0
    blip_max_s: float = 0.3
    blip_window_s: float = 1.0
    range_percentiles: tuple[float, float] = (5.0, 95.0)
    # Octave shift: whole-octave offsets tried, and how many scored frames (~2 s of singing) the LIVE view
    # waits before it stops re-estimating a growing Phrase and keeps its octave for the rest of that Phrase
    octave_shifts: tuple[int, ...] = (0, -1, 1, -2, 2)     # tie-break order: the smaller shift wins
    octave_live_lock_frames: int = 125
    phrase_gap_s: float = 0.5      # a pause this long ends a Phrase; an Auto Octave shift is chosen per Phrase


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

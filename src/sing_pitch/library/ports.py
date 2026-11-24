"""The three heavy parts of an import, as interfaces. Real adapters live in adapters.py; tests use fakes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ReferenceLine:
    """The original singer's pitch line, raw: one value per frame, never gated (so thresholds can change later).

    t: seconds from the start of the song (frame centres); hz: always a number, meaningless where confidence is low;
    confidence: 0..1. float32 on disk.
    """
    t: np.ndarray
    hz: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class SourceInfo:
    id: str            # stable id of the source (YouTube video id): the "already imported?" key
    title: str
    duration_s: float | None
    is_playlist: bool = False


class StemSeparator(Protocol):
    model_name: str

    def separate(self, audio: Path, out_dir: Path) -> None:
        """Write out_dir/vocals.wav and out_dir/backing.wav (the song with the vocal removed)."""


class MelodyExtractor(Protocol):
    model_name: str

    def extract(self, vocals: Path) -> ReferenceLine: ...


class SongDownloader(Protocol):
    def probe(self, url: str) -> SourceInfo:
        """Look the link up without downloading."""

    def download(self, url: str, dest_dir: Path) -> Path:
        """Download the audio as a wav inside dest_dir and return its path."""

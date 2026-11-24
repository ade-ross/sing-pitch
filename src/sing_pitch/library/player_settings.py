"""How one song was last set up in the player, stored in its library folder as player.json: Guide vocal on/off, the
two volumes (0-1), the Speed in percent and the Section that was chosen (a saved Section's name, None = the unnamed
mark), the Octave box (None = Auto, else the pinned shift) and Notes mode with its Minimum note length (math.inf = one
note per Phrase, written "phrase"). A song reopens with these."""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

PLAYER = "player.json"
SPEED_RANGE = (50, 120)
OCTAVE_CHOICES = (-2, -1, 0, 1)      # the pinned values of the Octave box; Auto is None
MIN_NOTE_RANGE = (0.1, 4.0)          # the Minimum note length slider, in seconds; its far end is one note per Phrase
DEFAULT_MIN_NOTE_S = 0.3


@dataclass(frozen=True)
class PlayerSettings:
    guide_on: bool = False
    guide_gain: float = 1.0
    backing_gain: float = 1.0
    speed_pct: int = 100
    section: str | None = None
    octave: int | None = None
    notes: bool = False
    min_note_s: float = DEFAULT_MIN_NOTE_S


def _min_note(value) -> float:
    if value == "phrase":
        return math.inf
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return DEFAULT_MIN_NOTE_S
    return float(value) if MIN_NOTE_RANGE[0] <= value <= MIN_NOTE_RANGE[1] else DEFAULT_MIN_NOTE_S


def _gain(value, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _from_dict(raw: dict) -> PlayerSettings:
    d = PlayerSettings()
    try:
        speed = int(raw.get("speed_pct", d.speed_pct))
    except (TypeError, ValueError):
        speed = d.speed_pct
    section = raw.get("section")
    octave = raw.get("octave")
    return PlayerSettings(
        guide_on=raw.get("guide_on") is True,
        guide_gain=_gain(raw.get("guide_gain", d.guide_gain), d.guide_gain),
        backing_gain=_gain(raw.get("backing_gain", d.backing_gain), d.backing_gain),
        speed_pct=speed if SPEED_RANGE[0] <= speed <= SPEED_RANGE[1] else d.speed_pct,
        section=section if isinstance(section, str) else None,
        octave=octave if isinstance(octave, int) and not isinstance(octave, bool) and octave in OCTAVE_CHOICES
        else None,
        notes=raw.get("notes") is True,
        min_note_s=_min_note(raw.get("min_note_s", d.min_note_s)),
    )


class PlayerSettingsStore:
    def __init__(self, song_dir: Path):
        self._path = Path(song_dir) / PLAYER

    def load(self) -> PlayerSettings:
        """The saved settings; the defaults for a song with none, or a file that cannot be read."""
        try:
            raw = json.loads(self._path.read_text())
            return _from_dict(raw if isinstance(raw, dict) else {})
        except (OSError, ValueError):
            return PlayerSettings()

    def save(self, settings: PlayerSettings) -> None:
        if settings == self.load():
            return
        tmp = self._path.with_suffix(".tmp")
        raw = asdict(settings)
        if math.isinf(raw["min_note_s"]):
            raw["min_note_s"] = "phrase"         # JSON has no infinity
        tmp.write_text(json.dumps(raw, indent=2) + "\n")
        os.replace(tmp, self._path)

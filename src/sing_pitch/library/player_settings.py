"""How one song was last set up in the player, stored in its library folder as player.json: Guide vocal on/off, the
two volumes (0-1), the Speed in percent and the Section that was chosen (a saved Section's name, None = the unnamed
mark) and the Octave box (None = Auto, else the pinned shift). A song reopens with these."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

PLAYER = "player.json"
SPEED_RANGE = (50, 120)
OCTAVE_CHOICES = (-2, -1, 0, 1)      # the pinned values of the Octave box; Auto is None


@dataclass(frozen=True)
class PlayerSettings:
    guide_on: bool = False
    guide_gain: float = 1.0
    backing_gain: float = 1.0
    speed_pct: int = 100
    section: str | None = None
    octave: int | None = None


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
        tmp.write_text(json.dumps(asdict(settings), indent=2) + "\n")
        os.replace(tmp, self._path)

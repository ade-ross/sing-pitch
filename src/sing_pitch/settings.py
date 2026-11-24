"""App settings that persist between runs: <home>/settings.json. Calibrated latency per output device, and
output devices Simone confirmed are headphones."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_OUTPUT = "default output"     # the key of an output whose name cannot be read

DEFAULT_LATENCY_S = 0.08     # uncalibrated guess: ~40 ms input + ~40 ms output (02-audio-io); calibrate once


@dataclass
class Settings:
    home: Path
    latency_s: dict[str, float] = field(default_factory=dict)
    last_calibrated: str | None = None
    headphones: list[str] = field(default_factory=list)
    output_device: str | None = None        # the Output Simone chose by name; None = the system default
    last_reference: str | None = None       # the Reference and song he left the window on: it opens on them again
    last_song: str | None = None

    @classmethod
    def load(cls, home: Path) -> Settings:
        path = Path(home) / "settings.json"
        try:
            raw = json.loads(path.read_text())
            return cls(Path(home), {k: float(v) for k, v in raw.get("latency_s", {}).items()},
                       raw.get("last_calibrated"), list(raw.get("headphones", [])), raw.get("output_device"),
                       raw.get("last_reference"), raw.get("last_song"))
        except (OSError, ValueError, AttributeError, TypeError):
            return cls(Path(home))

    def save(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        data = dict(latency_s=self.latency_s, last_calibrated=self.last_calibrated, headphones=self.headphones,
                    output_device=self.output_device, last_reference=self.last_reference, last_song=self.last_song)
        (self.home / "settings.json").write_text(json.dumps(data, indent=2) + "\n")

    def is_calibrated(self, output: str | None) -> bool:
        return (output or DEFAULT_OUTPUT) in self.latency_s

    def latency_for(self, output: str | None) -> float:
        """Calibrated latency of this output; else the most recent calibration; else the default guess."""
        output = output or DEFAULT_OUTPUT
        if output in self.latency_s:
            return self.latency_s[output]
        if self.last_calibrated in self.latency_s:
            return self.latency_s[self.last_calibrated]
        return DEFAULT_LATENCY_S

    def set_latency(self, output: str | None, latency_s: float) -> None:
        output = output or DEFAULT_OUTPUT
        self.latency_s[output] = round(float(latency_s), 4)
        self.last_calibrated = output

    def confirm_headphones(self, output: str) -> None:
        if output not in self.headphones:
            self.headphones.append(output)

    def headphones_confirmed(self, output: str | None) -> bool:
        return output in self.headphones

"""Sections of one song, stored in its library folder as sections.json: the saved ones by name, plus the last
unnamed mark (kept until replaced). Times are song times in original-tempo seconds."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

SECTIONS = "sections.json"


@dataclass(frozen=True)
class Section:
    name: str | None          # None = the unnamed mark
    start_s: float
    end_s: float


def _ordered(name: str | None, a: float, b: float) -> Section:
    return Section(name, float(min(a, b)), float(max(a, b)))


class SectionStore:
    def __init__(self, song_dir: Path):
        self._path = Path(song_dir) / SECTIONS

    def _load(self) -> dict:
        if not self._path.exists():
            return {"mark": None, "saved": []}
        return json.loads(self._path.read_text())

    def _write(self, data: dict) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        os.replace(tmp, self._path)

    def saved(self) -> list[Section]:
        return [Section(s["name"], s["start_s"], s["end_s"]) for s in self._load()["saved"]]

    def get(self, name: str) -> Section:
        return next(s for s in self.saved() if s.name == name)

    def save(self, name: str, start_s: float, end_s: float) -> Section:
        """Save under `name`; a Section of that name is replaced in place."""
        name = name.strip()
        if not name:
            raise ValueError("a Section needs a name")
        section = _ordered(name, start_s, end_s)
        data = self._load()
        entry = {"name": name, "start_s": section.start_s, "end_s": section.end_s}
        names = [s["name"] for s in data["saved"]]
        if name in names:
            data["saved"][names.index(name)] = entry
        else:
            data["saved"].append(entry)
        self._write(data)
        return section

    def rename(self, old: str, new: str) -> None:
        new = new.strip()
        data = self._load()
        names = [s["name"] for s in data["saved"]]
        if not new or (new != old and new in names):
            raise ValueError(f"there is already a Section called {new!r}" if new else "a Section needs a name")
        data["saved"][names.index(old)]["name"] = new
        self._write(data)

    def delete(self, name: str) -> None:
        data = self._load()
        data["saved"] = [s for s in data["saved"] if s["name"] != name]
        self._write(data)

    @property
    def mark(self) -> Section | None:
        m = self._load()["mark"]
        return None if m is None else Section(None, m["start_s"], m["end_s"])

    def set_mark(self, start_s: float | None, end_s: float | None = None) -> None:
        """The unnamed mark (None clears it)."""
        data = self._load()
        if start_s is None:
            data["mark"] = None
        else:
            s = _ordered(None, start_s, end_s)
            data["mark"] = {"start_s": s.start_s, "end_s": s.end_s}
        self._write(data)

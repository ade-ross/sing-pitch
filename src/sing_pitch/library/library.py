"""Song library on disk: <home>/songs/<song-id>/{vocals.wav, backing.wav, reference.npz, meta.json, source.*}.

A song appears in the library only when it is complete: an import works in a hidden staging folder and renames it
into place at the end, so a failed import leaves nothing behind.
"""
from __future__ import annotations

import dataclasses
import json
import re
import shutil
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ports import ReferenceLine
from .sections import SECTIONS, SectionStore

VOCALS, BACKING, REFERENCE, META = "vocals.wav", "backing.wav", "reference.npz", "meta.json"


@dataclass(frozen=True)
class SongMeta:
    id: str
    title: str
    source_kind: str             # "file" | "youtube" | "logic-stems"
    source_key: str              # "sha256:<hex>" | "youtube:<video id>" | "stems:<sha256 of the stem files>"
    imported_at: str
    stems_from: str              # "separated" | "logic"
    separator_model: str | None
    extractor_model: str
    source_url: str | None = None
    duration_s: float | None = None
    source_file: str | None = None   # name of the kept original inside the song folder


@dataclass(frozen=True)
class Song:
    meta: SongMeta
    dir: Path

    @property
    def id(self) -> str:
        return self.meta.id

    @property
    def vocals_path(self) -> Path:
        return self.dir / VOCALS

    @property
    def backing_path(self) -> Path:
        return self.dir / BACKING

    def load_reference(self) -> ReferenceLine:
        with np.load(self.dir / REFERENCE) as f:
            return ReferenceLine(t=f["t"], hz=f["hz"], confidence=f["confidence"])


def slugify(title: str) -> str:
    ascii_ = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_.lower()).strip("-")
    return slug[:60].strip("-") or "song"


def save_reference(path: Path, line: ReferenceLine) -> None:
    np.savez(path, t=line.t.astype(np.float32), hz=line.hz.astype(np.float32),
             confidence=line.confidence.astype(np.float32))


class Library:
    def __init__(self, home: Path):
        self.root = Path(home) / "songs"

    def list_songs(self) -> list[Song]:
        if not self.root.exists():
            return []
        songs = [self._open(d) for d in sorted(self.root.iterdir()) if (d / META).exists()]
        return sorted(songs, key=lambda s: s.meta.title.lower())

    def get(self, song_id: str) -> Song:
        d = self.root / song_id
        if not (d / META).exists():
            raise KeyError(f"no song {song_id!r} in the library")
        return self._open(d)

    def sections(self, song_id: str) -> SectionStore:
        """The song's saved Sections and unnamed mark."""
        return SectionStore(self.get(song_id).dir)

    def find_by_source(self, source_key: str) -> Song | None:
        return next((s for s in self.list_songs() if s.meta.source_key == source_key), None)

    def remove(self, song_id: str) -> None:
        shutil.rmtree(self.root / song_id)

    def new_staging(self) -> Path:
        d = self.root / f".staging-{uuid.uuid4().hex[:8]}"
        d.mkdir(parents=True)
        return d

    def commit(self, staging: Path, meta_fields: dict, title: str, replacing: str | None = None) -> Song:
        """Name the song, write meta.json and move the staging folder into place. With replacing=<song id> the new
        song takes that id and the old folder is deleted only after the new one is in place."""
        song_id = replacing or self._free_id(slugify(title))
        meta = SongMeta(id=song_id, title=title, **meta_fields)
        (staging / META).write_text(json.dumps(dataclasses.asdict(meta), indent=2) + "\n")
        final = self.root / song_id
        if replacing is None:
            staging.rename(final)
            return Song(meta, final)
        if (final / SECTIONS).exists():        # a re-import keeps the Sections he marked (song times stay valid)
            shutil.copy2(final / SECTIONS, staging / SECTIONS)
        old = self.root / f".old-{uuid.uuid4().hex[:8]}"
        final.rename(old)
        try:
            staging.rename(final)
        except BaseException:
            old.rename(final)
            raise
        shutil.rmtree(old, ignore_errors=True)
        return Song(meta, final)

    def discard(self, staging: Path) -> None:
        shutil.rmtree(staging, ignore_errors=True)

    def _free_id(self, base: str) -> str:
        candidate, n = base, 2
        while (self.root / candidate).exists():
            candidate, n = f"{base}-{n}", n + 1
        return candidate

    def _open(self, d: Path) -> Song:
        return Song(SongMeta(**json.loads((d / META).read_text())), d)

"""Import a song: source in, vocal + Backing + reference line stored in the library."""
from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import logic_stems
from .library import BACKING, REFERENCE, VOCALS, Library, Song, save_reference
from .ports import MelodyExtractor, SongDownloader, StemSeparator


MAX_DURATION_S = 15 * 60.0   # a song, not a podcast


class SongImportError(Exception):
    """The source cannot be imported; the message is for Simone."""


_YOUTUBE_ID = re.compile(r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|embed/|live/)|youtu\.be/)([A-Za-z0-9_-]{11})")


def youtube_id(url: str) -> str | None:
    """The video id when the link plainly names one, so a known song needs no network at all."""
    m = _YOUTUBE_ID.search(url)
    return m.group(1) if m else None


def file_key(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


class Importer:
    def __init__(self, library: Library, separator: StemSeparator, extractor: MelodyExtractor,
                 downloader: SongDownloader | None = None, progress: Callable[[str], None] = lambda _msg: None):
        self.library, self.separator, self.extractor, self.downloader = library, separator, extractor, downloader
        self.progress = progress

    def import_file(self, path: Path, *, force: bool = False) -> Song:
        path = Path(path)
        if not path.is_file():
            raise SongImportError(f"not a file: {path}")
        key = file_key(path)
        return self._existing(key, force) or self._build(
            title=path.stem, source=path, fields=dict(source_kind="file", source_key=key))

    def import_url(self, url: str, *, force: bool = False, max_duration_s: float = MAX_DURATION_S) -> Song:
        if self.downloader is None:
            raise SongImportError("no downloader available")
        known = youtube_id(url)
        if known and not force and (song := self.library.find_by_source(f"youtube:{known}")):
            return song
        info = self.downloader.probe(url)
        if info.is_playlist:
            raise SongImportError("that link is a playlist; give the link of one song")
        if info.duration_s and info.duration_s > max_duration_s:
            raise SongImportError(f"that video is too long for a song ({info.duration_s / 60:.0f} min)")
        key = f"youtube:{info.id}"
        existing = self._existing(key, force)
        if existing:
            return existing
        with tempfile.TemporaryDirectory(prefix="sing-pitch-dl-") as tmp:
            self.progress(f"downloading {info.title} ...")
            audio = self.downloader.download(url, Path(tmp))
            return self._build(title=info.title, source=audio, fields=dict(
                source_kind="youtube", source_key=key, source_url=url, duration_s=info.duration_s))

    def import_stems(self, folder: Path, *, force: bool = False) -> Song:
        folder = Path(folder)
        try:
            lead, others = logic_stems.find_stems(folder)
        except logic_stems.StemsError as e:
            raise SongImportError(str(e)) from e
        key = logic_stems.stems_key([lead, *others])
        existing = self._existing(key, force)
        if existing:
            return existing
        staging = self.library.new_staging()
        try:
            logic_stems.write_stems(lead, others, staging / VOCALS, staging / BACKING)
            self.progress("extracting the melody line ...")
            save_reference(staging / REFERENCE, self.extractor.extract(staging / VOCALS))
            return self.library.commit(staging, dict(
                source_kind="logic-stems", source_key=key, stems_from="logic", separator_model=None,
                extractor_model=self.extractor.model_name,
                imported_at=datetime.now().isoformat(timespec="seconds")), folder.name)
        except BaseException:
            self.library.discard(staging)
            raise

    def _existing(self, key: str, force: bool) -> Song | None:
        song = self.library.find_by_source(key)
        if song and force:
            self.library.remove(song.id)
            return None
        return song

    def _build(self, *, title: str, source: Path, fields: dict) -> Song:
        staging = self.library.new_staging()
        try:
            kept = staging / f"source{source.suffix.lower()}"
            shutil.copyfile(source, kept)
            self.progress("separating vocal and backing (about 3x realtime) ...")
            self.separator.separate(kept, staging)
            self.progress("extracting the melody line ...")
            line = self.extractor.extract(staging / VOCALS)
            save_reference(staging / REFERENCE, line)
            return self.library.commit(staging, dict(
                imported_at=datetime.now().isoformat(timespec="seconds"), stems_from="separated",
                separator_model=self.separator.model_name, extractor_model=self.extractor.model_name,
                source_file=kept.name, **fields), title)
        except BaseException:
            self.library.discard(staging)
            raise

"""Command line: `sing-pitch import <audio file | YouTube link | folder of Logic stems>`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from . import config
from .library import Importer, Library, SongImportError


def build_importer(model: str, progress: Callable[[str], None]) -> Importer:
    from .library import adapters

    adapters.require_ffmpeg()
    models = config.models_dir()
    return Importer(Library(config.data_dir()), adapters.AudioSeparatorSeparator(models, model),
                    adapters.RmvpeExtractor(models), adapters.YtDlpDownloader(), progress=progress)


def import_main(argv: list[str], make_importer: Callable[[], Importer] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sing-pitch import",
                                description="Import a song: audio file, YouTube link, or folder of Logic stems.")
    p.add_argument("source")
    p.add_argument("--force", action="store_true", help="import again even if the song is already in the library")
    p.add_argument("--model", default=config.DEFAULT_SEPARATOR_MODEL, help="audio-separator vocal model file name")
    args = p.parse_args(argv)

    def say(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    try:
        importer = make_importer() if make_importer else build_importer(args.model, say)
        src = args.source
        if src.startswith(("http://", "https://")):
            song = importer.import_url(src, force=args.force)
        elif Path(src).is_dir():
            song = importer.import_stems(Path(src), force=args.force)
        else:
            song = importer.import_file(Path(src), force=args.force)
    except SongImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    line = song.load_reference()
    voiced = float(np.mean(line.confidence >= 0.3)) if line.confidence.size else 0.0
    print(f"{song.id}: {song.meta.title}  ({song.meta.stems_from} stems, {voiced:.0%} of frames voiced)")
    print(f"  listen: {song.vocals_path}\n          {song.backing_path}")
    return 0

"""Command line: `sing-pitch import <audio file | YouTube link | folder of Logic stems>` and `sing-pitch calibrate`."""
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
    from .library.tempo import detect_tempo

    adapters.require_ffmpeg()
    models = config.models_dir()
    return Importer(Library(config.data_dir()), adapters.AudioSeparatorSeparator(models, model),
                    adapters.RmvpeExtractor(models), adapters.YtDlpDownloader(), progress=progress, tempo=detect_tempo)


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


def calibrate_main(argv: list[str], make_run=None, output_name: str | None = None, home: Path | None = None,
                   wait: Callable[[float], None] | None = None) -> int:
    """Clicks in the headphones; clap or sing "ta" on each. Stores the delay for the current output device."""
    import time

    from .calibration import CalibrationRun
    from .settings import DEFAULT_OUTPUT, Settings

    argparse.ArgumentParser(prog="sing-pitch calibrate",
                            description="Measure the headphone-to-mic latency used by Melody Takes.").parse_args(argv)
    if make_run is None:
        from .audio_input import SoundDeviceInput, list_inputs, pick_input
        from .playback.outputs import default_output_name

        output_name = default_output_name()
        make_run = lambda: CalibrationRun(SoundDeviceInput(pick_input(list_inputs())))   # noqa: E731
    output = output_name or DEFAULT_OUTPUT
    print(f"Output: {output}. In 1.5 s, 8 clicks: clap or sing 'ta' exactly on each one.", flush=True)
    run = make_run()
    run.start()
    (wait or time.sleep)(run.duration_s + 0.3)
    result = run.finish()
    if result.latency_s is None:
        print(f"error: heard {result.matched} of 8 answers. Try again, louder, closer to the mic.", file=sys.stderr)
        return 1
    settings = Settings.load(home or config.data_dir())
    settings.set_latency(output, result.latency_s)
    settings.save()
    print(f"{output}: {1000 * result.latency_s:.0f} ms ({result.matched}/8 answers, spread "
          f"{1000 * result.spread_s:.0f} ms). Saved as the default for Melody Takes.")
    return 0

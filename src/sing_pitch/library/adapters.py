"""Real separator, melody extractor and downloader. Heavy imports happen inside the methods, so the app starts
without the `import` extra installed."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np

from .. import config
from .importer import SongImportError
from .ports import ReferenceLine, SourceInfo


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise SongImportError("ffmpeg is not installed (needed for song import): run `brew install ffmpeg`")


class AudioSeparatorSeparator:
    """Vocal / Backing split with a RoFormer model through the audio-separator package."""

    def __init__(self, models_dir: Path | None = None, model: str = config.DEFAULT_SEPARATOR_MODEL):
        self.models_dir = Path(models_dir) if models_dir else config.models_dir()
        self.model_name = model

    def separate(self, audio: Path, out_dir: Path) -> None:
        try:
            from audio_separator.separator import Separator
        except ImportError as e:
            raise SongImportError("the import extra is missing: run `uv sync --extra import`") from e
        model_dir = self.models_dir / "audio-separator"
        model_dir.mkdir(parents=True, exist_ok=True)
        sep = Separator(model_file_dir=str(model_dir), output_dir=str(out_dir), output_format="WAV",
                        log_level=logging.WARNING)
        sep.load_model(model_filename=self.model_name)
        # models call the non-vocal stem "other" or "instrumental"; we always call it backing
        sep.separate(str(audio), custom_output_names={"Vocals": "vocals", "Other": "backing",
                                                      "Instrumental": "backing"})
        for name in ("vocals.wav", "backing.wav"):
            if not (Path(out_dir) / name).exists():
                raise SongImportError(f"the separator did not produce {name}")


class RmvpeExtractor:
    """Pitch line of a vocal stem with RMVPE (ONNX, CPU): 10 ms frames, Hz and confidence per frame."""
    model_name = "rmvpe-onnx-0.2.3"

    def __init__(self, models_dir: Path | None = None):
        self.model_path = (Path(models_dir) if models_dir else config.models_dir()) / "rmvpe.onnx"

    def extract(self, vocals: Path) -> ReferenceLine:
        try:
            import soundfile as sf
            from rmvpe_onnx import RMVPE
        except ImportError as e:
            raise SongImportError("the import extra is missing: run `uv sync --extra import`") from e
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        model = RMVPE(model_path=str(self.model_path), device="cpu")   # downloads once if the file is absent
        audio, sr = sf.read(str(vocals), dtype="float32")
        t, hz, confidence, _ = model.predict(audio, sr)
        return ReferenceLine(t=np.asarray(t, np.float32), hz=np.asarray(hz, np.float32),
                             confidence=np.asarray(confidence, np.float32))


class YtDlpDownloader:
    _OPTS = {"noplaylist": True, "quiet": True, "no_warnings": True, "noprogress": True}

    def _ydl(self, **extra):
        try:
            import yt_dlp
        except ImportError as e:
            raise SongImportError("the import extra is missing: run `uv sync --extra import`") from e
        return yt_dlp.YoutubeDL({**self._OPTS, **extra})

    def probe(self, url: str) -> SourceInfo:
        import yt_dlp
        try:
            with self._ydl() as ydl:
                info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as e:
            raise SongImportError(f"could not read that link: {e}") from e
        return SourceInfo(id=info["id"], title=info.get("title") or info["id"], duration_s=info.get("duration"),
                          is_playlist=info.get("_type") == "playlist")

    def download(self, url: str, dest_dir: Path) -> Path:
        import yt_dlp
        require_ffmpeg()
        opts = dict(format="bestaudio/best", outtmpl=f"{dest_dir}/%(id)s.%(ext)s",
                    postprocessors=[{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}])
        try:
            with self._ydl(**opts) as ydl:
                info = ydl.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as e:
            raise SongImportError(f"download failed: {e}") from e
        return Path(info["requested_downloads"][0]["filepath"])

"""Slow, opt-in: the real separator + RMVPE on a short generated clip. Run with `uv run pytest -m slow`.

Needs `uv sync --extra import`, ffmpeg and the models (SING_PITCH_MODELS or the default models dir; downloaded on
first use). The clip is macOS `say` speech over a synthetic backing: generated here, no copyrighted audio. Whether the
stems SOUND good is Simone's call, by listening; this only checks that the machinery produced sane output.
"""
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf
from audio import harmonic

from sing_pitch import config
from sing_pitch.library import Importer, Library

pytestmark = pytest.mark.slow
SR = 44100
TEXT = "Row row row your boat, gently down the stream. Merrily merrily merrily merrily, life is but a dream."


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    if shutil.which("say") is None:
        pytest.skip("needs macOS `say` to generate a voice")
    d = tmp_path_factory.mktemp("clip")
    subprocess.run(["say", "-o", str(d / "voice.aiff"), TEXT], check=True, stdin=subprocess.DEVNULL)
    voice, sr = sf.read(str(d / "voice.aiff"), dtype="float32")
    import soxr
    voice = soxr.resample(voice if voice.ndim == 1 else voice.mean(axis=1), sr, SR)
    voice = voice[: 12 * SR] / np.max(np.abs(voice)) * 0.6
    n = len(voice)
    chords = sum(harmonic(f, n / SR, amp=0.12, sr=SR)[:n] for f in (110.0, 138.6, 164.8, 220.0))
    hats = np.random.default_rng(3).standard_normal(n).astype(np.float32) * 0.03 * (np.arange(n) % (SR // 4) < 800)
    backing = (chords + hats).astype(np.float32)
    mix = voice + backing
    sf.write(str(d / "mix.wav"), mix / np.max(np.abs(mix)) * 0.9, SR, subtype="PCM_16")
    sf.write(str(d / "truth_vocal.wav"), voice, SR, subtype="PCM_16")
    return d


def sung_frames(truth_wav, n_frames):
    x, sr = sf.read(str(truth_wav), dtype="float32")
    hop = int(sr * 0.01)
    rms = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop + hop] ** 2)) for i in range(n_frames)])
    return rms > 0.1 * rms.max()


def real_importer(home):
    from sing_pitch.library import adapters
    models = config.models_dir()
    return Importer(Library(home), adapters.AudioSeparatorSeparator(models), adapters.RmvpeExtractor(models))


def test_real_import_of_a_file_makes_stems_and_a_voiced_reference_line(clip, tmp_path):
    song = real_importer(tmp_path).import_file(clip / "mix.wav")

    assert sf.info(str(song.vocals_path)).duration == pytest.approx(sf.info(str(clip / "mix.wav")).duration, abs=0.05)
    assert sf.info(str(song.backing_path)).duration == pytest.approx(sf.info(str(clip / "mix.wav")).duration, abs=0.05)
    line = song.load_reference()
    assert line.t.size > 500 and line.t[1] - line.t[0] == pytest.approx(0.01, abs=1e-4)
    sung = sung_frames(clip / "truth_vocal.wav", line.t.size)
    voiced = line.confidence >= 0.3
    assert voiced[sung].mean() > 0.5, f"only {voiced[sung].mean():.0%} of the sung frames came out voiced"
    assert np.median(line.hz[voiced & sung]) > 60            # a real pitch, not a floor value
    assert line.confidence[~sung].mean() < line.confidence[sung].mean()   # silence is less confident than speech


def test_real_import_of_stems_skips_the_separator_and_still_extracts_the_line(clip, tmp_path):
    folder = tmp_path / "Stems"
    folder.mkdir()
    shutil.copy(clip / "truth_vocal.wav", folder / "Vocals.wav")
    sf.write(str(folder / "Other.wav"), harmonic(110.0, 12.0, sr=SR)[: 12 * SR], SR)
    song = real_importer(tmp_path / "home").import_stems(folder)
    assert song.meta.stems_from == "logic"
    line = song.load_reference()
    sung = sung_frames(folder / "Vocals.wav", line.t.size)
    assert (line.confidence >= 0.3)[sung].mean() > 0.5

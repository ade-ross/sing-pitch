from importer_fakes import FakeDownloader, FakeExtractor, FakeSeparator, song_file

from sing_pitch import config
from sing_pitch.cli import import_main
from sing_pitch.library import Importer, Library


def fakes(home):
    return lambda: Importer(Library(home), FakeSeparator(), FakeExtractor(), FakeDownloader())


def test_the_import_command_picks_file_link_or_stems_folder_from_the_argument(tmp_path, capsys):
    home = tmp_path / "home"
    audio = song_file(tmp_path / "a.wav")
    assert import_main([str(audio)], make_importer=fakes(home)) == 0
    assert import_main(["https://youtu.be/abc123"], make_importer=fakes(home)) == 0
    out = capsys.readouterr().out
    assert "a" in out and "some-song-live" in out
    assert [s.meta.source_kind for s in Library(home).list_songs()] == ["file", "youtube"]


def test_a_missing_source_is_reported_and_exits_nonzero(tmp_path, capsys):
    assert import_main([str(tmp_path / "nope.wav")], make_importer=fakes(tmp_path / "home")) == 1
    assert "nope.wav" in capsys.readouterr().err


def test_the_model_folder_defaults_under_the_data_dir_and_can_be_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("SING_PITCH_HOME", str(tmp_path / "h"))
    monkeypatch.delenv("SING_PITCH_MODELS", raising=False)
    assert config.models_dir() == tmp_path / "h" / "models"
    monkeypatch.setenv("SING_PITCH_MODELS", str(tmp_path / "m"))
    assert config.models_dir() == tmp_path / "m"


def test_a_link_import_without_the_import_extra_says_how_to_install_it_and_exits_1(tmp_path, capsys, monkeypatch):
    import sys

    from sing_pitch.library.adapters import YtDlpDownloader

    monkeypatch.setitem(sys.modules, "yt_dlp", None)          # what `import yt_dlp` sees when the extra is missing
    make = lambda: Importer(Library(tmp_path / "home"), FakeSeparator(), FakeExtractor(), YtDlpDownloader())
    assert import_main(["https://youtu.be/abc12345678"], make_importer=make) == 1
    assert "uv sync --extra import" in capsys.readouterr().err


def test_downloading_without_the_import_extra_raises_the_friendly_error(tmp_path, monkeypatch):
    import sys

    import pytest

    from sing_pitch.library import SongImportError
    from sing_pitch.library import adapters

    monkeypatch.setitem(sys.modules, "yt_dlp", None)
    monkeypatch.setattr(adapters.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    with pytest.raises(SongImportError, match="uv sync --extra import"):
        adapters.YtDlpDownloader().download("https://youtu.be/abc12345678", tmp_path)


def corrupt_stems(tmp_path, which):
    import numpy as np
    import soundfile as sf

    folder = tmp_path / "corrupt"
    folder.mkdir()
    good = np.zeros((4800, 1), np.float32)
    for name in ("Vocals.wav", "Bass.wav"):
        if name == which:
            (folder / name).write_bytes(b"not a wav at all")
        else:
            sf.write(str(folder / name), good, 48000)
    return folder


def test_a_corrupt_vocal_stem_is_one_error_line_and_exit_1_without_a_traceback(tmp_path, capsys):
    folder = corrupt_stems(tmp_path, "Vocals.wav")
    assert import_main([str(folder)], make_importer=fakes(tmp_path / "home")) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("error: ") and "Vocals.wav" in err[0]
    assert Library(tmp_path / "home").list_songs() == []


def test_a_corrupt_other_stem_is_reported_the_same_way(tmp_path, capsys):
    folder = corrupt_stems(tmp_path, "Bass.wav")
    assert import_main([str(folder)], make_importer=fakes(tmp_path / "home")) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("error: ") and "Bass.wav" in err[0]


def test_a_truncated_stem_is_reported_the_same_way(tmp_path, capsys):
    import soundfile as sf

    folder = corrupt_stems(tmp_path, "none")
    whole = (folder / "Vocals.wav").read_bytes()
    (folder / "Vocals.wav").write_bytes(whole[:30])             # the header is cut off
    assert import_main([str(folder)], make_importer=fakes(tmp_path / "home")) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("error: ")

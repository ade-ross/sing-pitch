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

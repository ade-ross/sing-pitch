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

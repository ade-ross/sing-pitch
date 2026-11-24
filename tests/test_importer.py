import json

import numpy as np
import pytest
import soundfile as sf
from importer_fakes import FakeDownloader, FakeExtractor, FakeSeparator, song_file, write_wav
from audio import harmonic

from sing_pitch.library import Importer, Library, SongImportError


@pytest.fixture
def parts():
    return FakeSeparator(), FakeExtractor(), FakeDownloader()


def make_importer(tmp_path, parts):
    sep, ext, dl = parts
    return Importer(Library(tmp_path), separator=sep, extractor=ext, downloader=dl)


def test_importing_a_file_stores_vocal_backing_and_reference_line_in_the_library(tmp_path, parts):
    audio = song_file(tmp_path / "My Song.wav", seconds=2.0)
    song = make_importer(tmp_path / "home", parts).import_file(audio)

    reopened = Library(tmp_path / "home").get(song.id)
    assert reopened.vocals_path.exists() and reopened.backing_path.exists()
    assert sf.info(str(reopened.vocals_path)).duration == pytest.approx(2.0, abs=0.01)
    line = reopened.load_reference()
    assert line.t.shape == line.hz.shape == line.confidence.shape == (200,)
    assert line.t[1] == pytest.approx(0.01) and line.hz[0] == pytest.approx(220.0)
    assert reopened.meta.title == "My Song" and reopened.meta.source_kind == "file"
    assert reopened.meta.stems_from == "separated" and reopened.meta.separator_model == "fake-roformer"
    assert [s.id for s in Library(tmp_path / "home").list_songs()] == [song.id]


def test_importing_the_same_file_again_needs_no_work_and_gives_the_same_song(tmp_path, parts):
    sep, ext, _ = parts
    imp = make_importer(tmp_path / "home", parts)
    audio = song_file(tmp_path / "a.wav")
    first = imp.import_file(audio)
    again = imp.import_file(audio)
    assert again.id == first.id and len(Library(tmp_path / "home").list_songs()) == 1
    assert len(sep.calls) == 1 and len(ext.calls) == 1


def test_force_reimports_and_replaces_the_stored_song(tmp_path, parts):
    sep, _, _ = parts
    imp = make_importer(tmp_path / "home", parts)
    audio = song_file(tmp_path / "a.wav")
    imp.import_file(audio)
    imp.import_file(audio, force=True)
    assert len(sep.calls) == 2 and len(Library(tmp_path / "home").list_songs()) == 1


def test_a_failed_import_leaves_nothing_in_the_library(tmp_path, parts):
    class Boom(FakeExtractor):
        def extract(self, vocals):
            raise RuntimeError("model crashed")

    sep, _, dl = parts
    imp = Importer(Library(tmp_path / "home"), sep, Boom(), dl)
    with pytest.raises(RuntimeError):
        imp.import_file(song_file(tmp_path / "a.wav"))
    assert Library(tmp_path / "home").list_songs() == []
    assert not [p for p in (tmp_path / "home" / "songs").iterdir()]


def test_two_different_songs_with_the_same_title_get_distinct_ids(tmp_path, parts):
    imp = make_importer(tmp_path / "home", parts)
    (tmp_path / "x").mkdir(); (tmp_path / "y").mkdir()
    a = imp.import_file(write_wav(tmp_path / "x" / "Song.wav", harmonic(200.0, 1.0, sr=44100)))
    b = imp.import_file(write_wav(tmp_path / "y" / "Song.wav", harmonic(300.0, 1.0, sr=44100)))
    assert a.id != b.id


URL = "https://www.youtube.com/watch?v=abc12345678"


def test_a_youtube_link_is_downloaded_then_imported_and_remembered_by_video_id(tmp_path, parts):
    sep, ext, dl = parts
    imp = make_importer(tmp_path / "home", parts)
    song = imp.import_url(URL)
    assert song.meta.source_kind == "youtube" and song.meta.source_key == "youtube:abc12345678"
    assert song.meta.source_url == URL and song.meta.title == "Some Song / Live" and song.id == "some-song-live"
    assert song.meta.duration_s == 120.0
    assert song.vocals_path.exists() and song.load_reference().hz.size > 0

    again = imp.import_url("https://youtu.be/abc12345678")        # another spelling of the same video
    assert again.id == song.id
    assert len(dl.downloads) == 1 and len(sep.calls) == 1     # the second time nothing was downloaded or separated


@pytest.mark.parametrize("kwargs, message", [
    (dict(is_playlist=True), "playlist"),
    (dict(duration_s=40 * 60.0), "long"),
])
def test_playlists_and_very_long_videos_are_refused_before_downloading(tmp_path, kwargs, message):
    dl = FakeDownloader(**kwargs)
    imp = Importer(Library(tmp_path / "home"), FakeSeparator(), FakeExtractor(), dl)
    with pytest.raises(SongImportError, match=message):
        imp.import_url(URL)
    assert dl.downloads == []


def both(mono):
    """A mono stem as the stereo Backing holds it: the same signal in both channels."""
    return np.stack([mono, mono], axis=1)


def logic_folder(tmp_path, names=("Vocals", "Drums", "Bass"), sr_by_name=None):
    d = tmp_path / "Hello Stems"
    d.mkdir()
    for i, name in enumerate(names):
        sr = (sr_by_name or {}).get(name, 44100)
        write_wav(d / f"{name}.wav", harmonic(150.0 + 50 * i, 1.0, amp=0.2, sr=sr), sr)
    return d


def test_a_folder_of_logic_stems_is_imported_without_separating(tmp_path, parts):
    sep, ext, _ = parts
    folder = logic_folder(tmp_path)
    song = make_importer(tmp_path / "home", parts).import_stems(folder)

    assert sep.calls == [] and len(ext.calls) == 1
    assert song.meta.stems_from == "logic" and song.meta.separator_model is None
    assert song.meta.source_kind == "logic-stems" and song.meta.title == "Hello Stems"
    vocals, _ = sf.read(str(song.vocals_path), dtype="float32")
    backing, sr = sf.read(str(song.backing_path), dtype="float32")
    drums, _ = sf.read(str(folder / "Drums.wav"), dtype="float32")
    bass, _ = sf.read(str(folder / "Bass.wav"), dtype="float32")
    np.testing.assert_allclose(backing, both(drums + bass), atol=2e-4)       # Backing = everything that is not the vocal
    np.testing.assert_allclose(vocals, sf.read(str(folder / "Vocals.wav"), dtype="float32")[0], atol=2e-4)


def test_logic_stems_with_different_sample_rates_still_sum_to_one_backing(tmp_path, parts):
    folder = logic_folder(tmp_path, sr_by_name={"Bass": 48000})
    song = make_importer(tmp_path / "home", parts).import_stems(folder)
    info = sf.info(str(song.backing_path))
    assert info.samplerate == 44100 and info.duration == pytest.approx(1.0, abs=0.01)


def test_logic_stems_prefer_a_ready_made_backing_file(tmp_path, parts):
    folder = logic_folder(tmp_path, names=("Vocals", "Drums", "Backing"))
    song = make_importer(tmp_path / "home", parts).import_stems(folder)
    backing, _ = sf.read(str(song.backing_path), dtype="float32")
    ready, _ = sf.read(str(folder / "Backing.wav"), dtype="float32")
    np.testing.assert_allclose(backing, both(ready), atol=2e-4)


def test_logic_stems_without_a_vocal_file_say_so(tmp_path, parts):
    folder = logic_folder(tmp_path, names=("Drums", "Bass"))
    with pytest.raises(SongImportError, match="vocal"):
        make_importer(tmp_path / "home", parts).import_stems(folder)


def test_backing_vocals_in_logic_stems_stay_in_the_backing_not_the_lead(tmp_path, parts):
    folder = logic_folder(tmp_path, names=("Lead Vocals", "Backing Vocals", "Drums"))
    song = make_importer(tmp_path / "home", parts).import_stems(folder)
    vocals, _ = sf.read(str(song.vocals_path), dtype="float32")
    lead, _ = sf.read(str(folder / "Lead Vocals.wav"), dtype="float32")
    np.testing.assert_allclose(vocals, lead, atol=2e-4)
    backing, _ = sf.read(str(song.backing_path), dtype="float32")
    expected = sum(sf.read(str(folder / f"{n}.wav"), dtype="float32")[0] for n in ("Backing Vocals", "Drums"))
    np.testing.assert_allclose(backing, both(expected), atol=2e-4)       # the drums are in the Backing too


def test_importing_a_known_youtube_link_again_works_offline(tmp_path, parts):
    imp = make_importer(tmp_path / "home", parts)
    first = imp.import_url(URL)
    _, _, dl = parts
    dl.probe = lambda url: (_ for _ in ()).throw(RuntimeError("offline"))
    assert imp.import_url("https://youtu.be/abc12345678?t=5").id == first.id
    assert imp.import_url("https://www.youtube.com/shorts/abc12345678").id == first.id


def test_a_failed_force_reimport_keeps_the_stored_song_untouched(tmp_path, parts):
    sep, _, dl = parts
    home = tmp_path / "home"
    audio = song_file(tmp_path / "a.wav")
    first = make_importer(home, parts).import_file(audio)
    before = first.vocals_path.read_bytes()

    class Boom(FakeExtractor):
        def extract(self, vocals):
            raise RuntimeError("model crashed")

    with pytest.raises(RuntimeError):
        Importer(Library(home), sep, Boom(), dl).import_file(audio, force=True)
    kept = Library(home).get(first.id)
    assert kept.vocals_path.read_bytes() == before and kept.load_reference().hz.size > 0
    assert [p.name for p in (home / "songs").iterdir()] == [first.id]        # no staging folder left behind


def test_a_successful_force_reimport_replaces_the_song_under_the_same_id(tmp_path, parts):
    sep, _, _ = parts
    home = tmp_path / "home"
    imp = make_importer(home, parts)
    audio = song_file(tmp_path / "a.wav")
    first = imp.import_file(audio)
    renamed = audio.rename(tmp_path / "different name.wav")                  # same bytes, other title
    again = imp.import_file(renamed, force=True)
    assert again.id == first.id and len(sep.calls) == 2
    assert [s.id for s in Library(home).list_songs()] == [first.id]
    assert [p.name for p in (home / "songs").iterdir()] == [first.id]
    assert Library(home).get(first.id).meta.title == "different name"


def test_a_failed_force_reimport_of_stems_and_links_keeps_the_stored_song(tmp_path, parts):
    sep, _, dl = parts
    home = tmp_path / "home"

    class Boom(FakeExtractor):
        def extract(self, vocals):
            raise RuntimeError("model crashed")

    folder = logic_folder(tmp_path)
    good = make_importer(home, parts)
    stems_song, url_song = good.import_stems(folder), good.import_url(URL)
    bad = Importer(Library(home), sep, Boom(), dl)
    for call in (lambda: bad.import_stems(folder, force=True), lambda: bad.import_url(URL, force=True)):
        with pytest.raises(RuntimeError):
            call()
    assert {s.id for s in Library(home).list_songs()} == {stems_song.id, url_song.id}
    assert all(s.vocals_path.exists() for s in Library(home).list_songs())


def test_a_no_vocals_backing_stem_is_the_backing_and_not_a_second_lead_vocal(tmp_path, parts):
    for name in ("Song no vocals", "Song (No Vocals)", "Song_no_vocals", "Song without vocals"):
        folder = tmp_path / name.replace(" ", "-")
        folder.mkdir()
        for stem, hz in (("Lead Vocals", 220.0), (name, 150.0), ("Drums", 300.0)):
            write_wav(folder / f"{stem}.wav", harmonic(hz, 1.0, amp=0.2, sr=44100), 44100)
        song = make_importer(tmp_path / f"home-{folder.name}", parts).import_stems(folder)
        backing, _ = sf.read(str(song.backing_path), dtype="float32")
        ready, _ = sf.read(str(folder / f"{name}.wav"), dtype="float32")
        np.testing.assert_allclose(backing, both(ready), atol=2e-4)          # not summed with the drums


def test_the_backing_is_stereo_even_when_the_lead_vocal_is_mono_and_keeps_both_channels(tmp_path, parts):
    folder = tmp_path / "Mixed Stems"
    folder.mkdir()
    n = 44100
    left, right = harmonic(200.0, 1.0, amp=0.2, sr=44100), harmonic(300.0, 1.0, amp=0.2, sr=44100)
    write_wav(folder / "Vocals.wav", harmonic(220.0, 1.0, sr=44100), 44100)                 # mono lead
    write_wav(folder / "Drums.wav", np.stack([left, right], axis=1), 44100)                # stereo
    bass = harmonic(100.0, 1.0, amp=0.1, sr=44100)
    write_wav(folder / "Bass.wav", bass, 44100)                                             # mono
    song = make_importer(tmp_path / "home", parts).import_stems(folder)
    backing, _ = sf.read(str(song.backing_path), dtype="float32", always_2d=True)
    assert backing.shape == (n, 2)
    np.testing.assert_allclose(backing[:, 0], left + bass, atol=2e-4)
    np.testing.assert_allclose(backing[:, 1], right + bass, atol=2e-4)
    assert sf.info(str(song.vocals_path)).channels == 1

"""Sections are stored per song in the library: saved by name, renamed, deleted; the unnamed mark is kept too."""
import pytest
from melody_helpers import library_song, reference_line

from sing_pitch.library import Library, Section

NOTES = [(-900, 0.6), (-500, 0.6)]


def two_songs(tmp_path):
    a = library_song(tmp_path, reference_line(NOTES), title="Song A")
    b = library_song(tmp_path, reference_line(NOTES), title="Song B")
    return a.id, b.id


def test_sections_are_saved_by_name_per_song_and_survive_a_reopen(tmp_path):
    a, b = two_songs(tmp_path)
    Library(tmp_path).sections(a).save("Chorus 1", 12.0, 18.5)
    Library(tmp_path).sections(a).save("Verse", 2.0, 8.0)
    assert Library(tmp_path).sections(a).saved() == [Section("Chorus 1", 12.0, 18.5), Section("Verse", 2.0, 8.0)]
    assert Library(tmp_path).sections(b).saved() == []


def test_saving_under_an_existing_name_replaces_it_and_start_end_are_put_in_order(tmp_path):
    a, _ = two_songs(tmp_path)
    store = Library(tmp_path).sections(a)
    store.save("Chorus 1", 12.0, 18.5)
    store.save("Chorus 1", 20.0, 14.0)
    assert Library(tmp_path).sections(a).saved() == [Section("Chorus 1", 14.0, 20.0)]


def test_rename_and_delete(tmp_path):
    a, _ = two_songs(tmp_path)
    store = Library(tmp_path).sections(a)
    store.save("Chorus 1", 12.0, 18.5)
    store.save("Verse", 2.0, 8.0)
    store.rename("Chorus 1", "Chorus")
    with pytest.raises(ValueError):
        store.rename("Verse", "Chorus")             # names stay unique
    store.delete("Verse")
    assert Library(tmp_path).sections(a).saved() == [Section("Chorus", 12.0, 18.5)]


def test_the_unnamed_mark_is_kept_until_replaced(tmp_path):
    a, _ = two_songs(tmp_path)
    assert Library(tmp_path).sections(a).mark is None
    Library(tmp_path).sections(a).set_mark(5.0, 3.0)
    assert Library(tmp_path).sections(a).mark == Section(None, 3.0, 5.0)
    Library(tmp_path).sections(a).set_mark(None)
    assert Library(tmp_path).sections(a).mark is None


def test_a_reimport_of_the_song_keeps_its_sections(tmp_path):
    a, _ = two_songs(tmp_path)
    lib = Library(tmp_path)
    lib.sections(a).save("Chorus 1", 12.0, 18.5)
    old = lib.get(a)
    staging = lib.new_staging()
    for f in old.dir.iterdir():
        if f.name != "sections.json":
            (staging / f.name).write_bytes(f.read_bytes())
    fields = {k: v for k, v in vars(old.meta).items() if k not in ("id", "title")}
    lib.commit(staging, fields, old.meta.title, replacing=a)
    assert Library(tmp_path).sections(a).saved() == [Section("Chorus 1", 12.0, 18.5)]

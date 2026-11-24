"""A song folder that cannot be read is skipped with a message; it never stops the window from opening (ticket #40)."""
import pytest
from audio import SR, quiet
from melody_helpers import library_song, reference_line

from sing_pitch.audio_input import ArrayInput
from sing_pitch.library import Library

NOTES = [(-900, 0.6), (-500, 0.6), (-200, 0.8)]

BROKEN = {
    "not json": "{ this is not json",
    "empty file": "",
    "a list": "[1, 2]",
    "missing fields": '{"id": "x", "title": "X"}',
}


def broken_song(home, folder, text):
    (home / "songs" / folder).mkdir(parents=True)
    (home / "songs" / folder / "meta.json").write_text(text)


@pytest.mark.parametrize("text", BROKEN.values(), ids=BROKEN.keys())
def test_listing_skips_a_song_with_an_unreadable_meta_and_keeps_the_good_ones(tmp_path, text):
    good = library_song(tmp_path, reference_line(NOTES))
    broken_song(tmp_path, "broken", text)
    assert [s.id for s in Library(tmp_path).list_songs()] == [good.id]


def test_scan_says_which_folder_was_skipped_and_why(tmp_path):
    library_song(tmp_path, reference_line(NOTES))
    broken_song(tmp_path, "broken", "{ nope")
    songs, problems = Library(tmp_path).scan()
    assert len(songs) == 1 and len(problems) == 1
    assert "broken" in problems[0] and "meta.json" in problems[0]


def test_a_library_of_good_songs_has_no_problems(tmp_path):
    library_song(tmp_path, reference_line(NOTES))
    assert Library(tmp_path).scan()[1] == []


def test_finding_a_song_by_its_source_survives_a_broken_folder(tmp_path):
    good = library_song(tmp_path, reference_line(NOTES))
    broken_song(tmp_path, "aaa-broken", "{ nope")
    assert Library(tmp_path).find_by_source(good.meta.source_key).id == good.id


def window(qtbot, home):
    from sing_pitch.pitch.swiftf0 import SwiftF0Tracker
    from sing_pitch.ui import MainWindow

    win = MainWindow(SwiftF0Tracker(), home=home, input_factory=lambda _i: ArrayInput(quiet(0.5), SR, 256))
    qtbot.addWidget(win)
    return win


def test_the_window_opens_lists_the_good_song_and_says_what_it_skipped(qtbot, tmp_path):
    good = library_song(tmp_path, reference_line(NOTES), title="Good Song")
    broken_song(tmp_path, "broken", "{ nope")
    win = window(qtbot, tmp_path)
    assert [win.song_box.itemData(i) for i in range(win.song_box.count())] == [good.id]
    assert "broken" in win.summary_label.text() and "skipped" in win.summary_label.text().lower()


def test_the_window_opens_when_the_only_song_is_unreadable(qtbot, tmp_path):
    broken_song(tmp_path, "broken", "{ nope")
    win = window(qtbot, tmp_path)
    assert win.song_box.currentData() is None and "broken" in win.summary_label.text()


def test_a_healthy_library_leaves_the_usual_first_message(qtbot, tmp_path):
    library_song(tmp_path, reference_line(NOTES))
    assert window(qtbot, tmp_path).summary_label.text() == "Press Start and sing."

"""Per-song player settings (ticket #15): Guide vocal on/off, both volumes, the last Speed and the last Section are
kept in the song's library folder and come back when the song is chosen again."""
import json

from audio import quiet
from melody_helpers import library_song, reference_line
from test_melody_window import window
from test_speed_cache import CountingStretcher

from sing_pitch.library import Library, PlayerSettings

NOTES = [(-900, 0.6), (-500, 0.6), (-200, 0.8)]


# --- the store ---------------------------------------------------------------------------------------------------

def test_a_song_with_nothing_saved_opens_with_the_defaults(tmp_path):
    song = library_song(tmp_path, reference_line(NOTES))
    assert Library(tmp_path).player_settings(song.id).load() == PlayerSettings(
        guide_on=False, guide_gain=1.0, backing_gain=1.0, speed_pct=100, section=None)


def test_settings_are_saved_per_song_and_survive_a_reopen(tmp_path):
    a = library_song(tmp_path, reference_line(NOTES), title="Song A")
    b = library_song(tmp_path, reference_line(NOTES), title="Song B")
    Library(tmp_path).player_settings(a.id).save(PlayerSettings(True, 0.4, 0.7, 70, "Chorus 1"))
    assert Library(tmp_path).player_settings(a.id).load() == PlayerSettings(True, 0.4, 0.7, 70, "Chorus 1")
    assert Library(tmp_path).player_settings(b.id).load() == PlayerSettings()


def test_a_damaged_file_gives_the_defaults_and_values_are_clamped(tmp_path):
    song = library_song(tmp_path, reference_line(NOTES))
    path = song.dir / "player.json"
    path.write_text("{ not json")
    assert Library(tmp_path).player_settings(song.id).load() == PlayerSettings()
    path.write_text(json.dumps({"guide_on": True, "guide_gain": 7, "backing_gain": -1, "speed_pct": 400}))
    loaded = Library(tmp_path).player_settings(song.id).load()
    assert (loaded.guide_on, loaded.guide_gain, loaded.backing_gain, loaded.speed_pct) == (True, 1.0, 0.0, 100)


def test_a_reimport_of_the_song_keeps_its_player_settings(tmp_path):
    song = library_song(tmp_path, reference_line(NOTES))
    lib = Library(tmp_path)
    lib.player_settings(song.id).save(PlayerSettings(True, 0.5, 0.5, 80, None))
    staging = lib.new_staging()
    for f in song.dir.iterdir():
        if f.is_file() and f.name != "player.json":
            (staging / f.name).write_bytes(f.read_bytes())
    fields = {k: v for k, v in vars(song.meta).items() if k not in ("id", "title")}
    lib.commit(staging, fields, song.meta.title, replacing=song.id)
    assert Library(tmp_path).player_settings(song.id).load() == PlayerSettings(True, 0.5, 0.5, 80, None)


# --- the window --------------------------------------------------------------------------------------------------

def two_song_window(qtbot, tmp_path):
    a = library_song(tmp_path, reference_line(NOTES), title="Song A", seconds=2.0)
    b = library_song(tmp_path, reference_line(NOTES), title="Song B", seconds=2.0)
    win = window(qtbot, tmp_path, quiet(1.0), stretcher=CountingStretcher())
    win.reference_box.setCurrentText("Melody")
    return win, a, b


def controls(win):
    return (win.guide_box.isChecked(), win.guide_slider.value(), win.backing_slider.value(), win.speed_box.value(),
            win.section_box.currentData())


def test_the_controls_are_saved_for_the_chosen_song_as_they_change(qtbot, tmp_path):
    win, a, _ = two_song_window(qtbot, tmp_path)
    win.guide_box.setChecked(True)
    win.guide_slider.setValue(40)
    win.backing_slider.setValue(70)
    win.timeline.section_marked.emit(0.5, 1.5)
    win.section_name_edit.setText("Chorus 1")
    win.save_section_button.click()
    win.speed_box.setValue(70)
    assert Library(tmp_path).player_settings(a.id).load() == PlayerSettings(True, 0.4, 0.7, 70, "Chorus 1")
    qtbot.waitUntil(lambda: win.speed_label.text() == "", timeout=5000)      # let the stretch thread finish


def test_a_song_reopens_with_its_last_guide_volumes_speed_and_section(qtbot, tmp_path):
    win, a, b = two_song_window(qtbot, tmp_path)
    win.guide_box.setChecked(True)
    win.guide_slider.setValue(40)
    win.backing_slider.setValue(70)
    win.timeline.section_marked.emit(0.5, 1.5)
    win.section_name_edit.setText("Chorus 1")
    win.save_section_button.click()
    win.speed_box.setValue(70)
    qtbot.waitUntil(lambda: win.speed_label.text() == "", timeout=5000)

    other = window(qtbot, tmp_path, quiet(1.0), stretcher=CountingStretcher())      # a new window, a new run
    other.reference_box.setCurrentText("Melody")
    assert controls(other) == (True, 40, 70, 70, "Chorus 1")
    assert other.timeline.section == (0.5, 1.5)
    qtbot.waitUntil(lambda: other._speed_ready is not None, timeout=5000)      # its cached Speed loads in a thread


def test_choosing_another_song_restores_that_songs_settings_and_does_not_copy_the_first_ones(qtbot, tmp_path):
    win, a, b = two_song_window(qtbot, tmp_path)
    Library(tmp_path).player_settings(b.id).save(PlayerSettings(False, 0.2, 0.9, 100, None))
    win.guide_box.setChecked(True)
    win.guide_slider.setValue(40)                                # settings of Song A
    win.song_box.setCurrentIndex(win.song_box.findData(b.id))
    assert win.song_box.currentData() == b.id and controls(win) == (False, 20, 90, 100, None)
    assert Library(tmp_path).player_settings(a.id).load().guide_gain == 0.4      # untouched by the switch
    win.song_box.setCurrentIndex(win.song_box.findData(a.id))
    assert controls(win) == (True, 40, 100, 100, None)


def test_the_restored_gains_reach_the_player_at_start(qtbot, tmp_path):
    win, a, _ = two_song_window(qtbot, tmp_path)
    Library(tmp_path).player_settings(a.id).save(PlayerSettings(True, 0.4, 0.7, 100, None))
    win.song_box.setCurrentIndex(win.song_box.findData(a.id))
    win.reference_box.setCurrentText("Nearest note")
    win.reference_box.setCurrentText("Melody")
    assert controls(win)[:3] == (True, 40, 70)
    assert win._gains() == (0.7, 0.4)

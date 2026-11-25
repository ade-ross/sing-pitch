"""#58: text that outlives the state it described: the Pause button after Stop, a Section message after the next mark
or Clear, the tempo note outside Melody."""
import pytest
from audio import quiet
from melody_helpers import library_song, reference_line
from qa_helpers import NOTES, configure_latency, singing_song
from test_count_in_window import FakeDetector, melody_window
from test_navigate_and_loops import SCALE
from test_window_sections import section_window

from sing_pitch.library.tempo import Tempo, save_tempo
from sing_pitch.qa import QA

LINE = [(-900, 0.6), (-500, 0.6), (-200, 0.8)]


def test_the_pause_button_reads_pause_again_after_stop(tmp_path):
    singing_song(tmp_path, notes=NOTES * 3, tail_s=3.0)
    configure_latency(tmp_path, 0.12)
    with QA(tmp_path, mic="echo", time_scale=SCALE) as qa:
        qa.set("reference_box", "Melody")
        qa.start_take()
        qa.wait(0.5)
        qa.pause()
        assert qa.text("pause_button") == "Play"
        qa.stop_take()
        assert qa.text("pause_button") == "Pause" and not qa.widget("pause_button").isEnabled()


def test_a_missing_name_message_goes_when_the_mark_is_cleared(qtbot, tmp_path):
    win, _ = section_window(qtbot, tmp_path)
    win.timeline.section_marked.emit(0.2, 0.9)
    win.save_section_button.click()
    assert win.summary_label.text() == "Type a name for the Section to save it."
    win.clear_section()
    assert win.summary_label.text() == ""


def test_typing_a_name_takes_the_missing_name_message_away(qtbot, tmp_path):
    win, _ = section_window(qtbot, tmp_path)
    win.timeline.section_marked.emit(0.2, 0.9)
    win.save_section_button.click()
    win.section_name_edit.setText("C")
    assert win.summary_label.text() == ""


@pytest.mark.parametrize("then", ["a new mark", "Clear"])
def test_the_deleted_message_goes_with_the_next_mark_or_clear(qtbot, tmp_path, then):
    win, _ = section_window(qtbot, tmp_path)
    win.timeline.section_marked.emit(0.2, 0.9)
    win.section_name_edit.setText("A")
    win.save_section_button.click()
    win.delete_section_button.click()
    assert win.summary_label.text() == 'Deleted Section "A".'
    if then == "Clear":
        win.clear_section()
    else:
        win.timeline.section_marked.emit(0.5, 1.5)
    assert win.summary_label.text() == ""


def test_a_section_message_is_not_wiped_when_something_else_has_been_said_since(qtbot, tmp_path):
    win, _ = section_window(qtbot, tmp_path)
    win.timeline.section_marked.emit(0.2, 0.9)
    win.save_section_button.click()                         # "Type a name..."
    win.summary_label.setText("Singing...")
    win.clear_section()
    assert win.summary_label.text() == "Singing..."


def test_a_section_message_goes_when_the_reference_changes(qtbot, tmp_path):
    win, _ = section_window(qtbot, tmp_path)
    win.timeline.section_marked.emit(0.2, 0.9)
    win.save_section_button.click()
    win.reference_box.setCurrentText("Key")
    assert win.summary_label.text() == ""


def test_the_tempo_note_goes_when_the_reference_is_not_melody(qtbot, tmp_path):
    song = library_song(tmp_path, reference_line(LINE))
    save_tempo(song.dir, Tempo(143.55))
    win = melody_window(qtbot, tmp_path, FakeDetector())
    win.between_box.setCurrentText("Count-in")
    win.bpm_box.setValue(72.0)
    win.bpm_box.setValue(0.0)
    assert "detected tempo" in win.tempo_label.text()
    win.reference_box.setCurrentText("Key")
    assert win.tempo_label.text() == ""
    win.reference_box.setCurrentText("Nearest note")
    assert win.tempo_label.text() == ""


def test_the_jumped_message_goes_after_a_few_seconds_and_the_line_says_what_is_true_now(qtbot, tmp_path):
    win, _ = section_window(qtbot, tmp_path)
    win._jumped_timer.setInterval(30)
    win._say_jumped("Jumped. Take saved: something")
    assert win.summary_label.text() == "Jumped. Take saved: something"
    qtbot.waitUntil(lambda: win.summary_label.text() == "", timeout=2000)


def test_a_jumped_message_that_was_replaced_is_left_alone(qtbot, tmp_path):
    win, _ = section_window(qtbot, tmp_path)
    win._jumped_timer.setInterval(30)
    win._say_jumped("Jumped. Take saved: something")
    win.summary_label.setText("Pass 2 of X: In tune 80%")
    qtbot.wait(150)
    assert win.summary_label.text() == "Pass 2 of X: In tune 80%"

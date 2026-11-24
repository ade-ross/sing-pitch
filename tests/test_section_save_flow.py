"""Saving a Section never silently replaces another one (ticket #40, QA #26): a new mark clears the name field, and
saving onto a name that is taken asks first, inline in the window (no dialog)."""
from test_window_sections import section_window, song_id

from sing_pitch.library import Library, Section


def saved(tmp_path, win):
    return Library(tmp_path).sections(song_id(win)).saved()


def save_as(win, name, start, end):
    win.timeline.section_marked.emit(start, end)
    win.section_name_edit.setText(name)
    win.save_section_button.click()


def test_a_new_mark_clears_the_name_field(qtbot, tmp_path):
    win, _ = section_window(qtbot, tmp_path)
    save_as(win, "Chorus 1", 0.2, 0.9)
    assert win.section_name_edit.text() == "Chorus 1"
    win.timeline.section_marked.emit(0.5, 1.5)
    assert win.section_box.currentText() == "unsaved mark" and win.section_name_edit.text() == ""


def test_a_new_mark_clears_a_name_typed_for_the_last_one_even_when_nothing_was_saved(qtbot, tmp_path):
    win, _ = section_window(qtbot, tmp_path)
    win.timeline.section_marked.emit(0.2, 0.9)
    win.section_name_edit.setText("half typed")
    win.timeline.section_marked.emit(0.5, 1.5)
    assert win.section_name_edit.text() == ""


def test_choosing_unsaved_mark_or_deleting_or_clearing_empties_the_name_field(qtbot, tmp_path):
    win, _ = section_window(qtbot, tmp_path)
    save_as(win, "Chorus 1", 0.2, 0.9)
    win.section_box.setCurrentIndex(0)
    assert win.section_name_edit.text() == ""
    win.section_box.setCurrentText("Chorus 1")
    assert win.section_name_edit.text() == "Chorus 1"
    win.delete_section_button.click()
    assert win.section_name_edit.text() == ""


def test_saving_a_new_name_needs_no_question(qtbot, tmp_path):
    win, _ = section_window(qtbot, tmp_path)
    save_as(win, "Chorus 1", 0.2, 0.9)
    save_as(win, "Verse", 0.5, 1.5)
    assert saved(tmp_path, win) == [Section("Chorus 1", 0.2, 0.9), Section("Verse", 0.5, 1.5)]
    assert win.overwrite_button.isHidden() and win.cancel_overwrite_button.isHidden()


def test_saving_onto_a_taken_name_asks_and_changes_nothing_until_overwrite(qtbot, tmp_path):
    win, _ = section_window(qtbot, tmp_path)
    save_as(win, "Chorus 1", 0.2, 0.9)
    win.timeline.section_marked.emit(0.5, 1.5)
    win.section_name_edit.setText("Chorus 1")
    win.save_section_button.click()
    assert saved(tmp_path, win) == [Section("Chorus 1", 0.2, 0.9)]
    assert not win.overwrite_button.isHidden() and not win.cancel_overwrite_button.isHidden()
    assert "Chorus 1" in win.summary_label.text() and "already" in win.summary_label.text()
    win.overwrite_button.click()
    assert saved(tmp_path, win) == [Section("Chorus 1", 0.5, 1.5)]
    assert win.overwrite_button.isHidden() and win.cancel_overwrite_button.isHidden()
    assert win.section_box.currentText() == "Chorus 1"


def test_cancel_keeps_the_old_section_and_hides_the_question(qtbot, tmp_path):
    win, _ = section_window(qtbot, tmp_path)
    save_as(win, "Chorus 1", 0.2, 0.9)
    save_as(win, "Chorus 1", 0.5, 1.5)
    win.cancel_overwrite_button.click()
    assert saved(tmp_path, win) == [Section("Chorus 1", 0.2, 0.9)]
    assert win.overwrite_button.isHidden() and win.cancel_overwrite_button.isHidden()


def ask(win):
    """Chorus 1 is saved at 0.2-0.9; a new mark is typed onto the same name and Save asks."""
    save_as(win, "Chorus 1", 0.2, 0.9)
    save_as(win, "Chorus 1", 0.5, 1.5)
    assert not win.overwrite_button.isHidden()


def test_the_question_goes_away_when_a_new_mark_is_dragged(qtbot, tmp_path):
    win, _ = section_window(qtbot, tmp_path)
    ask(win)
    win.timeline.section_marked.emit(0.1, 0.4)
    assert win.overwrite_button.isHidden() and win.cancel_overwrite_button.isHidden()


def test_the_question_goes_away_when_the_name_is_edited(qtbot, tmp_path):
    win, _ = section_window(qtbot, tmp_path)
    ask(win)
    win.section_name_edit.setText("Chorus 2")
    assert win.overwrite_button.isHidden()


def test_the_question_goes_away_when_the_picker_changes(qtbot, tmp_path):
    win, _ = section_window(qtbot, tmp_path)
    ask(win)
    win.section_box.setCurrentText("Chorus 1")
    assert win.overwrite_button.isHidden()
    assert saved(tmp_path, win) == [Section("Chorus 1", 0.2, 0.9)]


def test_saving_the_selected_section_under_its_own_name_needs_no_question(qtbot, tmp_path):
    win, _ = section_window(qtbot, tmp_path)
    save_as(win, "Chorus 1", 0.2, 0.9)
    win.save_section_button.click()                  # Chorus 1 is selected and is what the strip shows
    assert win.overwrite_button.isHidden() and saved(tmp_path, win) == [Section("Chorus 1", 0.2, 0.9)]


def test_the_question_works_without_a_dialog(qtbot, tmp_path, monkeypatch):
    from PySide6 import QtWidgets

    def no_dialog(*a, **k):
        raise AssertionError("a modal dialog would block the test")

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", no_dialog)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", no_dialog)
    win, _ = section_window(qtbot, tmp_path)
    save_as(win, "Chorus 1", 0.2, 0.9)
    save_as(win, "Chorus 1", 0.5, 1.5)

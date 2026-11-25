"""#55: the History table can tell Passes apart (Section and Pass columns, time to the second) and an open History
window shows a Take as soon as it is saved, instead of saying "No Takes yet" until Refresh."""
import dataclasses

from audio import SR, harmonic
from frames import hz_at
from test_history import write_table

from sing_pitch.audio_input import ArrayInput
from sing_pitch.live import LiveRunner
from sing_pitch.pitch.swiftf0 import SwiftF0Tracker
from sing_pitch.reference import NearestNote
from sing_pitch.ui.history_view import TABLE_HEADERS, HistoryView


def pass_row(i, second, pass_index, section="Chorus 1", pct="77"):
    return {"id": i, "started_at": f"2026-11-25T07:40:{second:02d}", "reference": "melody", "song_id": "tulip",
            "pct_in_tune": pct, "median_abs_cents": "7", "section_name": section, "pass_index": str(pass_index),
            "speed_pct": "100"}


def cells(v, r):
    return {h: v.takes_table.item(r, c).text() for c, h in enumerate(TABLE_HEADERS)}


def test_four_passes_of_one_loop_show_as_four_different_rows(qtbot, tmp_path):
    write_table(tmp_path, [pass_row(f"p{n}", 10 * n, n) for n in range(1, 5)])
    v = HistoryView(tmp_path)
    qtbot.addWidget(v)
    assert "Section" in TABLE_HEADERS and "Pass" in TABLE_HEADERS
    rows = [cells(v, r) for r in range(v.takes_table.rowCount())]
    assert [r["Pass"] for r in rows] == ["4", "3", "2", "1"]                       # newest first
    assert [r["When"] for r in rows] == [f"2026-11-25 07:40:{s}" for s in ("40", "30", "20", "10")]
    assert {r["Section"] for r in rows} == {"Chorus 1"}
    assert len({tuple(r.values()) for r in rows}) == 4


def test_a_whole_song_take_has_no_section_or_pass(qtbot, tmp_path):
    whole = {"id": "w", "started_at": "2026-11-25T08:00:00", "reference": "nearest", "pct_in_tune": "60",
             "median_abs_cents": "9"}
    write_table(tmp_path, [whole])
    v = HistoryView(tmp_path)
    qtbot.addWidget(v)
    assert cells(v, 0)["Section"] == "" and cells(v, 0)["Pass"] == ""


def sing_into(home, tracker):
    runner = LiveRunner(ArrayInput(harmonic(hz_at(0), 2.0), SR, chunk=256), tracker, NearestNote(), home=home)
    runner.start()
    return runner.stop()


def test_an_open_history_window_shows_a_take_saved_after_it_was_opened(qtbot, tmp_path):
    v = HistoryView(tmp_path)
    qtbot.addWidget(v)
    v.show()
    assert "No Takes yet" in v.empty_label.text() and v.takes_table.rowCount() == 0
    record = sing_into(tmp_path, SwiftF0Tracker())
    qtbot.waitUntil(lambda: v.takes_table.rowCount() == 1, timeout=5000)           # nobody pressed Refresh
    assert v.takes_table.item(0, 0).data(0x0100) == record.id
    assert v.empty_label.isHidden()


def test_a_reload_keeps_the_chosen_row(qtbot, tmp_path):
    write_table(tmp_path, [pass_row(f"p{n}", 10 * n, n) for n in range(1, 4)])
    v = HistoryView(tmp_path)
    qtbot.addWidget(v)
    v.takes_table.selectRow(1)                                                   # Pass 2
    chosen = v.takes_table.item(1, 0).data(0x0100)
    write_table(tmp_path, [pass_row(f"p{n}", 10 * n, n) for n in range(1, 5)])     # Pass 4 arrives
    v.reload()
    rows = v.takes_table.selectionModel().selectedRows()
    assert len(rows) == 1 and v.takes_table.item(rows[0].row(), 0).data(0x0100) == chosen

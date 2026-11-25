"""#54: Stop says what became of the Take that was being sung, and a Take with nothing scored is not saved, so it never
becomes "Replay last Take"."""
import csv

import pytest
from audio import SR, harmonic, quiet
from frames import hz_at
from qa_helpers import NOTES, configure_latency, singing_song
from test_navigate_and_loops import SCALE, echo_singer

from sing_pitch.audio_input import ArrayInput
from sing_pitch.live import LiveRunner
from sing_pitch.pitch.swiftf0 import SwiftF0Tracker
from sing_pitch.qa import QA
from sing_pitch.reference import NearestNote
from sing_pitch.take_store import last_take_id


@pytest.fixture(scope="module")
def tracker():
    return SwiftF0Tracker()


def rows(home):
    path = home / "takes.csv"
    return list(csv.DictReader(open(path))) if path.exists() else []


def run(tracker, home, audio):
    runner = LiveRunner(ArrayInput(audio, SR, chunk=256), tracker, NearestNote(), home=home)
    runner.start()
    return runner, runner.stop()


def test_a_take_with_nothing_scored_is_not_saved(tracker, tmp_path):
    runner, record = run(tracker, tmp_path, quiet(1.0))
    assert rows(tmp_path) == [] and not (tmp_path / "takes").exists()
    assert runner.dropped is record and record.summary.n_scored == 0
    assert last_take_id(tmp_path) is None


def test_an_empty_last_take_does_not_replace_the_one_before_as_the_replay_target(tracker, tmp_path):
    _, first = run(tracker, tmp_path, harmonic(hz_at(0), 2.0))
    assert [r["id"] for r in rows(tmp_path)] == [first.id]
    runner, silent = run(tracker, tmp_path, quiet(1.0))
    assert runner.dropped is silent
    assert [r["id"] for r in rows(tmp_path)] == [first.id] and last_take_id(tmp_path) == first.id


def test_a_short_take_with_something_scored_is_still_saved_when_it_is_all_there_is(tracker, tmp_path):
    runner, record = run(tracker, tmp_path, harmonic(hz_at(0), 1.0))
    assert 0 < record.summary.n_scored * 0.016 < 3.0
    assert runner.dropped is None and [r["id"] for r in rows(tmp_path)] == [record.id]


def test_a_take_saved_before_this_fix_with_nothing_scored_is_never_the_replay_target(tracker, tmp_path):
    _, good = run(tracker, tmp_path, harmonic(hz_at(0), 2.0))
    with open(tmp_path / "takes.csv", newline="") as fh:                  # what the old app wrote for an empty Take
        lines = fh.read().splitlines()
    header = lines[0].split(",")
    empty = {c: "" for c in header} | {"id": "20260101-000000", "started_at": "2026-01-01T00:00:00",
                                       "reference": "nearest", "n_scored_frames": "0"}
    with open(tmp_path / "takes.csv", "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=header).writerow(empty)
    assert last_take_id(tmp_path) == good.id


def test_stop_in_a_dropped_last_pass_says_so_and_shows_the_pass_that_was_kept(tmp_path):
    singing_song(tmp_path, notes=NOTES * 3, tail_s=3.0)
    configure_latency(tmp_path, 0.12)
    with QA(tmp_path, mic="echo", time_scale=1.0) as qa:      # the tracker thread keeps up whatever the load
        qa.set("reference_box", "Melody")
        qa.set("guide_box", True)
        qa.set("guide_slider", 100)
        qa.set("backing_slider", 0)
        qa.set("notes_box", False)
        qa.mark_section(1.0, 9.0)
        qa.set("loop_box", True)
        qa.start_take()
        qa.wait_until(lambda: len(qa.takes()) == 1, timeout_s=120)             # Pass 1 saved at the wrap
        qa.wait_until(lambda: "Pass 2" in qa.text("loop_label"), timeout_s=60)
        qa.stop_take()                  # at once: Pass 2 has scored nothing yet, however slow or fast the machine is (#69)
        assert len(qa.takes()) == 1 and qa.takes()[0]["pass_index"] == "1"
        text = qa.summary()
        assert text.startswith("Pass 2 of 1.0-9.0 s not saved: "), text
        assert "Last saved: Pass 1 of 1.0-9.0 s: In tune" in text, text
        assert qa.takes()[0]["id"] in text
        qa.click("replay_button")                                             # the replay target is Pass 1
        assert qa.summary() == f"Replaying Take {qa.takes()[0]['id']}..."
        qa.click("replay_button")


def test_stop_with_nothing_sung_says_nothing_was_saved_and_replay_has_nothing_to_replay(tmp_path):
    singing_song(tmp_path, notes=NOTES, tail_s=1.0)
    with QA(tmp_path, mic="echo", time_scale=SCALE) as qa:
        qa.set("reference_box", "Nearest note")                               # echo hears the silent output: no singing
        qa.start_take()
        qa.wait(1.0)
        qa.stop_take()
        assert qa.takes() == []
        assert qa.summary() == "No voiced sound heard. The Take is not saved."
        qa.click("replay_button")
        assert qa.summary() == "No saved Take to replay yet."


def test_the_stop_line_says_how_much_was_sung_in_a_dropped_pass_and_in_a_dropped_empty_one():
    from sing_pitch.ui.main_window import stop_text

    # the text is built from the Pass, the Section and the scored frames of the two records
    class S:
        def __init__(self, n):
            self.n_scored, self.pct_in_tune, self.median_abs_cents, self.tendency = n, 80.0, 5.0, "centred"

    class R:
        def __init__(self, n, pass_index, rid="r"):
            self.summary, self.pass_index, self.section_name, self.id = S(n), pass_index, "Chorus", rid
            self.tolerance_cents = 25.0

    assert stop_text(R(10, 1), R(120, 2)).startswith("Pass 2 of Chorus not saved: only 1.9 s sung.   |   Last saved: Pass 1 of Chorus")
    assert stop_text(R(10, 1), R(0, 2)).startswith("Pass 2 of Chorus not saved: nothing sung.")
    assert stop_text(None, R(0, None)) == "No voiced sound heard. The Take is not saved."

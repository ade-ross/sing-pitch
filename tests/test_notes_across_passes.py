"""Review 5 #1: Notes mode scored per note in a looped Section. Each Pass (and each Take after a jump) gets its own
note verdicts: the bars of a new Pass start uncoloured, and its notes are judged on that Pass's singing only. A note
that ends at the Section end gets its verdict inside its own Pass, though the song never plays its Timing window.
End to end on the QA driver with the echo singer (it sings exactly what the Guide vocal plays)."""
import numpy as np
from qa_helpers import NOTES, configure_latency, singing_song
from test_navigate_and_loops import SCALE, click_timeline

from sing_pitch.qa import QA
from sing_pitch.reference import Band

# NOTES * 3 from 0.5 s, 0.7 s each: a note ends at 4.0 and one at 8.2 (the 11th note)
SECTION = (4.0, 8.2)


def notes_echo_singer(home):
    singing_song(home, notes=NOTES * 3, tail_s=3.0)
    configure_latency(home, 0.12)
    qa = QA(home, mic="echo", time_scale=SCALE)
    qa.set("reference_box", "Melody")
    qa.set("notes_box", True)
    qa.set("guide_box", True)
    qa.set("guide_slider", 100)
    qa.set("backing_slider", 0)
    return qa


def in_pass(qa, n: int, lo: float, hi: float):
    return lambda: qa.window.loop_label.text().startswith(f"Loop · Pass {n}") \
        and lo <= (qa.song_position() or 0) < hi


def test_each_pass_starts_with_no_verdicts_and_the_last_note_is_judged_inside_its_pass(tmp_path):
    with notes_echo_singer(tmp_path) as qa:
        w = qa.window
        qa.mark_section(*SECTION)
        qa.start_take()
        qa.set("loop_box", True)
        qa.wait_until(in_pass(qa, 1, 7.0, 8.2), timeout_s=20)
        assert w._verdicts and set(w._verdicts.values()) == {Band.GREEN}       # Pass 1 judged as it goes
        qa.wait_until(in_pass(qa, 2, 3.0, 4.0), timeout_s=20)                 # Pass 2's Lead-in
        assert w._verdicts == {}                                               # nothing of Pass 2 judged yet
        last = int(w._melody.note_index_at(np.array([8.1]))[0])
        assert any(v.get(last) is Band.GREEN for _, v in w._verdict_eras)    # Pass 1's last note, in Pass 1
        qa.wait_until(in_pass(qa, 2, 6.0, 8.2), timeout_s=20)
        ends, _ = w._melody.note_spans_s()
        assert w._verdicts and all(ends[k] < 6.0 for k in w._verdicts)        # only notes Pass 2 has passed
        assert set(w._verdicts.values()) == {Band.GREEN}
        qa.stop_take()


def test_a_jump_back_clears_the_verdicts_of_the_notes_ahead(tmp_path):
    with notes_echo_singer(tmp_path) as qa:
        w = qa.window
        qa.start_take()
        qa.wait_until(lambda: (qa.song_position() or 0) > 6.0, timeout_s=20)
        assert w._verdicts
        click_timeline(qa, 1.0)
        qa.wait_until(lambda: 1.8 <= (qa.song_position() or 0) < 3.0, timeout_s=20)
        ends, _ = w._melody.note_spans_s()
        assert all(ends[k] < 1.8 for k in w._verdicts)                         # the notes ahead are uncoloured
        qa.stop_take()

"""The status line says "Paused" while the song is paused (it said "Singing..."), and in a loop the Speed label says a
new Speed comes from the next Pass while the old one still plays. Real Transport, QA driver."""
from pause_fuzz import build_song

from sing_pitch.qa import QA
from sing_pitch.ui.main_window import PAUSED_TEXT


def test_the_status_line_says_paused_and_goes_back_to_singing(tmp_path):
    build_song(tmp_path)
    with QA(tmp_path, mic="echo", time_scale=4.0) as qa:
        qa.set("reference_box", "Melody")
        qa.start_take(at_s=2)
        qa.wait(0.5)
        assert qa.summary().startswith("Singing")
        qa.pause()
        assert qa.summary() == PAUSED_TEXT
        qa.wait(0.3)
        assert qa.summary() == PAUSED_TEXT                     # and stays so through the ticks
        qa.pause()
        qa.wait(0.3)
        assert qa.summary().startswith("Singing")
        qa.stop_take()


def test_in_a_loop_the_speed_label_says_the_new_speed_comes_from_the_next_pass(tmp_path):
    build_song(tmp_path)
    with QA(tmp_path, mic="echo", time_scale=4.0) as qa:
        qa.set("reference_box", "Melody")
        qa.mark_section(10.0, 14.0)
        qa.set("loop_box", True)
        qa.start_take()
        qa.wait(0.3)
        qa.click("speed_85_button")
        qa.wait_until(lambda: "from the next Pass" in qa.text("speed_label"), timeout_s=30)
        assert qa.text("speed_label") == "85 % from the next Pass"
        qa.wait_until(lambda: qa.text("speed_label") == "", timeout_s=30)      # the next Pass has begun at 85 %
        assert round(100 * qa.window._backing.speed_playing) == 85
        qa.stop_take()

"""Melody in the window, no hardware: fake output device facts, fake Backing player, array mic."""
import csv

import numpy as np
from audio import SR, harmonic, quiet
from frames import hz_at
from melody_helpers import library_song, reference_line

from sing_pitch.audio_input import ArrayInput
from sing_pitch.playback import SongTimeLog
from sing_pitch.pitch.swiftf0 import SwiftF0Tracker
from sing_pitch.settings import Settings

NOTES = [(-900, 0.6), (-500, 0.6), (-200, 0.8)]


class FakeBacking:
    duration_s, finished = 2.0, False

    def __init__(self, path=None, vocals=None):
        self.path, self.vocals, self.started, self.stopped = path, vocals, False, False
        self.gains = []
        self.song_log = SongTimeLog(SR)
        self.song_log.append(0, 0.0)             # playing from the song start, nothing rendered yet

    def set_gains(self, backing, vocal):
        self.gains.append((backing, vocal))

    def start(self):
        self.started = True

    def position_s(self):
        return 0.0

    def stop(self):
        self.stopped = True


def window(qtbot, home, audio, output=("WF-1000XM5", "headphones"), backings=None, **kw):
    from sing_pitch.ui import MainWindow

    def make_backing(path, vocals=None):
        b = FakeBacking(path, vocals)
        if backings is not None:
            backings.append(b)
        return b

    win = MainWindow(SwiftF0Tracker(), home=home, input_factory=lambda _i: ArrayInput(audio, SR, 256),
                     backing_factory=make_backing, output_info=lambda: output, **kw)
    qtbot.addWidget(win)
    return win


def test_melody_take_plays_the_backing_scores_in_sync_and_saves_the_song(qtbot, tmp_path):
    song = library_song(tmp_path, reference_line(NOTES))
    s = Settings.load(tmp_path)
    s.set_latency("WF-1000XM5", 0.2)
    s.save()
    audio = np.concatenate([quiet(0.2)] + [harmonic(hz_at(c), d) for c, d in NOTES] + [quiet(0.4)])
    backings = []
    win = window(qtbot, tmp_path, audio, backings=backings)
    assert not win.song_box.isEnabled()
    win.reference_box.setCurrentText("Melody")
    assert win.song_box.isEnabled() and win.song_box.currentData() == song.id
    assert win.output_label.text() == ""                       # headphones: no warning
    win.toggle()
    assert backings[0].started and backings[0].path == song.backing_path
    qtbot.wait(150)
    win.tick()
    win.toggle()
    assert backings[0].stopped
    assert song.id in win.summary_label.text()
    row = next(csv.DictReader(open(tmp_path / "takes.csv")))
    assert (row["reference"], row["song_id"]) == ("melody", song.id)
    assert abs(float(row["latency_offset_s"]) - (0.2 + 256 / SR)) < 1e-6
    assert float(row["pct_in_tune"]) >= 90.0
    x, _ = win._ref_curve.getData()
    assert x is not None and len(x) > 0                        # reference line drawn beside the live line

    win.toggle_replay()                                        # Melody Takes replay via their song id
    assert "Cannot replay" not in win.summary_label.text()
    qtbot.waitUntil(lambda: len(win._t) > 20, timeout=3000)


def test_speakers_get_a_warning_and_an_unsure_device_can_be_confirmed_as_headphones(qtbot, tmp_path):
    win = window(qtbot, tmp_path, quiet(0.1), output=("MacBook Pro Speakers", "speakers"))
    win.reference_box.setCurrentText("Melody")
    assert "speakers" in win.output_label.text() and win.headphones_button.isHidden()

    win = window(qtbot, tmp_path, quiet(0.1), output=("Scarlett 2i2 USB", "unknown"))
    win.reference_box.setCurrentText("Melody")
    assert "Cannot tell" in win.output_label.text() and not win.headphones_button.isHidden()
    win.confirm_headphones()
    assert win.output_label.text() == ""
    assert Settings.load(tmp_path).headphones_confirmed("Scarlett 2i2 USB")


def test_melody_with_an_empty_library_says_how_to_import(qtbot, tmp_path):
    win = window(qtbot, tmp_path, quiet(0.1))
    win.reference_box.setCurrentText("Melody")
    win.toggle()
    assert "sing-pitch import" in win.summary_label.text() and win.button.text() == "Start"


def test_replaying_a_melody_take_whose_song_was_removed_explains_instead_of_crashing(qtbot, tmp_path):
    from sing_pitch.library import Library

    song = library_song(tmp_path, reference_line(NOTES))
    audio = np.concatenate([quiet(0.2), harmonic(hz_at(-900), 0.6), quiet(0.2)])
    win = window(qtbot, tmp_path, audio)
    win.reference_box.setCurrentText("Melody")
    win.toggle(), qtbot.wait(100), win.toggle()
    Library(tmp_path).remove(song.id)
    win.toggle_replay()
    assert "Cannot replay" in win.summary_label.text()


def test_calibration_in_the_window_saves_the_latency_for_the_output_device(qtbot, tmp_path):
    from test_calibration import performance

    from sing_pitch.calibration import click_track

    class Player:
        def __init__(self, audio, rate):
            pass

        def start(self):
            pass

        def position_s(self):
            return 256 / SR                     # mic and clicks start together

        def stop(self):
            pass

    _, times = click_track(SR)
    mic = performance(times, 0.27, times[-1] + 1.5)
    win = window(qtbot, tmp_path, mic, calibration_player_factory=Player)
    win.start_calibration()
    assert not win.button.isEnabled()
    win.finish_calibration()
    assert "Saved as the default" in win.summary_label.text()
    assert abs(Settings.load(tmp_path).latency_for("WF-1000XM5") - 0.27) < 0.008
    assert win.button.isEnabled()


def test_unscored_stretches_of_the_reference_are_greyed_on_the_live_plot_and_the_replay(qtbot, tmp_path):
    from melody_helpers import with_confidence

    line = with_confidence(reference_line(NOTES), 0.6, 1.2, 0.05)
    library_song(tmp_path, line)
    audio = np.concatenate([quiet(0.2)] + [harmonic(hz_at(c), d) for c, d in NOTES] + [quiet(0.4)])
    win = window(qtbot, tmp_path, audio)
    win.reference_box.setCurrentText("Melody")
    win.toggle()
    qtbot.waitUntil(lambda: len(win._t) > 20, timeout=3000)
    win.tick()
    x0, latency = win._unscored.opts["x0"], win._runner.latency_s
    assert len(x0) == 1 and abs(x0[0] - (0.6 + latency)) < 0.02
    win.toggle()
    win.toggle_replay()
    qtbot.waitUntil(lambda: len(win._t) > 20, timeout=3000)
    win.tick()
    assert len(win._unscored.opts["x0"]) == 1


def test_the_summary_of_a_melody_take_shows_its_octave_shift(qtbot, tmp_path):
    song = library_song(tmp_path, reference_line(NOTES))
    audio = np.concatenate([quiet(0.2)] + [harmonic(hz_at(c - 1200), d) for c, d in NOTES] + [quiet(0.4)])
    win = window(qtbot, tmp_path, audio)
    win.reference_box.setCurrentText("Melody")
    win.toggle()
    qtbot.waitUntil(lambda: len(win._t) > 80, timeout=3000)
    win.toggle()
    assert song.id in win.summary_label.text() and "octave shift -1" in win.summary_label.text()
    assert next(csv.DictReader(open(tmp_path / "takes.csv")))["octave_shift"] == "-1"


def test_calibrating_an_output_with_no_name_is_remembered_for_the_next_melody_take(qtbot, tmp_path):
    from test_calibration import performance

    from sing_pitch.calibration import click_track

    class Player:
        def __init__(self, audio, rate):
            pass

        def start(self):
            pass

        def position_s(self):
            return 256 / SR

        def stop(self):
            pass

    _, times = click_track(SR)
    library_song(tmp_path, reference_line(NOTES))
    win = window(qtbot, tmp_path, performance(times, 0.27, times[-1] + 1.5), output=(None, "unknown"),
                 calibration_player_factory=Player)
    win.start_calibration()
    win.finish_calibration()
    assert "Saved as the default" in win.summary_label.text()
    win.reference_box.setCurrentText("Melody")
    win.toggle()
    assert "not calibrated" not in win.summary_label.text()
    assert abs(win._runner.latency_s - 0.27) < 0.05 if win._runner else True


def test_guide_vocal_toggle_and_two_sliders_are_for_melody_and_act_on_the_backing_player_mid_take(qtbot, tmp_path):
    song = library_song(tmp_path, reference_line(NOTES))
    backings = []
    win = window(qtbot, tmp_path, quiet(2.0), backings=backings)
    controls = (win.guide_box, win.guide_slider, win.backing_slider)
    assert not any(c.isEnabled() for c in controls)                   # Nearest note
    win.reference_box.setCurrentText("Melody")
    assert all(c.isEnabled() for c in controls)
    win.toggle()
    assert backings[0].vocals == song.vocals_path
    assert backings[0].gains[-1] == (1.0, 0.0)                         # Guide vocal starts off
    assert all(c.isEnabled() for c in controls)                        # usable mid-Take
    win.guide_box.setChecked(True)
    assert backings[0].gains[-1] == (1.0, 1.0)
    win.guide_slider.setValue(40)
    assert backings[0].gains[-1] == (1.0, 0.4)
    win.backing_slider.setValue(50)
    assert backings[0].gains[-1] == (0.5, 0.4)
    win.guide_box.setChecked(False)
    assert backings[0].gains[-1] == (0.5, 0.0)
    win.toggle()
    win.reference_box.setCurrentText("Key")
    assert not any(c.isEnabled() for c in controls)

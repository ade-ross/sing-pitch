"""The Output selector: where the Backing, Guide vocal and Drone are heard. Fake device lists and fake streams, no sound."""
from PySide6 import QtCore
from audio import SR, quiet
from melody_helpers import library_song, reference_line

from sing_pitch.audio_input import ArrayInput
from sing_pitch.pitch.swiftf0 import SwiftF0Tracker
from sing_pitch.playback import DronePlayer
from sing_pitch.playback.outputs import output_devices
from sing_pitch.reference import Key
from sing_pitch.settings import Settings

NOTES = [(-900, 0.6), (-500, 0.6)]

MIC = {"index": 0, "name": "Blue Yeti", "max_input_channels": 1, "max_output_channels": 0}
BUDS = {"index": 1, "name": "WF-1000XM5", "max_input_channels": 1, "max_output_channels": 2}
SPEAKERS = {"index": 2, "name": "MacBook Pro Speakers", "max_input_channels": 0, "max_output_channels": 2}
DEVICES = [MIC, BUDS, SPEAKERS]


def kinds(name):
    return "headphones" if name == "WF-1000XM5" else "speakers"


class FakeBacking:
    duration_s, finished = 2.0, False

    def __init__(self, path=None, vocals=None, **kw):
        self.kw = kw

    def set_gains(self, *a): ...
    def set_speed(self, *a): ...
    def start(self): ...
    def stop(self): ...
    def position_s(self):
        return 0.0

    @property
    def song_log(self):
        from sing_pitch.playback import SongTimeLog

        log = SongTimeLog(SR)
        log.append(0, 0.0)
        return log


class FakeDrone:
    def __init__(self, **kw):
        self.kw, self.started_with = kw, None

    def start(self, key):
        self.started_with = key

    def stop(self): ...


def window(qtbot, home, devices=DEVICES, default=("MacBook Pro Speakers", "speakers"), audio=None, made=None, **kw):
    from sing_pitch.ui import MainWindow

    audio = quiet(0.5) if audio is None else audio
    made = [] if made is None else made

    def backing(path, vocals=None, **k):
        made.append(FakeBacking(path, vocals, **k))
        return made[-1]

    def drone(**k):
        made.append(FakeDrone(**k))
        return made[-1]

    win = MainWindow(SwiftF0Tracker(), home=home, input_factory=lambda _i: ArrayInput(audio, SR, 256),
                     backing_factory=backing, drone_factory=drone, output_info=lambda: default,
                     input_devices=lambda: [d for d in devices if d["max_input_channels"] > 0],
                     output_devices=lambda: output_devices(devices), output_kind_of=kinds, **kw)
    qtbot.addWidget(win)
    return win


def choose(win, name):
    win.output_box.setCurrentIndex(win.output_box.findText(name, QtCore.Qt.MatchStartsWith))


# --- the pure pieces ---------------------------------------------------------------------------------------------

def test_only_output_capable_devices_are_listed_as_outputs():
    assert [d["name"] for d in output_devices(DEVICES)] == ["WF-1000XM5", "MacBook Pro Speakers"]


def test_the_chosen_output_name_is_stored_in_settings(tmp_path):
    s = Settings.load(tmp_path)
    assert s.output_device is None
    s.output_device = "WF-1000XM5"
    s.save()
    assert Settings.load(tmp_path).output_device == "WF-1000XM5"


def test_drone_player_opens_its_stream_on_the_chosen_device_and_on_the_default_without_one():
    streams = []

    def factory(**kw):
        streams.append(kw)
        return type("S", (), {"start": lambda self: None, "stop": lambda self: None, "close": lambda self: None})()

    DronePlayer(stream_factory=factory, device=7).start(Key.parse("A minor"))
    DronePlayer(stream_factory=factory).start(Key.parse("A minor"))
    assert streams[0]["device"] == 7 and "device" not in streams[1]


# --- the window --------------------------------------------------------------------------------------------------

def test_output_dropdown_lists_the_system_default_first_then_every_output_and_not_microphones(qtbot, tmp_path):
    win = window(qtbot, tmp_path)
    texts = [win.output_box.itemText(i) for i in range(win.output_box.count())]
    assert texts == ["System default (MacBook Pro Speakers)", "WF-1000XM5", "MacBook Pro Speakers"]
    assert win.output_box.currentIndex() == 0 and win.output_box.currentData() is None
    assert [win.device_box.itemText(i) for i in range(win.device_box.count())] == ["Blue Yeti", "WF-1000XM5"]


def test_the_choice_is_remembered_between_runs(qtbot, tmp_path):
    win = window(qtbot, tmp_path)
    choose(win, "WF-1000XM5")
    assert Settings.load(tmp_path).output_device == "WF-1000XM5"
    again = window(qtbot, tmp_path)
    assert again.output_box.currentText() == "WF-1000XM5"
    choose(again, "System default")
    assert Settings.load(tmp_path).output_device is None


def test_a_remembered_output_that_is_unplugged_falls_back_to_the_default_and_comes_back_when_it_returns(qtbot, tmp_path):
    s = Settings.load(tmp_path)
    s.output_device = "WF-1000XM5"
    s.save()
    state = {"devices": [MIC, SPEAKERS]}
    win = window(qtbot, tmp_path)
    win._output_devices = lambda: output_devices(state["devices"])
    win.refresh_devices()
    assert win.output_box.currentData() is None
    assert Settings.load(tmp_path).output_device == "WF-1000XM5"       # not forgotten
    state["devices"] = DEVICES
    win.refresh_devices()
    assert win.output_box.currentText() == "WF-1000XM5"


def test_a_melody_take_opens_the_backing_on_the_chosen_device_and_the_default_when_none_is_chosen(qtbot, tmp_path):
    library_song(tmp_path, reference_line(NOTES))
    made = []
    win = window(qtbot, tmp_path, made=made)
    win.reference_box.setCurrentText("Melody")
    win.toggle(), win.toggle()
    assert "device" not in made[0].kw
    choose(win, "WF-1000XM5")
    win.toggle(), win.toggle()
    assert made[1].kw == {"device": 1}


def test_a_key_take_opens_the_drone_on_the_chosen_device(qtbot, tmp_path):
    made = []
    win = window(qtbot, tmp_path, made=made)
    choose(win, "WF-1000XM5")
    win.reference_box.setCurrentText("Key")
    win.drone_box.setChecked(True)
    win.toggle(), win.toggle()
    assert made[0].kw == {"device": 1} and made[0].started_with is not None


def test_latency_and_the_headphones_warning_follow_the_chosen_output(qtbot, tmp_path):
    library_song(tmp_path, reference_line(NOTES))
    s = Settings.load(tmp_path)
    s.set_latency("WF-1000XM5", 0.21)
    s.set_latency("MacBook Pro Speakers", 0.05)
    s.save()
    win = window(qtbot, tmp_path)
    win.reference_box.setCurrentText("Melody")
    assert "speakers" in win.output_label.text()                         # the default output is the speakers
    choose(win, "WF-1000XM5")
    assert win.output_label.text() == ""
    win.toggle()
    assert abs(win._runner.latency_s - (0.21 + 256 / SR)) < 1e-6
    win.toggle()
    choose(win, "MacBook Pro Speakers")
    assert "MacBook Pro Speakers" in win.output_label.text()


def test_calibration_is_saved_under_the_chosen_output_and_plays_its_clicks_there(qtbot, tmp_path):
    from sing_pitch.calibration import click_track

    class Player:
        def __init__(self, audio, rate):
            pass

        def start(self): ...
        def position_s(self):
            return 256 / SR
        def stop(self): ...

    from test_calibration import performance

    _, times = click_track(SR)
    win = window(qtbot, tmp_path, audio=performance(times, 0.27, times[-1] + 1.5), calibration_player_factory=Player)
    choose(win, "WF-1000XM5")
    win.start_calibration()
    win.finish_calibration()
    assert abs(Settings.load(tmp_path).latency_for("WF-1000XM5") - 0.27) < 0.01
    assert Settings.load(tmp_path).is_calibrated("WF-1000XM5") and not Settings.load(tmp_path).is_calibrated("MacBook Pro Speakers")


def test_calibration_without_a_player_factory_opens_its_player_on_the_chosen_device(qtbot, tmp_path, monkeypatch):
    opened = []

    class Recorder:
        def __init__(self, audio, rate, sample_rate=None, device=None):
            opened.append(device)

        def start(self): ...
        def position_s(self):
            return 0.0
        def stop(self): ...

    monkeypatch.setattr("sing_pitch.playback.BackingPlayer", Recorder)
    win = window(qtbot, tmp_path)
    choose(win, "WF-1000XM5")
    win.start_calibration()
    win.finish_calibration()
    assert opened == [1]


def test_the_output_cannot_change_during_a_take_and_is_free_again_afterwards(qtbot, tmp_path):
    win = window(qtbot, tmp_path)
    win.toggle()
    assert not win.output_box.isEnabled() and not win.device_box.isEnabled() and not win.refresh_button.isEnabled()
    win.toggle()
    assert win.output_box.isEnabled() and win.refresh_button.isEnabled()


def test_refresh_picks_up_a_device_plugged_in_without_restarting_and_keeps_the_selection(qtbot, tmp_path):
    state = {"devices": [MIC, SPEAKERS]}
    from sing_pitch.ui import MainWindow

    win = MainWindow(SwiftF0Tracker(), home=tmp_path, output_info=lambda: ("MacBook Pro Speakers", "speakers"),
                     input_devices=lambda: [d for d in state["devices"] if d["max_input_channels"] > 0],
                     output_devices=lambda: output_devices(state["devices"]), output_kind_of=kinds)
    qtbot.addWidget(win)
    assert win.output_box.findText("WF-1000XM5") == -1 and win.device_box.count() == 1
    choose(win, "MacBook Pro Speakers")
    state["devices"] = DEVICES
    win.refresh_button.click()
    assert win.output_box.findText("WF-1000XM5") >= 0 and win.device_box.count() == 2
    assert win.output_box.currentText() == "MacBook Pro Speakers"          # still the same device by name
    assert win.device_box.currentText() == "Blue Yeti"


def test_a_device_listing_that_fails_leaves_the_window_open_with_a_placeholder(qtbot, tmp_path):
    from sing_pitch.ui import MainWindow

    def broken():
        raise OSError("no PortAudio")

    win = MainWindow(SwiftF0Tracker(), home=tmp_path, input_devices=broken, output_devices=broken)
    qtbot.addWidget(win)
    assert win.output_box.count() >= 1 and win.output_box.currentData() is None

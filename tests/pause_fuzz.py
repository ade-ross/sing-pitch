"""Random-action hunt for pause bugs (#50), through the QA driver: the real window, the fake mic and the fake output.

A run is a seeded list of user actions (Start, Space, the Pause button, seek, arrows, Section, Loop, Lead-in/Count-in,
Speed, Guide vocal, resize, replay, Stop ...). After every action the invariants are checked:

- at most one output stream is open at any time, none when no Take (or calibration) runs;
- the Pause button, the window's `_paused` and the Transport's state agree with what the user did last;
- while paused, after a short settle: the recorded output is silent, and the song position, the Timeline playhead, the
  plot (now-line, x range) and the count of scored frames do not move, even though the mic keeps singing;
- after a resume the song goes on from where it stood and everything moves again.

run(seed, steps) returns the violations with the action list that led to them. Used by tests/test_pause_random.py (a
short, fixed set of seeds) and by hand: `uv run python tests/pause_fuzz.py <first seed> <count> <steps>`."""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from qa_helpers import NOTES, configure_latency, singing_song

from sing_pitch.qa import QA

SCALE = 4.0
SETTLE_S = 0.9         # app seconds after a pause before everything must stand still (tracker look-ahead + latency)
WATCH_S = 0.6          # ... and the stretch it is watched over
LONGEST_WAIT_S = 1.0   # longest wait between two actions


def build_song(home: Path, backing_hz: float = 220.0):
    """A ~55 s song: vocals the mic sings from, a reference line, and an AUDIBLE backing (a quiet tone) so that a
    sound that goes on while paused is heard by the recording."""
    song = singing_song(home, notes=NOTES * 8, tail_s=20.0, title="Pause Song")
    vocals, sr = sf.read(str(song.vocals_path), dtype="float32", always_2d=True)
    t = np.arange(len(vocals)) / sr
    tone = (0.2 * np.sin(2 * np.pi * backing_hz * t)).astype(np.float32)
    sf.write(str(song.backing_path), np.stack([tone, tone], axis=1), sr)
    configure_latency(home, 0.12)
    return song


class _Probe:
    """What the hunt records, from patches on the classes (not on the instances: a closure on the window, the hub or the
    runner would make a reference cycle, and a Qt object in a cycle is freed by gc on any thread, see handoff-segfault)."""

    max_streams = 0                  # most output streams the hub played in one step
    scored: dict[int, list[float]] = {}   # per runner (its id): voice time of every scored frame (with a Band) it handed over


def _install_probe():
    from sing_pitch.live import LiveRunner
    from sing_pitch.testmode import FakeAudioHub

    step, drain = FakeAudioHub._step, LiveRunner.drain

    def counting_step(self) -> None:                 # hub thread
        _Probe.max_streams = max(_Probe.max_streams, len(self._outputs))
        step(self)

    def recording_drain(self) -> list:
        out = drain(self)
        _Probe.scored.setdefault(id(self), []).extend(s.t for s in out if s.band is not None)
        return out

    FakeAudioHub._step, LiveRunner.drain = counting_step, recording_drain
    _Probe.max_streams, _Probe.scored = 0, {}

    def restore() -> None:
        FakeAudioHub._step, LiveRunner.drain = step, drain

    return restore


class Hunt:
    def __init__(self, home: Path, seed: int):
        self._restore = _install_probe()
        self.rng = random.Random(seed)
        try:
            self.song = build_song(home)
            self.qa = QA(home, mic=str(self.song.vocals_path), loop_mic=True, time_scale=SCALE,
                         record_output=home / "out.wav")
        except BaseException:
            self._restore()
            raise
        self.win, self.hub = self.qa.window, self.qa.hub
        self.actions: list[str] = []
        self.violations: list[str] = []
        self.expect_paused = False
        self.backing_seen = None
        self.pause_pos: float | None = None       # song position seen while paused, until it is resumed or moved
        self.sections_saved = 0
        self.frozen_checks = 0
        self.threads_before = self._tracker_threads()
        self.visible_before = self._visible_windows()
        self.set("reference_box", "Melody")
        self.set("guide_box", True)
        self.set("guide_slider", 60)
        self.set("backing_slider", 60)

    @staticmethod
    def _tracker_threads() -> int:
        import threading

        return sum(t.name == "sing-pitch-tracker" and t.is_alive() for t in threading.enumerate())

    @staticmethod
    def _visible_windows() -> set:
        from PySide6 import QtWidgets

        return {id(w) for w in QtWidgets.QApplication.topLevelWidgets() if w.isVisible()}

    # --- instrumentation: see _Probe ----------------------------------------------------------------------------
    @property
    def max_streams(self) -> int:
        return _Probe.max_streams

    @max_streams.setter
    def max_streams(self, value: int) -> None:
        _Probe.max_streams = value

    @property
    def scored_t(self) -> list[float]:
        """Voice times of the scored frames of the running Take's runner."""
        return _Probe.scored.get(id(self.win._runner), [])

    def set(self, name: str, value) -> bool:
        """A change of a control, as the user makes it; nothing when the window has greyed the control out."""
        if not self.qa.widget(name).isEnabled():
            return False
        self.qa.set(name, value)
        return True

    # --- bookkeeping ------------------------------------------------------------------------------------------
    def say(self, text: str) -> None:
        self.actions.append(f"{len(self.actions):02d} t={self.hub.now_s:7.2f} {text}")

    def bad(self, text: str) -> None:
        self.violations.append(f"[after #{len(self.actions) - 1}] {text}")

    @property
    def backing(self):
        return self.win._backing

    @property
    def subject(self):
        """What Space can pause: the running Take's Backing, or the replay on show."""
        return self.win._backing or self.win._replay

    def _sync_expectation(self) -> None:
        """A new Backing (a new Start) plays from the start; none = nothing to be paused."""
        if self.subject is not self.backing_seen:
            self.backing_seen = self.subject
            self.expect_paused = False
            self.pause_pos = None

    def _flip(self) -> None:
        self.expect_paused = not self.expect_paused
        self.pause_pos = None

    # --- actions (what a user does) -----------------------------------------------------------------------------
    def a_start(self) -> None:
        if self.qa.running or self.win._replay is not None or not self.win.button.isEnabled():
            return
        self.say("Start")
        self.qa.click("button")
        if self.win._start_waiting:
            self.qa.wait_until(lambda: self.qa.running or not self.win._start_waiting, timeout_s=30)

    def a_stop(self) -> None:
        if self.qa.running:
            self.say("Stop")
            self.qa.click("button")

    def a_space(self) -> None:
        self.say("Space")
        was = self.subject
        self.qa.press("Space")
        if was is not None:
            self._flip()

    def a_space_burst(self) -> None:
        """Several Space presses with no time between them, as a held key repeats."""
        n = self.rng.randint(2, 5)
        self.say(f"Space x{n} at once")
        for _ in range(n):
            was = self.subject
            self.qa.press("Space")
            if was is not None:
                self._flip()

    def a_space_to_focus(self) -> None:
        """Space typed with a control focused, as a real keyboard delivers it (the QA press goes to the window)."""
        from PySide6 import QtCore, QtWidgets
        from PySide6.QtTest import QTest

        names = ["pause_button", "button", "bpm_box", "guide_box", "guide_slider", "reference_box", "song_box", "drone_box",
                 "calibrate_button", "refresh_button", "replay_button", "history_button", "device_box", "output_box"]
        name = self.rng.choice(names)
        widget = self.qa.widget(name)
        if not widget.isEnabled():
            return
        self.say(f"Space with focus on {name}")
        widget.setFocus(QtCore.Qt.TabFocusReason)
        QtWidgets.QApplication.processEvents()
        was, running, replay = self.subject, self.qa.running, self.win._replay is not None
        QTest.keyClick(QtWidgets.QApplication.focusWidget() or widget, QtCore.Qt.Key_Space)
        QtWidgets.QApplication.processEvents()
        if running != self.qa.running or replay != (self.win._replay is not None):
            return                               # a Start / Stop / replay button was pressed too: another question
        if was is not None and self.subject is was:
            self._flip()

    def a_pause_button(self) -> None:
        if self.win.pause_button.isEnabled():
            self.say("Pause button")
            self.qa.click("pause_button")
            self._flip()

    def a_seek(self) -> None:
        if self.backing is None:
            return
        to = round(self.rng.uniform(0.0, self.backing.duration_s - 6.0), 2)
        self.say(f"seek {to}")
        self.qa.seek(to)
        self.pause_pos = None

    def a_arrow(self) -> None:
        if self.backing is not None:
            key = self.rng.choice(["Left", "Right"])
            self.say(f"arrow {key}")
            self.qa.press(key)
            self.pause_pos = None

    def a_mark(self) -> None:
        if self.win._speed_song() is None:
            return
        start = round(self.rng.uniform(2.0, 30.0), 1)
        end = round(start + self.rng.uniform(1.0, 6.0), 1)
        self.say(f"mark Section {start}-{end}")
        self.qa.mark_section(start, end)
        self.pause_pos = None

    def a_save_section(self) -> None:
        if not self.win.save_section_button.isEnabled():
            return
        self.sections_saved += 1
        self.say(f"save Section s{self.sections_saved}")
        self.set("section_name_edit", f"s{self.sections_saved}")
        self.qa.click("save_section_button")

    def a_choose_section(self) -> None:
        box = self.win.section_box
        i = self.rng.randrange(box.count())
        self.say(f"choose Section #{i}")
        self.set("section_box", box.itemText(i))
        self.pause_pos = None

    def a_loop(self) -> None:
        if not self.win.loop_box.isEnabled():
            return
        if self.rng.random() < 0.5:
            self.say("press L")
            self.qa.press("L")
        else:
            on = not self.win.loop_box.isChecked()
            self.say(f"loop box {on}")
            self.set("loop_box", on)
        self.pause_pos = None

    def a_between(self) -> None:
        kind = self.rng.choice(["Lead-in", "Count-in"])
        self.say(f"between {kind}")
        if self.win.bpm_box.value() == 0:
            self.set("bpm_box", 96.0)
        self.set("between_box", kind)

    def a_speed_preset(self) -> None:
        name = self.rng.choice(["speed_70_button", "speed_85_button", "speed_100_button"])
        if self.win.speed_box.isEnabled():
            self.say(f"Speed preset {name}")
            self.qa.click(name)

    def a_speed_key(self) -> None:
        key = self.rng.choice(["+", "-"])
        self.say(f"Speed key {key}")
        self.qa.press(key)

    def a_guide(self) -> None:
        on = not self.win.guide_box.isChecked()
        self.say(f"Guide vocal {on}")
        self.set("guide_box", on)

    def a_resize(self) -> None:
        w, h = self.rng.randrange(760, 1500), self.rng.randrange(560, 900)
        self.say(f"resize {w}x{h}")
        self.win.resize(w, h)
        self.qa._app.processEvents()

    def a_replay(self) -> None:
        if self.qa.running:
            return
        if self.win._replay is not None:
            self.say("Stop replay")
            self.qa.click("replay_button")
        elif self.win.replay_button.isEnabled() and self.qa.takes():
            self.say("Replay last Take")
            self.qa.click("replay_button")

    def a_replay_from_history(self) -> None:
        takes = self.qa.takes()
        if self.qa.running or not takes:
            return
        take_id = self.rng.choice(takes)["id"]
        self.say(f"replay_take {take_id}")
        self.win.replay_take(take_id)

    def a_history(self) -> None:
        self.say("open History, then click back into the main window")
        self.win.open_history()
        self.win.activateWindow()         # Space belongs to the window that is active: he clicks back to use it
        self.qa._app.processEvents()

    def a_refresh(self) -> None:
        self.say("Refresh devices")
        self.win.refresh_devices()

    def a_calibrate(self) -> None:
        if self.win.calibrate_button.isEnabled() and not self.qa.running:
            self.say("Calibrate")
            self.qa.click("calibrate_button")

    ACTIONS = [(8, "a_start"), (3, "a_stop"), (10, "a_space"), (2, "a_space_burst"), (4, "a_space_to_focus"), (4, "a_pause_button"),
               (6, "a_seek"), (3, "a_arrow"), (4, "a_mark"), (2, "a_save_section"), (3, "a_choose_section"),
               (5, "a_loop"), (3, "a_between"), (4, "a_speed_preset"), (3, "a_speed_key"), (3, "a_guide"),
               (4, "a_resize"), (3, "a_replay"), (2, "a_replay_from_history"), (1, "a_refresh"), (1, "a_calibrate"),
               (1, "a_history")]

    # --- invariants -------------------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        w, backing = self.win, self.backing
        return dict(
            frame=self.hub._frame,
            position=None if backing is None else backing.position_s(),
            playhead=w.timeline.playhead_s,
            now=w._now_line.value(),
            x_range=tuple(w._plot.viewRange()[0]),
            scored_n=len(self.scored_t),
            drawn_n=len(w._t),
        )

    def paused_since(self) -> float | None:
        """Voice time the song has been paused since (the open paused span of the draw clock), None if not paused."""
        spans = getattr(self.win._draw_clock(), "paused_spans", lambda: [])()
        return spans[-1][0] if spans and np.isinf(spans[-1][1]) else None

    def loudest(self, f0: int, f1: int) -> float:
        step = self.hub.step
        chunks = self.hub._recording[f0 // step + 1: f1 // step]
        return float(max((np.abs(c).max() for c in chunks), default=0.0))

    def check_state(self) -> None:
        w, subject = self.win, self.subject
        if self.max_streams > 1:
            self.bad(f"{self.max_streams} output streams were open at once")
        self.max_streams = len(self.hub._outputs)
        open_now = len(self.hub._outputs)
        if self.backing is None and w._calibration is None and open_now:
            self.bad(f"{open_now} output stream(s) open with no Take running")
        if self.backing is not None:
            if open_now > 1:
                self.bad(f"{open_now} output streams open now")
            transport_paused = not self.backing.transport.playing
            shown_paused = w.pause_button.text() == "Play"
            if len({w._paused, shown_paused, transport_paused}) != 1:
                self.bad(f"disagree: _paused={w._paused} button={w.pause_button.text()!r} "
                         f"transport paused={transport_paused}")
        elif subject is not None:
            if (w.pause_button.text() == "Play") != w._paused:
                self.bad(f"the replay: _paused={w._paused} but the button says {w.pause_button.text()!r}")
        if subject is not None and w._paused != self.expect_paused:
            self.bad(f"the user paused={self.expect_paused} but _paused={w._paused}")
        if subject is None and w.pause_button.isEnabled():
            self.bad("the Pause button is enabled with nothing to pause")
        if subject is not None and not w.pause_button.isEnabled():
            self.bad("the Pause button is greyed out while something plays")

    def settle(self, subject) -> None:
        """Give what lags the pause (the mic's frames, the tracker, the timer) time to arrive: until the plot has stood
        still for a third of a second, at most SETTLE_S. A plot that never stands still is then caught by the watch."""
        last, still_since = None, self.hub.now_s
        while self.hub.now_s - still_since < 0.35 and self.hub.now_s < still_since + SETTLE_S + 0.35 and \
                self.subject is subject and self.expect_paused:
            self.qa.wait(0.05)
            now = (self.win._now_line.value(), tuple(self.win._plot.viewRange()[0]), len(self.win._t))
            if now != last:
                last, still_since = now, self.hub.now_s
            if self.hub.now_s - still_since >= SETTLE_S + 0.35:
                break

    def check_frozen(self) -> None:
        """Paused: settle, then watch; nothing may move and nothing may be heard."""
        subject, backing = self.subject, self.backing
        if subject is None or not self.expect_paused:
            return
        self.settle(subject)
        if self.subject is not subject or not self.expect_paused:
            return
        self.frozen_checks += 1
        a = self.snapshot()
        self.qa.wait(WATCH_S)
        b = self.snapshot()
        if self.subject is not subject:
            return
        keys = ["playhead", "now", "x_range"]
        if backing is not None:
            keys.append("position")
            loud = self.loudest(a["frame"], b["frame"])
            if loud > 1e-6:
                self.bad(f"SOUND while paused: peak {loud:.3f} over app time "
                         f"{a['frame'] / 48000:.2f}-{b['frame'] / 48000:.2f}")
        else:
            keys.append("drawn_n")        # a replay draws nothing new while paused
        for key in keys:
            if not np.allclose(np.asarray(a[key], float), np.asarray(b[key], float), atol=1e-6):
                self.bad(f"{key} moved while paused: {a[key]} -> {b[key]}")
        if backing is not None:
            since = self.paused_since()
            if since is None:
                self.bad("the song is paused but the draw clock has no open paused span")
            else:
                late_scored = [t for t in self.scored_t if t >= since]
                late_drawn = [t for t in self.win._t if t >= since]
                if late_scored:
                    self.bad(f"{len(late_scored)} frames sung while paused (from voice time {since:.2f}) were scored")
                if late_drawn:
                    self.bad(f"{len(late_drawn)} frames sung while paused (from voice time {since:.2f}) are drawn")
        self.pause_pos = a["position"]

    def check_moving(self) -> None:
        """Not paused with a song playing: the song, the Timeline and the plot go on (unless it has ended)."""
        subject, backing = self.subject, self.backing
        if subject is None or self.expect_paused or getattr(subject, "finished", False):
            return
        before = self.snapshot()
        self.qa.wait(0.5)
        after = self.snapshot()
        if self.subject is not subject or self.expect_paused or getattr(subject, "finished", False):
            return
        if backing is not None:
            if self.pause_pos is not None and before["position"] < self.pause_pos - 0.01 and backing.loop is None:
                self.bad(f"after the resume the song is at {before['position']:.2f}, it was paused at "
                         f"{self.pause_pos:.2f}")
            looping = backing.loop is not None
        else:
            looping = False
        self.pause_pos = None
        if not looping:
            if backing is not None and after["position"] <= before["position"] + 0.1:
                self.bad(f"the song does not go on: {before['position']:.2f} -> {after['position']:.2f}")
            if abs(after["playhead"] - before["playhead"]) < 0.1 and self.win._replay is None:
                self.bad(f"the Timeline playhead does not move: {before['playhead']:.2f} -> {after['playhead']:.2f}")
            if after["now"] == before["now"] and self.win._replay is None:
                self.bad("the plot's now-line does not move")

    def step(self) -> None:
        weights, names = zip(*self.ACTIONS)
        getattr(self, self.rng.choices(names, weights)[0])()
        self.qa.wait(self.rng.uniform(0.05, LONGEST_WAIT_S))
        self._sync_expectation()
        self.check_state()
        if self.rng.random() < 0.7:
            self.check_frozen()
            self.check_moving()
            self.check_state()

    def check_close(self) -> None:
        """The red close button, in whatever state the window is: everything stops, nothing keeps the app alive."""
        self.say("close the window")
        self.qa.window.close()
        self.qa._app.processEvents()
        self.qa.wait(0.5)
        if self.win._runner is not None or self.win._replay is not None or self.win._calibration is not None:
            self.bad("after close a Take, replay or calibration is still going")
        if self.hub._outputs:
            self.bad(f"after close {len(self.hub._outputs)} output stream(s) are still open")
        if self._tracker_threads() > self.threads_before:
            self.bad("after close a tracker worker thread is still alive")
        if self._visible_windows() - self.visible_before:
            self.bad("after close a window is still visible, the app would not quit")

    def run(self, steps: int) -> None:
        try:
            for _ in range(steps):
                self.step()
            self.check_close()
        finally:
            self.qa.close()
            self._restore()
            self.win = self.hub = self.qa = self.song = None      # nothing of the window outlives the run


def run(seed: int, steps: int = 30, home: Path | None = None) -> tuple[list[str], list[str], int]:
    """(violations, the actions that led to them, how many pauses were watched)."""
    home = home or Path(tempfile.mkdtemp(prefix=f"pausefuzz{seed}-"))
    hunt = Hunt(home, seed)
    try:
        hunt.run(steps)
    except Exception as exc:                # a crash is a finding too
        hunt.violations.append(f"EXCEPTION {type(exc).__name__}: {exc}")
    return hunt.violations, hunt.actions, hunt.frozen_checks


if __name__ == "__main__":
    first, count, steps = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    failed = 0
    for seed in range(first, first + count):
        violations, actions, watched = run(seed, steps)
        print(f"seed {seed}: {len(violations)} violation(s), {watched} pauses watched", flush=True)
        if violations:
            failed += 1
            print("\n".join(violations))
            print("--- actions\n" + "\n".join(actions) + "\n---", flush=True)
    print(f"{failed} of {count} seeds failed")
    sys.exit(1 if failed else 0)

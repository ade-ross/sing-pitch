"""QA driver: run the real MainWindow offscreen on the fake mic and output (testmode.py) and drive it from a script.

    from sing_pitch.qa import QA
    with QA(home, mic="path/to/vocals.wav") as qa:      # or mic="echo"; home = a scratch SING_PITCH_HOME
        qa.set("reference_box", "Melody")
        qa.start_take(at_s=30)                        # Start, the song from 0:30, the file mic from 0:30 too
        qa.wait(60)                                   # 60 s of app time
        qa.stop_take()
        print(qa.summary(), qa.takes()[-1]["pct_in_tune"])

Widgets are addressed by objectName, which is the window's attribute name ("button" = Start/Stop, "pause_button",
"reference_box", "song_box", "loop_box", "guide_slider", ...; see names()). Times are app time: the fake audio
clock, which runs at time_scale x the wall clock."""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Callable

from .testmode import ECHO, FakeMicFactory, Flags, build

DEFAULT_TIMEOUT_S = 30.0


def _app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")     # before the first QApplication
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class QA:
    """The window on fakes. mic: a wav path (a singer of that file, see testmode.FakeMic), "echo" (hears the
    output mix) or None (the real mic: not for agents). latency_s: how late the mic hears the output; None = the
    latency the window uses for the fake output (settings.json of `home`, else its default), so a file mic is a
    perfect singer. record_output: a wav path written on close() with everything played."""

    def __init__(self, home: Path, mic: str | Path | None = ECHO, latency_s: float | None = None,
                 mic_offset_s: float = 0.0, loop_mic: bool = False, record_output: str | Path | None = None,
                 time_scale: float = 1.0, tracker=None, size: tuple[int, int] = (1200, 700)):
        self._app = _app()
        from .pitch.swiftf0 import SwiftF0Tracker
        from .ui import MainWindow

        self.home = Path(home)
        flags = Flags(fake_mic=None if mic is None else str(mic), fake_mic_offset=mic_offset_s, fake_mic_loop=loop_mic,
                      fake_latency=latency_s, fake_output=True,
                      record_output=None if record_output is None else str(record_output))
        self._record_output = record_output
        self.hub, self.mic, kwargs = build(flags, self.home, time_scale)
        self.window = MainWindow(tracker or SwiftF0Tracker(), home=self.home, **kwargs)
        self.window.resize(*size)
        self.window.show()
        self._app.processEvents()

    # --- lifetime ---------------------------------------------------------------------------------------------------
    def close(self) -> None:
        """Stop a running Take (it is saved), close the window, stop the fake audio, write the recording."""
        if self.window is None:
            return
        if self.running:
            self.stop_take()
        self.window.close()
        self.window.deleteLater()
        self.window = None
        self._app.processEvents()
        self.hub.close()
        if self._record_output is not None:
            self.hub.write_recording(Path(self._record_output))

    def __enter__(self) -> QA:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- widgets ------------------------------------------------------------------------------------------------------
    def widget(self, name: str):
        from PySide6 import QtWidgets

        found = self.window.findChild(QtWidgets.QWidget, name)
        if found is None:
            raise KeyError(f"no widget named {name!r}; known: {', '.join(self.names())}")
        return found

    def names(self) -> list[str]:
        from PySide6 import QtWidgets

        return sorted(w.objectName() for w in self.window.findChildren(QtWidgets.QWidget) if w.objectName()
                      and not w.objectName().startswith("qt_"))

    def click(self, name: str) -> None:
        """A left click in the middle of the widget, as a mouse would (a disabled widget ignores it)."""
        from PySide6 import QtCore
        from PySide6.QtTest import QTest

        QTest.mouseClick(self.widget(name), QtCore.Qt.LeftButton)
        self._app.processEvents()

    def set(self, name: str, value) -> None:
        """Set a combo box (by its text), check box, slider, spin box or line edit, as a user change would."""
        from PySide6 import QtWidgets

        w = self.widget(name)
        if not w.isEnabled():
            raise RuntimeError(f"{name} is disabled")
        if isinstance(w, QtWidgets.QComboBox):
            at = w.findText(str(value))
            if at < 0:
                raise ValueError(f"{name} has no item {value!r}: {[w.itemText(i) for i in range(w.count())]}")
            w.setCurrentIndex(at)
        elif isinstance(w, QtWidgets.QCheckBox):
            w.setChecked(bool(value))
        elif isinstance(w, (QtWidgets.QSlider, QtWidgets.QSpinBox)):
            w.setValue(int(value))
        elif isinstance(w, QtWidgets.QLineEdit):
            w.setText(str(value))
        else:
            raise TypeError(f"cannot set a {type(w).__name__}")
        self._app.processEvents()

    def press(self, key: str) -> None:
        """A key on the window: "Space", "Left", "Right", "L", "+", "-", ... (Qt.Key_<name>, or the character)."""
        from PySide6 import QtCore
        from PySide6.QtTest import QTest

        if len(key) == 1 and not key.isalnum():
            QTest.keyClicks(self.window, key)
        else:
            QTest.keyClick(self.window, getattr(QtCore.Qt, f"Key_{key}"))
        self._app.processEvents()

    def text(self, name: str) -> str:
        return self.widget(name).text()

    # --- time ---------------------------------------------------------------------------------------------------------
    def app_time(self) -> float:
        return self.hub.now_s

    def wait(self, seconds: float) -> None:
        """Let `seconds` of app time pass, running the window's event loop meanwhile."""
        until = self.hub.now_s + seconds
        self.wait_until(lambda: self.hub.now_s >= until, timeout_s=seconds + DEFAULT_TIMEOUT_S)

    def wait_until(self, condition: Callable[[], bool], timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        """Run the event loop until condition() is true; TimeoutError after timeout_s of app time."""
        until = self.hub.now_s + timeout_s
        while not condition():
            if self.hub.now_s > until:
                raise TimeoutError(f"still waiting after {timeout_s} s of app time")
            self._app.processEvents()
            time.sleep(0.002)
        self._app.processEvents()

    # --- Takes ----------------------------------------------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self.window._runner is not None

    def start_take(self, at_s: float | None = None) -> None:
        """Click Start with the clock stopped; with at_s, seek the song there before its first block plays and
        start the file mic at the same place (a perfect singer from there on)."""
        if isinstance(self.mic, FakeMicFactory) and not isinstance(self.mic.source, str):
            self.mic.offset_s = 0.0 if at_s is None else float(at_s)
        with self.hub.held():
            self.click("button")
            if not self.running:
                raise RuntimeError(f"the Take did not start: {self.summary()}")
            if at_s is not None:
                self.window.seek(float(at_s))
        self._app.processEvents()

    def stop_take(self) -> None:
        if self.running:
            self.click("button")

    def pause(self) -> None:
        """Space: pause or resume the song."""
        self.press("Space")

    def seek(self, song_s: float) -> None:
        """What a click on the Timeline at song_s does."""
        self.window.timeline.seek_requested.emit(float(song_s))
        self._app.processEvents()

    def mark_section(self, start_s: float, end_s: float) -> None:
        """What a drag across the Timeline from start_s to end_s does: the unsaved mark."""
        self.window.timeline.section_marked.emit(float(start_s), float(end_s))
        self._app.processEvents()

    # --- state ----------------------------------------------------------------------------------------------------------
    def song_position(self) -> float | None:
        """Song time of the Backing (None without a Melody Take running)."""
        backing = self.window._backing
        return None if backing is None else backing.position_s()

    def summary(self) -> str:
        return self.window.summary_label.text()

    def takes(self) -> list[dict]:
        """Rows of takes.csv, oldest first."""
        path = self.home / "takes.csv"
        if not path.exists():
            return []
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    def plot_data(self) -> dict:
        """What the plot shows now: voice line (t, midi), reference line (x, midi), the now-line, the x range."""
        import numpy as np

        w = self.window

        def xy(curve):
            x, y = curve.getData()
            return (np.array([]), np.array([])) if x is None else (np.asarray(x), np.asarray(y))

        voice_t, voice_midi = xy(w._curve)
        ref_x, ref_midi = xy(w._ref_curve)
        return dict(voice_t=voice_t, voice_midi=voice_midi, reference_x=ref_x, reference_midi=ref_midi,
                    now=w._now_line.value(), x_range=tuple(w._plot.viewRange()[0]))

    def screenshot(self, path: str | Path) -> Path:
        self._app.processEvents()
        path = Path(path)
        if not self.window.grab().save(str(path)):
            raise OSError(f"could not save {path}")
        return path

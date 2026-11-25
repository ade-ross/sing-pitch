from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["import"]:
        from .cli import import_main

        return import_main(argv[1:])
    if argv[:1] == ["calibrate"]:
        from .cli import calibrate_main

        return calibrate_main(argv[1:])
    from .testmode import parse_flags

    flags, argv = parse_flags(argv)
    from PySide6 import QtCore, QtWidgets

    from .pitch.swiftf0 import SwiftF0Tracker
    from .ui import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    hub, kwargs = None, {}
    if flags.any:                  # test mode: fake mic and/or null output (docs/qa.md)
        from .config import data_dir
        from .testmode import build

        hub, mic, kwargs = build(flags, data_dir())
        delay = f", mic {mic.delay_s:.3f} s late" if mic is not None else ""
        print(f"test mode: mic {flags.fake_mic or 'real'}, output {'fake' if flags.fake_output else 'real'}{delay}")
    window = MainWindow(SwiftF0Tracker(), **kwargs)
    window.show()
    stall = None
    if hub is not None and hub.faults.ui_stall_every_s is not None:
        from .testmode import UiStall

        stall = UiStall(hub, hub.faults)
    if "--smoke" in argv:
        # headless check: build the window, tracker and device list, then quit
        QtCore.QTimer.singleShot(300, app.quit)
    code = app.exec()
    if stall is not None:
        stall.stop()
    if hub is not None:
        hub.close()
        if flags.record_output:
            hub.write_recording(flags.record_output)
    if "--smoke" in argv:
        print("smoke ok")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

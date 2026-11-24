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
    from PySide6 import QtCore, QtWidgets

    from .pitch.swiftf0 import SwiftF0Tracker
    from .ui import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = MainWindow(SwiftF0Tracker())
    window.show()
    if "--smoke" in argv:
        # headless check: build the window, tracker and device list, then quit
        QtCore.QTimer.singleShot(300, app.quit)
    code = app.exec()
    if "--smoke" in argv:
        print("smoke ok")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

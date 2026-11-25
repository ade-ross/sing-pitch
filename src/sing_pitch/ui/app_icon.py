"""The app's icon (docs/logo/: logo.svg on Apple's icon grid, rendered by docs/logo/icon.html). Set on the application,
macOS shows it in the Dock while the app runs."""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtGui

ICON_PATH = Path(__file__).with_name("assets") / "icon.png"


def app_icon() -> QtGui.QIcon:
    return QtGui.QIcon(str(ICON_PATH))

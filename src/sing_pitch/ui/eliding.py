"""A combo box that stays `chars` wide whatever its items say."""
from __future__ import annotations

from PySide6 import QtWidgets


def eliding_combo(chars: int = 16) -> QtWidgets.QComboBox:
    """Long names are elided in the box (the popup shows them whole), so a YouTube-length song title or Section name
    cannot widen the window (#25)."""
    box = QtWidgets.QComboBox()
    box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
    box.setMinimumContentsLength(chars)
    return box

"""A combo box that stays `chars` wide whatever its items say."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets


def eliding_combo(chars: int = 16) -> QtWidgets.QComboBox:
    """Long names are elided in the box (the popup shows them whole), so a YouTube-length song title or Section name
    cannot widen the window (#25)."""
    box = QtWidgets.QComboBox()
    box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
    box.setMinimumContentsLength(chars)
    return box


class ElidedLabel(QtWidgets.QLabel):
    """A one-line label that never asks for room: whatever its text says, it asks the layout for no minimum width and
    shows what fits, with "…" for the rest and the whole text in its tooltip. A status text can then never widen the
    window (#25, #35, #52), and a window that resizes itself under a running Take breaks the sync (#49).
    `text()` is always the full text. `takes_the_rest`: the label wants no room of its own and shows in whatever the
    row has left over (give it a stretch factor in the row); otherwise it asks for its text's width and gives way only
    when the row is short."""

    def __init__(self, text: str = "", takes_the_rest: bool = False):
        super().__init__()
        self._full, self._tip = text, ""
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored if takes_the_rest else QtWidgets.QSizePolicy.Preferred,
                           QtWidgets.QSizePolicy.Preferred)
        self._fit()

    def text(self) -> str:
        return self._full

    def setText(self, text: str) -> None:
        if text != self._full:
            self._full = text
            self._fit()

    def setToolTip(self, tip: str) -> None:
        self._tip = tip
        self._fit()

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(0, super().minimumSizeHint().height())

    def sizeHint(self) -> QtCore.QSize:     # the room the whole text would like; the layout may give less
        return QtCore.QSize(self.fontMetrics().horizontalAdvance(self._full) + 2 * self.margin(),
                            super().sizeHint().height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        shown = self._full
        if self.width() > 1 and self.fontMetrics().horizontalAdvance(self._full) > self.width():
            shown = self.fontMetrics().elidedText(self._full, QtCore.Qt.ElideRight, self.width())
        elided = shown != self._full
        tip = (f"{self._full}\n\n{self._tip}" if self._tip else self._full) if elided else self._tip
        if shown != super().text():
            super().setText(shown)
        if tip != super().toolTip():
            super().setToolTip(tip)

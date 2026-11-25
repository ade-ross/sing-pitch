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
    """A label that never asks for room: whatever its text says, it asks the layout for no minimum width and shows
    what fits, with "…" for the rest and the whole text in its tooltip. A status text can then never widen the window
    (#25, #35, #52, #66), and a window that resizes itself under a running Take breaks the sync (#49).
    `text()` is always the full text. `lines` > 1: the text wraps (at spaces, or anywhere inside a word too long for
    a line) into that many lines; the label is always exactly that tall, so a longer text never changes its height.
    `takes_the_rest`: the label wants no room of its own and shows in whatever the row has left over (give it a
    stretch factor in the row); otherwise it asks for its text's width and gives way only when the row is short."""

    def __init__(self, text: str = "", takes_the_rest: bool = False, lines: int = 1):
        super().__init__()
        self._full, self._tip, self._lines = text, "", lines
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored if takes_the_rest else QtWidgets.QSizePolicy.Preferred,
                           QtWidgets.QSizePolicy.Fixed if lines > 1 else QtWidgets.QSizePolicy.Preferred)
        if lines > 1:
            self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
            self.setFixedHeight(lines * self.fontMetrics().lineSpacing() + 2 * self.margin())
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

    def _wrapped(self, width: int) -> str:
        """The text as at most `lines` lines no wider than `width`; the last one elided when there is more."""
        fm = self.fontMetrics()
        todo, lines = " ".join(self._full.split()), []
        while todo and len(lines) < self._lines:
            if len(lines) == self._lines - 1:
                lines.append(fm.elidedText(todo, QtCore.Qt.ElideRight, width))
                break
            cut = len(todo)
            while cut > 1 and fm.horizontalAdvance(todo[:cut]) > width:
                cut -= 1
            if cut < len(todo) and (space := todo.rfind(" ", 0, cut + 1)) > 0:
                cut = space                         # break at a space when there is one on this line
            lines.append(todo[:cut].rstrip())
            todo = todo[cut:].lstrip()
        return "\n".join(lines)

    def _fit(self) -> None:
        width = max(self.width() - 2 * self.margin(), 0)
        if self._lines > 1:
            shown = self._wrapped(width) if width > 1 else self._full
            elided = width > 1 and shown.replace("\n", " ") != " ".join(self._full.split())
        else:
            shown = self._full
            if width > 1 and self.fontMetrics().horizontalAdvance(self._full) > width:
                shown = self.fontMetrics().elidedText(self._full, QtCore.Qt.ElideRight, width)
            elided = shown != self._full
        tip = (f"{self._full}\n\n{self._tip}" if self._tip else self._full) if elided else self._tip
        if shown != super().text():
            super().setText(shown)
        if tip != super().toolTip():
            super().setToolTip(tip)

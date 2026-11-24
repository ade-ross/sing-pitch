"""History view: % in tune and median cents off per Take over dates, filterable by Reference and song."""
from __future__ import annotations

from pathlib import Path

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..config import data_dir
from ..history import WHOLE_SONG, History, Series, load_history

REFERENCE_LABEL = {"nearest": "Nearest note", "key": "Key", "melody": "Melody"}
NOTES_LABEL = {"off": "Line", "on": "Notes"}
GUIDE_LABEL = {"off": "Guide off", "on": "Guide on", "mixed": "Guide switched mid-Take"}
EMPTY_TEXT = "No Takes yet. Sing a Take and it will show up here."
EMPTY_FILTER_TEXT = "No scored Takes match this filter."


class HistoryView(QtWidgets.QWidget):
    def __init__(self, home: Path | None = None):
        super().__init__()
        self.setWindowTitle("sing-pitch history")
        self.resize(900, 600)
        self._home = home or data_dir()

        self.reference_box = QtWidgets.QComboBox()
        self.song_box = QtWidgets.QComboBox()
        self.guide_box = QtWidgets.QComboBox()
        self.speed_box = QtWidgets.QComboBox()
        self.section_box = QtWidgets.QComboBox()
        self.notes_box = QtWidgets.QComboBox()
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.empty_label = QtWidgets.QLabel(EMPTY_TEXT)
        self.empty_label.setAlignment(QtCore.Qt.AlignCenter)

        self.plot_widget = pg.GraphicsLayoutWidget(show=False)
        self.plot_widget.setBackground("#1e1e1e")
        self._pct_plot = self.plot_widget.addPlot(row=0, col=0, axisItems={"bottom": pg.DateAxisItem()})
        self._cents_plot = self.plot_widget.addPlot(row=1, col=0, axisItems={"bottom": pg.DateAxisItem()})
        self._pct_plot.setLabel("left", "% in tune")
        self._pct_plot.setYRange(0, 100)
        self._cents_plot.setLabel("left", "median cents off")
        self._cents_plot.setYRange(0, 50)
        for p in (self._pct_plot, self._cents_plot):
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setMouseEnabled(False, False)
            p.hideButtons()
        self._cents_plot.setXLink(self._pct_plot)
        self.pct_curve = self._pct_plot.plot(pen=pg.mkPen("#2ecc71", width=2), symbol="o", symbolSize=7,
                                             symbolBrush="#2ecc71")
        self.cents_curve = self._cents_plot.plot(pen=pg.mkPen("#f1c40f", width=2), symbol="o", symbolSize=7,
                                                 symbolBrush="#f1c40f")

        top = QtWidgets.QHBoxLayout()
        for w in (QtWidgets.QLabel("Reference"), self.reference_box, QtWidgets.QLabel("Song"), self.song_box,
                  QtWidgets.QLabel("Guide vocal"), self.guide_box, QtWidgets.QLabel("Speed"), self.speed_box,
                  QtWidgets.QLabel("Section"), self.section_box, QtWidgets.QLabel("Notes mode"), self.notes_box):
            top.addWidget(w)
        top.addStretch(1)
        top.addWidget(self.refresh_button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.empty_label, 1)
        layout.addWidget(self.plot_widget, 1)

        self.refresh_button.clicked.connect(self.reload)
        for box in self._boxes():
            box.currentIndexChanged.connect(self._apply_filters)
        self.reload()

    def _boxes(self) -> tuple[QtWidgets.QComboBox, ...]:
        return self.reference_box, self.song_box, self.guide_box, self.speed_box, self.section_box, self.notes_box

    def reload(self) -> None:
        """Re-read the summary table and rebuild the filter boxes, keeping the current choice when it still exists."""
        self.history: History = load_history(self._home)
        h = self.history
        choices = (   # (box, items as (label, value), enabled)
            (self.reference_box, [(REFERENCE_LABEL.get(r, r), r) for r in h.references], not h.is_empty),
            (self.song_box, [(s, s) for s in h.songs], bool(h.songs)),   # songs exist only once Melody Takes are saved
            (self.guide_box, [(GUIDE_LABEL[g], g) for g in h.guide_states], len(h.guide_states) > 1),
            (self.speed_box, [(f"{p} %", p) for p in h.speeds], len(h.speeds) > 1),
            (self.section_box, [("Whole song", WHOLE_SONG)] * bool(h.sections) + [(s, s) for s in h.sections],
             bool(h.sections)),
            (self.notes_box, [(NOTES_LABEL[m], m) for m in h.note_modes], len(h.note_modes) > 1),
        )
        for box, items, enabled in choices:
            keep = box.currentData()
            box.blockSignals(True)
            box.clear()
            box.addItem("All", None)
            for label, value in items:
                box.addItem(label, value)
            box.setCurrentIndex(max(box.findData(keep), 0))
            box.blockSignals(False)
            box.setEnabled(enabled)
        self._apply_filters()

    def _apply_filters(self) -> None:
        ref, song, guide, speed, section, notes = (box.currentData() for box in self._boxes())
        self.show_series(self.history.series(ref, song, guide, speed, section, notes))

    def show_series(self, series: Series) -> None:
        if series.started_at:
            xs = [t.timestamp() for t in series.started_at]
            self.pct_curve.setData(xs, series.pct_in_tune)
            self.cents_curve.setData(xs, series.median_abs_cents)
        else:
            self.pct_curve.setData([], [])
            self.cents_curve.setData([], [])
            self.empty_label.setText(EMPTY_TEXT if self.history.is_empty else EMPTY_FILTER_TEXT)
        self.empty_label.setVisible(not series.started_at)
        self.plot_widget.setVisible(bool(series.started_at))

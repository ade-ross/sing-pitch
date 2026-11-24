"""History view: % in tune and median cents off per Take over dates, filterable by Reference and song."""
from __future__ import annotations

from pathlib import Path

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..config import data_dir
from ..history import History, Series, load_history

REFERENCE_LABEL = {"nearest": "Nearest note", "key": "Key", "melody": "Melody"}
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
        for w in (QtWidgets.QLabel("Reference"), self.reference_box, QtWidgets.QLabel("Song"), self.song_box):
            top.addWidget(w)
        top.addStretch(1)
        top.addWidget(self.refresh_button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.empty_label, 1)
        layout.addWidget(self.plot_widget, 1)

        self.refresh_button.clicked.connect(self.reload)
        self.reference_box.currentIndexChanged.connect(self._apply_filters)
        self.song_box.currentIndexChanged.connect(self._apply_filters)
        self.reload()

    def reload(self) -> None:
        """Re-read the summary table and rebuild the filter boxes, keeping the current choice when it still exists."""
        self.history: History = load_history(self._home)
        keep_ref, keep_song = self.reference_box.currentData(), self.song_box.currentData()
        for box, label, items, keep in (
            (self.reference_box, "All", self.history.references, keep_ref),
            (self.song_box, "All", self.history.songs, keep_song),
        ):
            box.blockSignals(True)
            box.clear()
            box.addItem(label, None)
            for item in items:
                box.addItem(REFERENCE_LABEL.get(item, item) if box is self.reference_box else item, item)
            box.setCurrentIndex(max(box.findData(keep), 0))
            box.blockSignals(False)
        self.reference_box.setEnabled(not self.history.is_empty)
        self.song_box.setEnabled(bool(self.history.songs))  # songs exist only once Melody Takes are saved
        self._apply_filters()

    def _apply_filters(self) -> None:
        self.show_series(self.history.series(self.reference_box.currentData(), self.song_box.currentData()))

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

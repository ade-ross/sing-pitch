"""Run Python's cycle collector on the UI thread only.

The automatic collector runs on whichever thread happens to allocate when a threshold is crossed: in this app that
is often the tracker worker ("sing-pitch-tracker") or a Speed preparer. If that collection finalises a Qt wrapper,
PySide cannot delete the C++ object on that thread and hands it to the UI thread, and that hand-over crashed the
test suite (exit 139: shiboken mainThreadDeletionHandler, Shiboken::Object::destroy inside a DeferredDelete).
Freeing nothing Qt is not enough either: a collection on a worker thread still walks every live Qt wrapper
(shiboken's SbkObject_tp_traverse) while the UI thread is changing them, and that crashed too when the suite was
stressed with collections on the tracker thread after every reference cycle through a widget had been removed.

So the automatic collector is switched off and a timer on the UI thread collects instead, with the same generation
thresholds. pyqtgraph ships this exact remedy (pyqtgraph.util.garbage_collector.GarbageCollector, "to ensure that
garbage collection only happens in the GUI thread, as otherwise Qt can crash"); this module only makes it
idempotent and owned in one place. Worker threads still free everything acyclic at once by reference counting;
only cyclic garbage waits, at most one interval."""
from __future__ import annotations

import threading

from pyqtgraph.util.garbage_collector import GarbageCollector

INTERVAL_S = 1.0
_collector: GarbageCollector | None = None


def collect_on_the_ui_thread_only() -> None:
    """Idempotent. Call on the UI thread once a QApplication exists (the window does it when it is built)."""
    global _collector
    if _collector is None:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("the UI-thread garbage collector must be installed from the UI thread")
        _collector = GarbageCollector(interval=INTERVAL_S)     # gc.disable() + a QTimer that collects here

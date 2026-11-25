"""Find Qt objects that only the cyclic garbage collector would free.

Why it matters: Python frees an object in a reference cycle only when the cycle collector runs, and that can happen
on whatever thread happens to allocate: the tracker worker, a Speed preparer. A Qt wrapper finalised there makes
PySide hand the C++ deletion to the UI thread, and that crashed the suite twice (exit 139, shiboken
mainThreadDeletionHandler / Shiboken::Object::destroy in a DeferredDelete). The app now collects on the UI thread only
(sing_pitch/ui/gui_gc.py), and on top of that a live Qt object must be freed by reference counting the moment its
owner drops it, never left to the collector."""
from __future__ import annotations

import collections
import gc


def collect_and_report_live_qt_garbage() -> collections.Counter:
    """Run a full collection on THIS thread and return {type name: count} of the live Qt objects it found in cycles.

    Everything found is freed here, on the calling (UI) thread, as it should be."""
    import shiboken6
    from PySide6.QtCore import QObject
    from PySide6.QtWidgets import QGraphicsItem

    debug = gc.get_debug()
    gc.set_debug(debug | gc.DEBUG_SAVEALL)       # keep what the collector finds so it can be looked at
    try:
        gc.collect()
    finally:
        gc.set_debug(debug)
    found: collections.Counter = collections.Counter()
    for o in gc.garbage:
        try:
            if isinstance(o, (QObject, QGraphicsItem)) and shiboken6.isValid(o):
                found[type(o).__name__] += 1
        except ReferenceError:      # a weakref.proxy whose target went with it (numba keeps some while it compiles)
            pass
    gc.garbage.clear()
    gc.collect()                                  # now free them, here
    return found

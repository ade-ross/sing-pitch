import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must be set before any Qt import

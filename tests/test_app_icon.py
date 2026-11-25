"""The app's Dock icon: the logo (docs/logo/) shipped in the package and set on the application at launch."""
import subprocess
import sys

from PySide6 import QtGui

from sing_pitch.ui.app_icon import ICON_PATH, app_icon


def test_the_icon_ships_in_the_package_as_a_square_with_transparent_corners(qapp):
    image = QtGui.QImage(str(ICON_PATH))
    assert (image.width(), image.height()) == (1024, 1024)
    assert image.hasAlphaChannel() and image.pixelColor(5, 5).alpha() == 0     # Apple's grid: room around the logo
    assert image.pixelColor(512, 512).alpha() == 255
    assert not app_icon().isNull()


def test_launching_the_app_sets_the_icon_on_the_application():
    code = ("import sys; from PySide6 import QtWidgets; from sing_pitch.__main__ import main; main(['--smoke']); "
            "print('ICON', not QtWidgets.QApplication.instance().windowIcon().isNull())")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                         env={**__import__("os").environ, "QT_QPA_PLATFORM": "offscreen"}).stdout
    assert "ICON True" in out

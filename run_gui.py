"""
run_gui.py
==========
Start the LSR MMTO GUI, from the repository folder, in the lsr-stress-mmto-ad environment:

    python run_gui.py                       an empty window: File > New or File > Open
    python run_gui.py projects/beam.mmto    with a project open

The desktop shortcut (make_shortcut.py) starts it with the environment's pythonw, which has no
console: then the environment's library folders are found without activating it, what the app
prints goes to a log file (LOG), and a failure to start is shown in a message box.

Loading the libraries takes several seconds (longer the first time after a restart), so a small
picture with the icon says the app is starting until the window shows.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
_app = None
LOG = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "LSR MMTO", "gui.log")


def without_console():
    """Started with no console (pythonw): the environment's library folders on the search path, as
    activating it would put them, and the output into LOG. True when that is the case."""
    if sys.stdout is not None:
        return False
    if os.name == "nt" and not os.environ.get("CONDA_PREFIX"):
        p = sys.prefix
        folders = [p, os.path.join(p, "Library", "mingw-w64", "bin"), os.path.join(p, "Library", "usr", "bin"),
                   os.path.join(p, "Library", "bin"), os.path.join(p, "Scripts"), os.path.join(p, "bin")]
        folders = [f for f in folders if os.path.isdir(f)]
        os.environ["PATH"] = os.pathsep.join(folders + [os.environ.get("PATH", "")])
        os.environ["CONDA_PREFIX"] = p
        for f in folders:
            os.add_dll_directory(f)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    sys.stdout = sys.stderr = open(LOG, "w", buffering=1, encoding="utf-8", errors="replace")
    return True


def splash_screen():
    """The start-up picture: the icon and "Starting LSR MMTO...", shown at once, before the libraries load."""
    from PyQt5 import QtCore, QtGui, QtWidgets
    global _app                                                     # kept: a QApplication without a name is deleted
    _app = app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    picture = QtGui.QPixmap(320, 260)
    picture.fill(QtGui.QColor("#ffffff"))
    paint = QtGui.QPainter(picture)
    paint.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
    icon = QtGui.QPixmap(os.path.join(ROOT, "gui", "assets", "lsr_mmto.png"))
    paint.drawPixmap(100, 28, 120, 120, icon)
    paint.setPen(QtGui.QColor("#222222"))
    paint.setFont(QtGui.QFont("Segoe UI", 16))
    paint.drawText(QtCore.QRect(0, 160, 320, 36), QtCore.Qt.AlignCenter, "LSR MMTO")
    paint.setPen(QtGui.QColor("#666666"))
    paint.setFont(QtGui.QFont("Segoe UI", 10))
    paint.drawText(QtCore.QRect(0, 200, 320, 30), QtCore.Qt.AlignCenter, "Starting...")
    paint.setPen(QtGui.QColor("#d0d0d0"))
    paint.drawRect(0, 0, 319, 259)
    paint.end()
    splash = QtWidgets.QSplashScreen(picture)
    splash.show()
    app.processEvents()
    return splash


def failed_to_start(error):
    """The reason in a message box, since there is no console to read it in."""
    import traceback
    traceback.print_exc()
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)  # noqa: F841
    QtWidgets.QMessageBox.critical(None, "LSR MMTO", f"LSR MMTO could not start:\n\n{error}\n\nThe details are in "
                                                     f"{LOG}")


if __name__ == "__main__":
    windowed = without_console()
    if os.name == "nt":                                             # the taskbar shows the app's own icon
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("UW-ERSL.LSR-MMTO")
    splash = splash_screen()
    try:
        sys.path.insert(0, ROOT)
        from utils import project_paths
        project_paths.bootstrap()
        from gui.main_window import main
    except Exception as e:                                          # noqa: BLE001
        splash.close()
        if not windowed:
            raise
        failed_to_start(e)
        sys.exit(1)
    sys.exit(main(sys.argv[1:], splash=splash))

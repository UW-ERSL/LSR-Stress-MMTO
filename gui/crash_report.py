"""The crash report: one file that keeps what is needed to understand a crash or a freeze of the GUI.

It holds, each line with its time:
- every thread's stack when the app crashes outright (Python's faulthandler, which works even when Python
  itself can no longer run);
- every error the app catches and shows in a message box, and errors in background threads;
- every thread's stack when the window stops responding for FREEZE_AFTER seconds, written once per freeze,
  then how long the freeze lasted when the window responds again. faulthandler's own timer writes it, so it
  works even when the window is stuck inside a library (a solve, a 3D draw);
- what the user did: buttons pressed, menu items chosen, windows opened, projects opened. Written as it
  happens, so the last actions before a crash are kept.

The file is cut back to its last KEEP lines at each start, and whenever it passes TRIM_AT lines, so it stays
small however long the app is used. start() is called once, by main(); the tests call it with their own file.
"""
import datetime
import faulthandler
import os
import sys
import threading
import time
import traceback

FOLDER = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "LSR MMTO")
PATH = os.path.join(FOLDER, "crash_report.log")
KEEP, TRIM_AT = 2000, 4000                                          # lines kept, and when to cut back to them
FREEZE_AFTER = 10                                                   # seconds without a response

_state = {"file": None, "path": None, "lines": 0, "freeze_after": FREEZE_AFTER, "watch": None}
_lock = threading.RLock()


def started():
    return _state["file"] is not None


def where():
    """The file in use, or None before start()."""
    return _state["path"]


def start(path=PATH, freeze_after=FREEZE_AFTER):
    """Opens the file (cut back to its last KEEP lines), writes the session's first line, and starts
    watching for crashes, errors in threads, freezes and what the user does. Needs the QApplication."""
    from PyQt5 import QtWidgets
    if started():
        stop()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _state.update(path=path, freeze_after=freeze_after)
    _open(keep=KEEP)
    write(f"LSR MMTO started (Python {sys.version.split()[0]}, process {os.getpid()})")
    threading.excepthook = _thread_error
    app = QtWidgets.QApplication.instance()
    _state["watch"] = _Watch(app, freeze_after)
    app.aboutToQuit.connect(stop)


def stop():
    """Ends the session: the freeze timer and the watchers off, the last line written, the file closed."""
    with _lock:
        if not started():
            return
        watch, _state["watch"] = _state["watch"], None
        if watch is not None:
            watch.close()
        faulthandler.cancel_dump_traceback_later()
        threading.excepthook = threading.__excepthook__
        write("LSR MMTO closed")
        faulthandler.disable()
        _state["file"].close()
        _state["file"] = None


def write(text):
    """One entry, with its time. Does nothing before start(), so the rest of the app can call it freely."""
    with _lock:
        f = _state["file"]
        if f is None:
            return
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{stamp}  {text}\n")
        f.flush()
        _state["lines"] += text.count("\n") + 1
        if _state["lines"] > TRIM_AT:
            _open(keep=KEEP)


def error(kind, value, tb):
    """An error the app caught, with its traceback."""
    write("Error:\n" + "".join(traceback.format_exception(kind, value, tb)).rstrip())


def _thread_error(args):
    write(f"Error in the background thread {args.thread.name if args.thread else '?'}:\n" +
          "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)).rstrip())


def _open(keep):
    """The file reopened with only its last keep lines, and faulthandler pointed at it."""
    path = _state["path"]
    old = _state["file"]
    if old is not None:
        old.close()
    lines = []
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[-keep:]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    _state["file"] = open(path, "a", encoding="utf-8", errors="replace")
    _state["lines"] = len(lines)
    faulthandler.enable(_state["file"], all_threads=True)
    if _state["watch"] is not None:                                 # the freeze timer to the new file
        _state["watch"].arm()


class _Watch:
    """The freeze detector and the record of what the user does.

    Freeze: a timer on the window's thread rearms faulthandler's timer every second. While the window
    responds, faulthandler's timer never runs out; when the window stops responding, it runs out after
    freeze_after seconds and writes every thread's stack. The next tick sees the gap and writes how long it was.
    """

    def __init__(self, app, freeze_after):
        from PyQt5 import QtCore
        self.freeze_after = freeze_after
        self.last = time.monotonic()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(1000)
        self.actions = _actions_class()()
        app.installEventFilter(self.actions)
        self.app = app
        self.arm()

    def arm(self):
        faulthandler.dump_traceback_later(self.freeze_after, repeat=False, file=_state["file"])

    def tick(self):
        now = time.monotonic()
        gap, self.last = now - self.last, now
        if gap > self.freeze_after:
            write(f"The window responded again after {gap:.0f} s")
        self.arm()

    def close(self):
        self.timer.stop()
        self.app.removeEventFilter(self.actions)


def _actions_class():
    from PyQt5 import QtCore, QtWidgets

    class Actions(QtCore.QObject):
        """Buttons pressed, menu items chosen and windows opened, one line each."""

        def eventFilter(self, obj, event):                          # noqa: N802 (Qt's name)
            try:
                kind = event.type()
                if kind == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
                    if isinstance(obj, QtWidgets.QMenu):
                        action = obj.activeAction()
                        if action is not None and action.text():
                            write(f"chose: {action.text().replace('&', '')}")
                    elif isinstance(obj, QtWidgets.QAbstractButton) and obj.isEnabled() and \
                            obj.rect().contains(event.pos()):
                        write(f"pressed: {_label(obj)}")
                elif kind == QtCore.QEvent.Show and isinstance(obj, QtWidgets.QWidget) and obj.isWindow() and \
                        obj.windowType() in (QtCore.Qt.Window, QtCore.Qt.Dialog) and obj.windowTitle():
                    write(f"opened: {obj.windowTitle()}")
            except Exception:                                       # noqa: BLE001 (the record never breaks the app)
                pass
            return False

    return Actions


def _label(button):
    """What the user sees on a button: its text, else its tooltip, else its name."""
    return (button.text() or button.toolTip() or button.objectName() or type(button).__name__).replace("&", "")

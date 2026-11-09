"""The crash report (gui/crash_report.py): what normal use, an error in a button, an error in a background
thread and a freeze each leave in the file, and that the file stays short.

The GUI runs in a separate process: faulthandler and the freeze timer are process-wide, and pytest has its
own faulthandler. The freeze limit there is 2 seconds instead of 10, so the test is quick.
"""
import os
import subprocess
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SESSION = r'''
import os, sys, threading, time
import torch  # noqa: F401 (as run_gui.py)
from PyQt5 import QtCore, QtTest, QtWidgets
app = QtWidgets.QApplication(sys.argv)
from gui import crash_report, main_window as G

def wait(seconds):
    QtTest.QTest.qWait(int(seconds * 1000))                     # Qt's own event loop, as the app runs

QtWidgets.QMessageBox.critical = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok)
crash_report.start(sys.argv[1], freeze_after=2)
w = G.MMTOWindow()
G.install_error_handler(w)
w.show()
wait(3)                                                         # the window's start-up is not normal use

crash_report.write("=== normal")
ok = QtWidgets.QPushButton("Planted button", w)
ok.show()
wait(0.2)
QtTest.QTest.mouseClick(ok, QtCore.Qt.LeftButton)
wait(3)

crash_report.write("=== error in a button")
bad = QtWidgets.QPushButton("Planted error", w)
bad.clicked.connect(lambda: (_ for _ in ()).throw(ValueError("planted error in a button")))
bad.show()
wait(0.2)
QtTest.QTest.mouseClick(bad, QtCore.Qt.LeftButton)
wait(0.5)

crash_report.write("=== error in a thread")
def fails():
    raise RuntimeError("planted error in a thread")
t = threading.Thread(target=fails, name="planted thread")
t.start()
t.join()

crash_report.write("=== freeze")
QtCore.QTimer.singleShot(0, lambda: time.sleep(4))
wait(6)

import shutil
shutil.copy(sys.argv[1], sys.argv[1] + ".before_trim")         # the trim below removes all of the above
crash_report.write("=== trim")
for i in range(crash_report.TRIM_AT + 100):
    crash_report.write(f"filler {i}")
crash_report.write("=== end")
crash_report.stop()
'''


@pytest.fixture(scope="module")
def report():
    """The crash report written by one scripted session, split at its === markers."""
    folder = tempfile.mkdtemp(prefix="_test_crash_")
    path = os.path.join(folder, "crash_report.log")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    done = subprocess.run([sys.executable, "-c", SESSION, path], cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=180)
    assert done.returncode == 0, done.stderr[-3000:]
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    with open(path + ".before_trim", encoding="utf-8") as f:
        before_trim = f.read().splitlines()
    parts, name = {}, "before"
    for line in before_trim:
        if "  === " in line:
            name = line.split("  === ", 1)[1]
            continue
        parts.setdefault(name, []).append(line)
    return {k: "\n".join(v) for k, v in parts.items()}, lines


def test_normal_use_writes_only_actions(report):
    parts, _ = report
    assert "pressed: Planted button" in parts["normal"]
    for trouble in ("Error", "Timeout", "responded again"):
        assert trouble not in parts["normal"], f"normal use wrote '{trouble}':\n{parts['normal']}"


def test_an_error_in_a_button_is_written(report):
    text = report[0]["error in a button"]
    assert "pressed: Planted error" in text and "ValueError: planted error in a button" in text, text


def test_an_error_in_a_thread_is_written(report):
    text = report[0]["error in a thread"]
    assert "planted thread" in text and "RuntimeError: planted error in a thread" in text, text


def test_a_freeze_writes_every_stack_then_how_long(report):
    text = report[0]["freeze"]
    assert "Timeout" in text, f"no stack was written during the freeze:\n{text}"
    assert "Thread" in text and "<lambda>" in text, f"the stack does not show the frozen function:\n{text}"
    assert "The window responded again after" in text, text


def test_the_file_stays_short(report):
    _, lines = report
    from gui import crash_report
    assert len(lines) <= crash_report.TRIM_AT + 10, len(lines)
    assert lines[-1].endswith("LSR MMTO closed") and "=== end" in lines[-2]
    assert not any("=== normal" in line for line in lines), "the oldest lines were not cut"

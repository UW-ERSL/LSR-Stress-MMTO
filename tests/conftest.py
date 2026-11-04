"""
conftest.py
===========
No test waits for a click. Off-screen, a message box or a dialog that opens during a test would
wait for an answer that never comes, and the whole run would hang. Here every message box and dialog
answers at once (Cancel, or OK where that is its only button), and a test that met one it did not
ask for fails at its end, naming the dialog's title and text. A test that means to answer a dialog
says so, as test_gui.answers() does. The app's settings (what a user's window keeps between sessions) go to
a folder of the test run's own.
"""
import tempfile

import pytest

_unasked = []


def _own_settings():
    """The app's settings file in a folder of the test run's own, so tests never change the user's."""
    try:
        from PyQt5 import QtCore
    except ImportError:
        return
    folder = tempfile.mkdtemp(prefix="lsr_mmto_test_settings_")
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, folder)


_own_settings()


def _install():
    from PyQt5 import QtWidgets
    Box = QtWidgets.QMessageBox
    saved = {
        (Box, "question"): Box.question, (Box, "warning"): Box.warning,
        (Box, "information"): Box.information, (Box, "critical"): Box.critical,
        (QtWidgets.QDialog, "exec_"): QtWidgets.QDialog.exec_, (Box, "exec_"): Box.exec_,
        (QtWidgets.QFileDialog, "getOpenFileName"): QtWidgets.QFileDialog.getOpenFileName,
        (QtWidgets.QFileDialog, "getSaveFileName"): QtWidgets.QFileDialog.getSaveFileName,
        (QtWidgets.QFileDialog, "getExistingDirectory"): QtWidgets.QFileDialog.getExistingDirectory,
        (QtWidgets.QInputDialog, "getText"): QtWidgets.QInputDialog.getText,
    }

    def box(kind, answer):
        def answered(parent, title, text, *a, **k):
            _unasked.append(f"{kind} '{title}': {text}")
            return answer
        return staticmethod(answered)

    def dialog(self, *a, **k):
        _unasked.append(f"dialog '{self.windowTitle()}'")
        return QtWidgets.QDialog.Rejected

    def file_dialog(kind, answer):
        def answered(*a, **k):
            _unasked.append(f"file dialog ({kind})")
            return answer
        return staticmethod(answered)

    Box.question = box("question", Box.Cancel)
    Box.warning = box("warning", Box.Ok)
    Box.information = box("information", Box.Ok)
    Box.critical = box("critical", Box.Ok)
    QtWidgets.QDialog.exec_ = dialog
    Box.exec_ = lambda self, *a, **k: (_unasked.append(f"message '{self.windowTitle()}': {self.text()}"),
                                       Box.Cancel)[1]
    QtWidgets.QFileDialog.getOpenFileName = file_dialog("open", ("", ""))
    QtWidgets.QFileDialog.getSaveFileName = file_dialog("save", ("", ""))
    QtWidgets.QFileDialog.getExistingDirectory = file_dialog("folder", "")
    QtWidgets.QInputDialog.getText = file_dialog("text", ("", False))
    return saved


@pytest.fixture(autouse=True)
def no_dialog_waits():
    """Dialogs answer at once; one the test did not answer itself fails the test."""
    try:
        saved = _install()
    except ImportError:                          # a test session without Qt
        yield
        return
    _unasked.clear()
    try:
        yield
    finally:
        for (owner, name), value in saved.items():
            setattr(owner, name, value)
    if _unasked:
        met = "; ".join(_unasked)
        _unasked.clear()
        pytest.fail(f"a dialog opened that the test did not answer: {met}", pytrace=False)

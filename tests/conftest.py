"""
conftest.py
===========
No test waits for a click. Off-screen, a message box or a dialog that opens during a test would
wait for an answer that never comes, and the whole run would hang. Here every message box and dialog
answers at once (Cancel, or OK where that is its only button), and a test that met one it did not
ask for fails at its end, naming the dialog's title and text. A test that means to answer a dialog
says so, as test_gui.answers() does. The app's settings (what a user's window keeps between sessions, the
message panel's height) go to a folder of the test run's own. And no solve is left unchecked: plain CG, which
PyTO-AD falls back to for a system that no deflation space fits, fails the test that reaches it (every iterative
solve is DPCG on a checked solver, optimize/iterative_solve_check.py). And each physics' tiny example runs at
most once per test session (once per worker with -n), for every test that only needs a run of that physics to
look at (tiny_run). And the scratch folders a test run makes in the repository folder (_test_*) are removed by the
process that made them after its last test, with any an earlier run left a day ago or more (pytest_sessionfinish).
"""
import importlib
import os
import sys
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


@pytest.fixture(autouse=True)
def no_unchecked_solve(monkeypatch):
    """Plain CG fails the test: PyTO-AD reaches it only for a system no deflation space fits, and its solves are
    not checked."""
    import scipy.sparse.linalg

    def plain_cg(*args, **kwargs):
        raise AssertionError("an iterative solve ran plain CG, which is not checked: the physics' deflation_space "
                             "has no space for this system (iterative_solve_check.py)")
    monkeypatch.setattr(scipy.sparse.linalg, "cg", plain_cg)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tiny_example_of(physics):
    """The tiny_example function of a physics, from its own test file, tests/physics/test_<physics>_physics.py;
    the test fails, saying what to add, when that file or its tiny_example is missing."""
    file = f"tests/physics/test_{physics}_physics.py"
    if not os.path.exists(os.path.join(ROOT, file)):
        pytest.fail(f"the {physics} physics has no test file: add {file}, with a tiny_example "
                    f"(start from tests/physics/test_structural_physics.py)", pytrace=False)
    sys.path.insert(0, os.path.join(ROOT, "tests", "physics"))
    module = importlib.import_module(f"test_{physics}_physics")
    if not hasattr(module, "tiny_example"):
        pytest.fail(f"{file} has no tiny_example: add one (see tiny_example in "
                    f"tests/physics/test_structural_physics.py)", pytrace=False)
    return module.tiny_example


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory):
    """tiny_run(physics): that physics' tiny example (tiny_example_of) run and saved, once per test session (once
    per worker with -n); (the project, the run read back from its saved folder). A test that needs a run of a
    physics, and nothing about how that run was set up, reads this one instead of running its own. Only for
    tests that change nothing: no monkeypatch around the call, no edit of the project or the run, since every
    later reader gets the same ones."""
    from lsr_mmto.problem.project import run
    from lsr_mmto.results import load_results
    runs = {}

    def made(physics):
        if physics not in runs:
            folder = str(tmp_path_factory.mktemp(f"tiny_{physics}"))
            project = tiny_example_of(physics)(folder)
            run(project, save_dir=os.path.join(folder, "run"), verbose=False)
            runs[physics] = project, load_results.load(os.path.join(folder, "run", "run_data.npz"))
        return runs[physics]
    return made


# the scratch folders this process makes in the repository folder (tempfile.mkdtemp with a "_test_" prefix and
# dir=ROOT, as the GUI tests do), so this process removes exactly its own at the end and never another's
_made_here = []
_mkdtemp = tempfile.mkdtemp


def _recording_mkdtemp(*args, **kwargs):
    folder = _mkdtemp(*args, **kwargs)
    if os.path.basename(folder).startswith("_test_") and \
            os.path.dirname(os.path.abspath(folder)) == os.path.abspath(ROOT):
        _made_here.append(folder)
    return folder


tempfile.mkdtemp = _recording_mkdtemp


def _removed(folders, tries=1):
    """folders removed, tried up to tries times half a second apart (a folder a process that has just ended held
    open frees once Windows lets go of it); returns those still there. Touches no window: closing one runs the
    app's own close (Save?) on a window a test may have half taken apart."""
    import gc
    import shutil
    import time
    left = [d for d in folders if os.path.exists(d)]
    for attempt in range(tries):
        if not left:
            break
        if attempt:
            time.sleep(0.5)
        gc.collect()
        for d in left:
            shutil.rmtree(d, ignore_errors=True)
        left = [d for d in left if os.path.exists(d)]
    return left


_left_by_workers = []


def pytest_testnodedown(node, error):
    """On the main process, as each worker of -n ends: the scratch folders it could not remove, kept to remove
    once it has ended and let go of them."""
    _left_by_workers.extend(getattr(node, "workeroutput", {}).get("scratch_left", []))


def _last_change(folder):
    """When anything in folder last changed (seconds since the epoch)."""
    times = [os.path.getmtime(folder)]
    for here, _, files in os.walk(folder):
        times += [os.path.getmtime(os.path.join(here, f)) for f in files]
    return max(times)


def pytest_sessionfinish(session):
    """After this process's last test: the scratch folders it made in the repository folder (_test_*) that are still
    there, removed. A test removes its own as it ends, but on Windows a file still held open stops that and the
    folder is left behind. Only folders this test run made are removed, so a second test run in the same folder
    keeps its own. A worker of -n hands the ones it cannot remove to the main process, which removes them once that
    worker has ended, removes folders an earlier run left (untouched for a day), and names any it still cannot
    remove at the end of the run."""
    import glob
    import time
    left = _removed(_made_here)
    if hasattr(session.config, "workerinput"):
        session.config.workeroutput["scratch_left"] = left
        return
    still = _removed(left + _left_by_workers, tries=10)
    for d in glob.glob(os.path.join(ROOT, "_test_*")):
        try:
            if time.time() - _last_change(d) > 24 * 3600:
                _removed([d])
        except OSError:
            pass
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if still and reporter is not None:
        names = ", ".join(sorted(os.path.basename(d) for d in still))
        reporter.write_line(f"Scratch folders that could not be removed (delete them by hand): {names}", yellow=True)

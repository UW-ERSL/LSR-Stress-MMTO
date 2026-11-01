"""
test_gui.py
===========
The LSR MMTO window, off-screen: the names it takes from PyTO-AD exist; each project opened
and saved with no edits is written back unchanged (line endings aside: Windows writes CRLF,
which git does not count as a change); only .mmto projects open; New, Save and Project.load
agree; the title marks unsaved changes and closing asks; the unit system goes both ways
between the project and the Units dialog; selections made of triangle numbers are found
before the STL is changed. About ten seconds.

    python tests/test_gui.py        (or: python -m pytest tests)
"""
import contextlib
import json
import os
import shutil
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils import project_paths  # noqa: E402

project_paths.bootstrap()

from PyQt5 import QtGui, QtWidgets  # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from gui import main_window as G  # noqa: E402
from project import Project  # noqa: E402  # type: ignore

PROJECTS = os.path.join(ROOT, "projects")
NAMES = ("Airplane bracket", "L-bracket", "U-bracket", "beam", "rectangular beam")
Box = QtWidgets.QMessageBox


@contextlib.contextmanager
def answers(question=Box.Yes):
    """Message boxes answered without showing them: question() gives question, warning() OK."""
    saved = Box.question, Box.warning
    Box.question = staticmethod(lambda *a, **k: question)
    Box.warning = staticmethod(lambda *a, **k: Box.Ok)
    try:
        yield
    finally:
        Box.question, Box.warning = saved


@contextlib.contextmanager
def scratch():
    """A folder one level below the repository, like projects/, so STL paths are written the same."""
    d = tempfile.mkdtemp(prefix="_test_gui_", dir=ROOT)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_what_it_takes_from_pyto_ad():
    from gui import pyto_ad
    assert hasattr(pyto_ad.ProjectsWindow, "load_geometry") and hasattr(pyto_ad.MainWindow, "init_state")
    assert {c["name"] for c in pyto_ad.MainWindow.BUTTON_CONFIG} >= set(G.SIDEBAR)
    w = G.MMTOWindow()
    assert list(w.sidebar_buttons) == list(G.SIDEBAR) and w.windowTitle() == "LSR MMTO"
    assert [m.title() for m in (a.menu() for a in w.menuBar().actions())] == ["&File", "&Help"]
    s = w.settings
    s.update_settings("IPS", s.temperature_unit, s.angle_unit)
    assert s.unit_system == "IPS"


def test_open_and_save_unchanged():
    w = G.MMTOWindow()
    with scratch() as d:
        for name in NAMES:
            original = os.path.join(PROJECTS, f"{name}.mmto")
            path = shutil.copy(original, os.path.join(d, f"{name}.mmto"))
            assert w.open_project(path), name
            assert not w.is_dirty() and w.windowTitle() == f"{name} - LSR MMTO", w.windowTitle()
            assert w.stl_geom is not None, name
            assert w.save_project()
            assert open(path).read() == open(original).read(), name               # line endings aside


def test_only_mmto_projects_open():
    w = G.MMTOWindow()
    pyto = os.path.join(ROOT, "0-Common", "Models", "UBracket", "UBracket.pyto")
    with answers(), scratch() as d:
        assert not w.open_project(pyto)
        assert ".pyto file opens in PyTO-AD" in G.mmto_file_error(pyto)
        plain = os.path.join(d, "plain.mmto")
        json.dump({"stl_file_path": "x.STL"}, open(plain, "w"))
        assert not w.open_project(plain) and "no mmto section" in G.mmto_file_error(plain)
        assert w.project is None


def test_new_save_load():
    stl = os.path.join(ROOT, "0-Common", "Models", "LBracket_BM1", "LBracket_BM1.STL")
    w = G.MMTOWindow()
    assert w.new_project(stl)
    assert w.is_dirty() and w.windowTitle() == "LBracket_BM1* - LSR MMTO"
    with scratch() as d:
        assert w.save_project_as(os.path.join(d, "new"))                       # .mmto added
        assert w.path.endswith("new.mmto") and not w.is_dirty()
        assert Project.load(w.path).to_dict() == w.project.to_dict()


def test_unsaved_changes():
    w = G.MMTOWindow()
    with scratch() as d:
        path = shutil.copy(os.path.join(PROJECTS, "beam.mmto"), d)
        assert w.open_project(path)
        w.project.n_dof = 1234
        w._refresh_title()
        assert w.is_dirty() and w.windowTitle() == "beam* - LSR MMTO"
        with answers(Box.Cancel):
            event = QtGui.QCloseEvent()
            w.closeEvent(event)
            assert not event.isAccepted()                                     # Cancel keeps the window
            assert not w.open_project(path) and w.project.n_dof == 1234
        with answers(Box.Discard):
            assert w.open_project(path) and w.project.n_dof != 1234 and not w.is_dirty()


def test_units_both_ways():
    w = G.MMTOWindow()
    real_exec = G.UnitsWindow.exec_
    with scratch() as d:
        assert w.open_project(shutil.copy(os.path.join(PROJECTS, "L-bracket.mmto"), d))
        assert w.settings.unit_system == "mmKS"                               # project -> dialog
        try:
            G.UnitsWindow.exec_ = lambda self: 0                               # closed without Apply
            w.open_units_window()
            assert w.project.units == "mmKS" and not w.is_dirty()
            def apply_ips(self):
                self.settings.update_settings("IPS", self.settings.temperature_unit, self.settings.angle_unit)
                return 1
            G.UnitsWindow.exec_ = apply_ips                                    # dialog -> project
            w.open_units_window()
            assert w.project.units == "IPS" and w.is_dirty()
        finally:
            G.UnitsWindow.exec_ = real_exec
        with answers(Box.Discard):
            assert w.open_project(shutil.copy(os.path.join(PROJECTS, "U-bracket.mmto"), d))
        assert w.project.units == "none" and w.settings.unit_system == "MKS"   # none: the dialog shows MKS


def test_triangle_selections():
    found = {n: G.triangle_selections(Project.load(os.path.join(PROJECTS, f"{n}.mmto"))) for n in NAMES}
    assert found["L-bracket"] == ["clamp"] and found["Airplane bracket"] == [], found
    w = G.MMTOWindow()
    with scratch() as d:
        assert w.open_project(shutil.copy(os.path.join(PROJECTS, "L-bracket.mmto"), d))
        stl = w.project.stl
        with answers(Box.No):
            w.open_geometry_window()                                           # warned, and declined
        assert w.project.stl == stl and not w.is_dirty()


if __name__ == "__main__":
    failed = 0
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]:
        try:
            fn()
            print(f"ok   {name}")
        except Exception as e:                                     # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)

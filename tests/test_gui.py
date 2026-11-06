"""
test_gui.py
===========
The LSR MMTO window, off-screen: the names it takes from PyTO-AD exist; each project opened
and saved with no edits is written back unchanged (line endings aside: Windows writes CRLF,
which git does not count as a change); only .mmto projects open; New, Save and Project.load
agree; the title marks unsaved changes and closing asks; the unit system goes both ways
between the project and the Units dialog; selections made of triangle numbers are found
before the STL is changed. The Materials, Formulation and Settings panels: OK with no edits
changes nothing, Cancel drops edits, every setting has a row, quantities the database cannot
give are greyed out with the reason, a new database brings its VAE settings. Supports and loads:
a region typed or filled in is the selection a script makes, a code line is never run,
picks applied in PyTO-AD's Structural Loads window are adopted, the preview counts the run's
nodes and runs the rigid-body check. About a minute.

    python tests/test_gui.py        (or: python -m pytest tests)
"""
import contextlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform.startswith("win"):                         # off-screen Qt on Windows finds no fonts itself
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _quiet(kind, context, message):
    """Qt's messages, less the ones the off-screen mode prints for every 3D view it cannot draw."""
    if not any(s in message for s in ("OpenGL", "propagateSizeHints")):
        sys.stderr.write(message + "\n")


QtCore.qInstallMessageHandler(_quiet)

from gui import main_window as G  # noqa: E402
from lsr_mmto.problem.project import Project  # noqa: E402

PROJECTS = os.path.join(ROOT, "projects")
NAMES = ("Airplane bracket", "L-bracket", "U-bracket", "beam", "rectangular beam")
Box = QtWidgets.QMessageBox


@contextlib.contextmanager
def answers(question=Box.Yes):
    """Message boxes answered without showing them: question() gives question, warning() and
    information() OK."""
    saved = Box.question, Box.warning, Box.information
    Box.question = staticmethod(lambda *a, **k: question)
    Box.warning = staticmethod(lambda *a, **k: Box.Ok)
    Box.information = staticmethod(lambda *a, **k: Box.Ok)
    try:
        yield
    finally:
        Box.question, Box.warning, Box.information = saved


@contextlib.contextmanager
def scratch():
    """A folder one level below the repository, like projects/, so STL paths are written the same."""
    d = tempfile.mkdtemp(prefix="_test_gui_", dir=ROOT)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_what_it_takes_from_pyto_ad():
    from gui import pyto_ad_imports
    assert hasattr(pyto_ad_imports.ProjectsWindow, "load_geometry") and hasattr(pyto_ad_imports.MainWindow, "init_state")
    assert {c["name"] for c in pyto_ad_imports.MainWindow.BUTTON_CONFIG} >= set(G.SIDEBAR) - set(G.PANELS) - \
        {"Supports and loads", "Run", "Results"}
    assert all(hasattr(pyto_ad_imports.MainWindow, m) for m in ("update_LivVar", "_raise_if_open",
                                                         "start_box_node_selection"))
    assert all(hasattr(pyto_ad_imports.StructuralLoadsWindow, m) for m in ("redraw_loads", "refresh_applied_list"))
    w = G.MMTOWindow()
    assert list(w.sidebar_buttons) == list(G.SIDEBAR) and w.windowTitle() == "LSR MMTO"
    assert [m.title() for m in (a.menu() for a in w.menuBar().actions())] == ["&File", "&Help"]
    s = w.settings
    s.update_settings("IPS", s.temperature_unit, s.angle_unit)
    assert s.unit_system == "IPS"


def test_open_and_save_unchanged():
    """A project opened and saved with no edits is written back unchanged. A copy of an example takes into
    its folder, at its first save, what it uses (the part, the database and its network), as every save
    does; from then on it is written back unchanged."""
    w = G.MMTOWindow()
    with scratch() as d:
        for name in NAMES:
            original = os.path.join(PROJECTS, f"{name}.mmto")
            path = shutil.copy(original, os.path.join(d, f"{name}.mmto"))
            assert w.open_project(path), name
            assert not w.is_dirty() and w.windowTitle() == f"{name} - LSR MMTO", w.windowTitle()
            assert w.stl_geom is not None, name
            assert w.save_project()                                              # the first save brings them in
            assert w.project.stl.startswith("part/") and w.project.database.startswith("materials/"), name
            assert os.path.isfile(w.project.stl_path()) and os.path.isfile(os.path.join(d, w.project.database))
            first = open(path).read()
            assert w.open_project(path) and w.save_project()
            assert open(path).read() == first, name                              # line endings aside


def test_a_setting_missing_from_the_file_is_no_change():
    """A project file written before a setting existed opens with that setting at its default and
    counts as unchanged; the next save writes the setting into the file."""
    w = G.MMTOWindow()
    with scratch() as d:
        path = os.path.join(d, "beam.mmto")
        data = json.load(open(os.path.join(PROJECTS, "beam.mmto")))
        section, key = next((s, k) for s, values in data["mmto"]["settings"].items() if isinstance(values, dict)
                            for k in values)
        del data["mmto"]["settings"][section][key]                         # as if the setting were new
        json.dump(data, open(path, "w"), indent=2)
        assert w.open_project(path)
        assert not w.is_dirty() and "*" not in w.windowTitle(), w.windowTitle()
        assert "settings missing from the file were set to their defaults" in w.message_text.toPlainText()
        assert w.save_project()
        assert key in json.load(open(path))["mmto"]["settings"][section]


def test_only_mmto_projects_open():
    w = G.MMTOWindow()
    with answers(), scratch() as d:
        pyto = os.path.join(d, "u.pyto")                                   # a PyTO-AD project file
        json.dump({"stl_file_path": "x.STL"}, open(pyto, "w"))
        assert not w.open_project(pyto)
        assert "is not an LSR MMTO project (.mmto)" in G.mmto_file_error(pyto)
        plain = os.path.join(d, "plain.mmto")
        json.dump({"stl_file_path": "x.STL"}, open(plain, "w"))
        assert not w.open_project(plain) and "no mmto section" in G.mmto_file_error(plain)
        assert w.project is None


def test_new_save_load():
    """New makes a project folder with the STL copied in, saved at once; Save As copies everything into
    another folder and takes the new name."""
    stl = os.path.join(ROOT, "data", "models", "LBracket_BM1", "LBracket_BM1.STL")
    w = G.MMTOWindow()
    with scratch() as d:
        folder = os.path.join(d, "Bracket study")
        assert w.new_project(stl, folder)
        assert w.path == os.path.join(folder, "Bracket study.mmto") and os.path.isfile(w.path)
        assert not w.is_dirty() and w.windowTitle() == "Bracket study - LSR MMTO"
        assert w.project.stl == "part/LBracket_BM1.STL" and os.path.isfile(os.path.join(folder, w.project.stl))
        assert w.own_folder()
        with answers():
            assert not w.new_project(stl, folder)                              # not into a folder in use
        assert w.save_project_as(os.path.join(d, "new", "new"))                # .mmto added
        assert w.path == os.path.join(d, "new", "new.mmto") and not w.is_dirty() and w.project.name == "new"
        assert os.path.isfile(os.path.join(d, "new", "part", "LBracket_BM1.STL"))
        assert Project.load(w.path).to_dict() == w.project.to_dict()


def test_save_as_brings_in_region_files():
    """Save As of a project whose region names a file of its own (the airplane's collar, inside_stl):
    the file is copied into part/ of the new folder, and the saved project points at that copy."""
    w = G.MMTOWindow()
    with scratch() as d:
        assert w.open_project(os.path.join(PROJECTS, "Airplane bracket.mmto"))
        assert w.save_project_as(os.path.join(d, "plane", "plane"))
        p = Project.load(w.path)
        files = [s.to_dict().get("path") for s in p.selections.values() if s.to_dict().get("type") == "inside_stl"]
        assert files and all(os.path.isfile(os.path.join(p.base_dir, f)) for f in files), files
        assert all(os.path.dirname(os.path.abspath(os.path.join(p.base_dir, f))) == os.path.join(d, "plane", "part")
                   for f in files)


def test_setup_script_follows_window_saves():
    """File > New writes <name>_setup.py in the project folder; every save in the window writes it again
    to match the project; a save from a script leaves it alone."""
    stl = os.path.join(ROOT, "data", "models", "LBracket_BM1", "LBracket_BM1.STL")
    w = G.MMTOWindow()
    with scratch() as d:
        folder = os.path.join(d, "Bracket study")
        assert w.new_project(stl, folder)
        script = os.path.join(folder, "Bracket study_setup.py")
        assert os.path.isfile(script) and f"n_dof={w.project.n_dof}," in open(script).read()
        w.project.n_dof = 4000
        assert w.save_project() and "n_dof=4000," in open(script).read()
        p = Project.load(w.path)
        p.n_dof = 5000
        p.save(w.path)                                                        # a script's save
        assert "n_dof=4000," in open(script).read()


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
    """Units go from the project to the dialog and back; Apply writes the part's extent, the ruler and the
    Snap steps again in the new length unit."""
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
            w.ruler_box.setChecked(True)
            G.UnitsWindow.exec_ = apply_ips                                    # dialog -> project
            w.open_units_window()
            assert w.project.units == "IPS" and w.is_dirty()
            assert w.extent.text().endswith(" in") and "mm" not in w.extent.text()   # the lengths shown follow
            assert w.snap.itemText(1) == "Snap: 0.2 in" and "in" in w.facts.toolTip()
            assert w.plotter.actors and any(r.GetTitle() == "in" for r in w._rulers if hasattr(r, "GetTitle"))
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
        other = os.path.join(ROOT, "data", "models", "Beam_Liao", "Ex1_Beam_design.stl")
        with answers(Box.No):
            assert not w.replace_part(other)                                   # warned, and declined
        assert w.project.stl == stl and not w.is_dirty()


def test_replace_part():
    """File > Replace part changes the project's STL (copied into the part folder of a project of its own),
    draws the new part and drops the design of the old one. There is no Geometry button."""
    assert "Geometry" not in G.SIDEBAR
    w = G.MMTOWindow()
    entry = [a for m in w.menuBar().actions() if m.menu() for a in m.menu().actions()
             if a.text().replace("&", "") == "Replace part..."]
    assert len(entry) == 1
    other = os.path.join(ROOT, "data", "models", "Beam_Liao", "Ex1_Beam_design.stl")
    with scratch() as d:
        assert w.open_project(shutil.copy(os.path.join(PROJECTS, "L-bracket.mmto"), d))
        assert w.save_project_as(os.path.join(d, "P", "P.mmto"))
        w.set_design(lambda _: None, show=True)
        with answers(Box.Yes):
            assert w.replace_part(other)                                       # warned (clamp), and accepted
        assert w.project.stl == "part/Ex1_Beam_design.stl" and os.path.isfile(w.project.stl_path())
        assert os.path.normcase(w.stl_geom.file_path) == os.path.normcase(w.project.stl_path())
        assert w._design is None and w.view == "setup" and w.is_dirty()
        assert any(line.startswith("Part replaced") for line in w.message_text.toPlainText().splitlines())


@contextlib.contextmanager
def answered_in_turn(*replies):
    """Questions answered with replies, one after the other, without showing them; the texts asked are kept
    in the list given back. warning() and information() give OK."""
    asked, saved = [], (Box.question, Box.warning, Box.information)
    Box.question = staticmethod(lambda _parent, _title, text, *a, **k: asked.append(text) or replies[len(asked) - 1])
    Box.warning = staticmethod(lambda *a, **k: Box.Ok)
    Box.information = staticmethod(lambda *a, **k: Box.Ok)
    try:
        yield asked
    finally:
        Box.question, Box.warning, Box.information = saved


def test_scale_part():
    """File > Scale part: the window shows the size the part will have; the selections typed for the old size
    (a Box, a Cylinder) are named first, not picked faces or a length in element sizes; a project not in a
    folder of its own is saved as one first and only its part/ is written, every coordinate exactly times the
    factor; scaling again multiplies again, one STL in part/; the extent follows; Cancel changes nothing."""
    from stl import mesh as stl_mesh
    from lsr_mmto.problem import selections as S

    def vectors(path):
        return stl_mesh.Mesh.from_file(path).vectors

    entry = [a for m in G.MMTOWindow().menuBar().actions() if m.menu() for a in m.menu().actions()
             if a.text().replace("&", "") == "Scale part..."]
    assert len(entry) == 1
    shipped = os.path.join(ROOT, "data", "models", "LBracket_BM1", "LBracket_BM1.STL")
    before, original = open(shipped, "rb").read(), vectors(shipped).copy()
    with scratch() as d:
        w = opened("L-bracket", d)                                  # a project file of its own, the STL shipped
        lo, hi = w.part_bounds()
        dialog = G.ScalePartDialog(w, hi - lo, w.length_unit())
        dialog.by.setText("1/1000")
        size = hi - lo
        assert dialog.factor() == 0.001 and dialog.will_be.text() == \
            "The part will be " + " \u00d7 ".join(f"{float(f'{v / 1000:.3g}'):g}" for v in size) + f" {w.length_unit()}"
        dialog.by.setText("0")
        assert dialog.factor() is None and not dialog.buttons.button(QtWidgets.QDialogButtonBox.Ok).isEnabled()
        saved_exec = G.ScalePartDialog.exec_
        G.ScalePartDialog.exec_ = lambda dialog: 0                  # the window's Cancel
        try:
            assert not w.scale_part()
        finally:
            G.ScalePartDialog.exec_ = saved_exec
        w.project.selections.update(pad=S.box([0, 0, 0], [10, 10, 1]), bore=S.cylinder([5, 5, 0], "z", 2.0),
                                    edge=S.near_face("xmin", "1.5h"))
        project_file = open(w.path).read()
        with answered_in_turn(Box.Cancel) as asked:                 # Cancel at the typed selections
            assert not w.scale_part(2.0)
        assert asked == ["These selections were typed for the old size and are not scaled: pad, bore."]
        assert open(w.path).read() == project_file and not os.path.exists(os.path.join(d, "part"))
        with answered_in_turn(Box.Ok, Box.Save) as asked:
            assert w.scale_part(2.0)
        assert len(asked) == 2 and "Save the project there first?" in asked[1] and w.own_folder()
        assert os.listdir(os.path.join(d, "part")) == ["LBracket_BM1.STL"]
        assert np.array_equal(vectors(w.project.stl_path()), original * 2.0)
        assert open(shipped, "rb").read() == before                 # the STL it came from is untouched
        assert np.allclose(w.part_bounds()[1], hi * 2.0) and w.extent.text() != "" and \
            G.without_minus_zero(w.length_format(float(np.max(size * 2))) % (hi[0] * 2)) in w.extent.text()
        assert "Part scaled by 2" in w.message_text.toPlainText().splitlines()
        with answered_in_turn(Box.Ok):
            assert w.scale_part(3)
        with answered_in_turn(Box.Ok):
            assert w.scale_part(0.5)
        assert os.listdir(os.path.join(d, "part")) == ["LBracket_BM1.STL"]
        assert np.array_equal(vectors(w.project.stl_path()), original * 6.0 * 0.5)
        assert open(shipped, "rb").read() == before

    example = os.path.join(PROJECTS, "L-bracket.mmto")              # an example: saved as a project of its own
    example_text, models = open(example).read(), sorted(os.listdir(os.path.dirname(shipped)))
    w = G.MMTOWindow()
    assert w.open_project(example) and w.template
    real = G.ProjectFolderDialog.exec_
    with scratch() as d:
        def fill(dialog):
            dialog.location.setText(d)
            dialog.name.setText("Mine")
            return 1
        G.ProjectFolderDialog.exec_ = fill
        try:
            with answered_in_turn(Box.Save) as asked:               # clamp and corner: nothing typed
                assert w.scale_part(25.4)
        finally:
            G.ProjectFolderDialog.exec_ = real
        assert len(asked) == 1 and not w.template and w.path == os.path.join(d, "Mine", "Mine.mmto")
        assert os.listdir(os.path.join(d, "Mine", "part")) == ["LBracket_BM1.STL"]
        assert np.array_equal(vectors(w.project.stl_path()), original * 25.4)
        assert open(example).read() == example_text and open(shipped, "rb").read() == before
        assert sorted(os.listdir(os.path.dirname(shipped))) == models


def test_start_page():
    """With no project open the start page covers the 3D view, and the side buttons, Run, Export STL and Part /
    Design are greyed with a tip; opening a project hides the page and they work. On a laptop screen
    (1366 x 768) the page stays whole: nothing in it cut short, overlapping or outside the 3D view."""
    w = G.MMTOWindow()
    w.show()
    APP.processEvents()
    page = w.start_page
    assert page.isVisible() and page.geometry().width() >= w.pv_frame.width() - 2
    assert list(page.examples) == ["L-bracket", "U-bracket", "Beam", "Airplane bracket"]
    waiting = list(w.sidebar_buttons.values()) + [w.run_button, w.export_button, w.part_button, w.design_button]
    assert not any(b.isEnabled() for b in waiting)
    assert all(b.toolTip() == G.NO_PROJECT for b in waiting)
    G.apply_style(APP)
    try:
        w.resize(1366, 768)
        for _ in range(5):
            APP.processEvents()
        shown = [x for x in page.findChildren((QtWidgets.QLabel, QtWidgets.QPushButton)) if x.isVisible()]
        boxes = [QtCore.QRect(x.mapTo(page, QtCore.QPoint(0, 0)), x.size()) for x in shown]
        assert len(shown) == 2 + 3 + 4 and all(page.rect().contains(b) for b in boxes)
        assert all(x.height() >= (x.heightForWidth(x.width()) if x.hasHeightForWidth() else x.sizeHint().height())
                   for x in shown), "squashed"
        assert not any(a.intersects(b) for i, a in enumerate(boxes) for b in boxes[i + 1:])
        top = min(b.top() for b in boxes)
        bottom = page.height() - max(b.bottom() for b in boxes)
        assert w.pv_frame.height() < 600 and top > 0 and bottom > 0 and abs(top / (top + bottom) - 0.4) < 0.1
    finally:
        APP.setStyleSheet("")
    assert w.open_project(os.path.join(PROJECTS, "L-bracket.mmto"))
    assert page.isHidden()
    assert all(b.isEnabled() for b in w.sidebar_buttons.values()) and w.run_button.isEnabled()
    assert w.part_button.isEnabled() and not w.export_button.isEnabled() and w.sidebar_buttons["Units"].toolTip() == ""
    w.close()


def test_side_marks():
    """Units, Materials, Supports and loads and Formulation: a tick once set up and the run's check finds nothing
    wrong there, a red mark for an error of that step, plain while not set up (never red then); a warning marks
    nothing; the other buttons have no mark. The marks follow every change."""
    from lsr_mmto.problem.project import Project as P
    marks = lambda w: {n: w.sidebar_buttons[n].property("mark") for n in G.SIDEBAR}       # noqa: E731
    w = G.MMTOWindow()
    assert set(marks(w).values()) <= {"plain", None}
    with scratch() as d:
        assert w.open_project(shutil.copy(os.path.join(PROJECTS, "L-bracket.mmto"), d))
        assert w.project.validate() == [] and w.project.warnings()                # warnings: still ticked
        assert marks(w) == {"Units": "check", "Materials": "check", "Supports and loads": "check",
                            "Formulation": "check", "Settings": None, "Run": None, "Results": None,
                            "Display Options": None}
        w.project.bcs.append({"kind": "fixed", "on": "nowhere", "dofs": "xyz"})
        w.project_changed()                                                    # as Supports and loads says
        assert marks(w)["Supports and loads"] == "error" and marks(w)["Formulation"] == "check"
        w.project.bcs.pop()
        w.project.formulation["constraints"][0]["bound"] = 0.0
        w.project_changed()
        assert marks(w)["Supports and loads"] == "check" and marks(w)["Formulation"] == "error"
        w.project.formulation["constraints"][0]["bound"] = 0.15
        w.project.settings["vae"]["latentDim"] += 1                            # an error of Settings: no mark
        w.project_changed()
        assert any(e.step == "settings" for e in w.project.validate())
        assert all(m in ("check", None) for m in marks(w).values()), marks(w)
        with answers(Box.Discard):
            assert w.open_project(shutil.copy(os.path.join(PROJECTS, "U-bracket.mmto"), d))
        assert w.project.units == "none" and marks(w)["Units"] == "plain"         # not set up yet
        empty = P("empty", stl=w.project.stl, base_dir=w.project.base_dir)
        assert {"no supports", "no loads", "no material database", "no objective"} <= set(empty.validate())
        with answers(Box.Discard):
            w._show(empty, None)
        assert {n: marks(w)[n] for n in G.SIDEBAR_STEPS} == dict.fromkeys(G.SIDEBAR_STEPS, "plain")
        empty.bcs.append({"kind": "fixed", "on": "nowhere", "dofs": "xyz"})       # begun, and wrong: red
        w.project_changed()
        assert marks(w)["Supports and loads"] == "error"


# ---------------------------------------------------------------------- Materials, Formulation, Settings
from gui import formulation_panel, materials_panel, settings_panel  # noqa: E402


def opened(name, d):
    w = G.MMTOWindow()
    assert w.open_project(shutil.copy(os.path.join(PROJECTS, f"{name}.mmto"), d))
    return w


def test_the_materials_window_works_out_the_database_pca_once_and_prints_nothing(monkeypatch, capsys):
    """Opening the Materials window: the PCA of the database (its effective dimension) is worked out once,
    for the recommended latent dimension and the network's check alike, and nothing goes to the terminal.
    It was worked out by each of them and printed by the encoder, once more for each project check."""
    from lsr_mmto.materials import databases
    calls = []
    real = databases.participation_ratio
    monkeypatch.setattr(databases, "participation_ratio", lambda X: (calls.append(X.shape), real(X))[1])
    with scratch() as d:
        w = opened("U-bracket", d)
        databases._READ_CACHE.clear()                                       # as for a database just opened
        calls.clear()
        capsys.readouterr()
        real_exec = QtWidgets.QDialog.exec_
        try:
            QtWidgets.QDialog.exec_ = lambda self: 0
            w.handle_button_click(next(c for c in w.BUTTON_CONFIG if c["name"] == "Materials"))
        finally:
            QtWidgets.QDialog.exec_ = real_exec
        out = capsys.readouterr().out
        assert len(calls) == 1, calls
        assert out == "", out
        w.update_side_marks()                                               # the project's checks: no more
        assert len(calls) == 1, calls


def test_panels_with_no_edits_change_nothing():
    """OK with no edits, on every project and every panel: the project is as it was, to the last digit
    (the U-bracket's mass bound is 0.0035000000000000005)."""
    assert list(G.MMTOWindow().sidebar_buttons) == list(G.SIDEBAR)
    with scratch() as d:
        for name in NAMES:
            w = opened(name, d)
            for cls in (materials_panel.MaterialsPanel, formulation_panel.FormulationPanel, settings_panel.SettingsPanel):
                assert cls(w, w.project).apply(w.project) == [], (name, cls.__name__)
            assert not w.is_dirty(), name
        real = QtWidgets.QDialog.exec_
        try:
            QtWidgets.QDialog.exec_ = lambda self: 1                          # OK, through the sidebar
            for key in ("Materials", "Formulation", "Settings"):
                w.handle_button_click(next(c for c in w.BUTTON_CONFIG if c["name"] == key))
            assert not w.is_dirty()
        finally:
            QtWidgets.QDialog.exec_ = real


def test_cancel_changes_nothing():
    with scratch() as d:
        w = opened("beam", d)
        before = w.project.settings["common"]["maxIterations"]
        real = QtWidgets.QDialog.exec_
        try:
            def edit_then_cancel(self):
                if isinstance(self, settings_panel.SettingsPanel):
                    self.rows[("common", "maxIterations")][0].set(5)
                return 0
            QtWidgets.QDialog.exec_ = edit_then_cancel
            w.open_panel("Settings")
        finally:
            QtWidgets.QDialog.exec_ = real
        assert w.project.settings["common"]["maxIterations"] == before != 5 and not w.is_dirty()


def test_settings_panel_covers_every_setting():
    from lsr_mmto.problem import settings
    from lsr_mmto.problem.settings import DEFAULT_SETTINGS, VAE_PARAMS
    keys = settings_panel.all_setting_keys()
    want = {(s, k) for s, v in DEFAULT_SETTINGS.items() for k in v} | {("vae", k) for k in VAE_PARAMS} | \
        {("mesh", "n_dof"), ("mesh", "single_layer")}
    assert len(keys) == len(set(keys)) and set(keys) == want, set(keys) ^ want
    assert all(settings.BY_KEY[k].label and settings.BY_KEY[k].help for _, k in keys)


def test_settings_panel_edits():
    with scratch() as d:
        w = opened("L-bracket", d)
        s = settings_panel.SettingsPanel(w, w.project)
        row, label, reset = s.rows[("common", "maxIterations")]
        assert not label.font().bold() and not reset.isVisibleTo(s)                # reset: only on a change
        row.set(5)
        s._mark("common", "maxIterations")
        assert label.font().bold() and reset.isVisibleTo(s)
        assert s.rows[("group", "Advanced")][0].text() == "Advanced"
        s.rows[("advanced", "timeLimit")][0].set(60)
        s._mark("advanced", "timeLimit")
        assert s.rows[("group", "Advanced")][0].text() == "Advanced  (1 changed)"
        s.rows[("common", "material_filter_radius")][0].none.setChecked(True)  # same as the density filter
        s.rows[("mesh", "n_dof")][0].widget.setText("abc")
        assert s.problems() == ["Mesh dofs is not a number"]
        s.reset("mesh", "n_dof")
        assert s.problems() == []
        lines = s.apply(w.project)
        assert w.project.settings["common"]["maxIterations"] == 5 and w.project.n_dof == 30000
        assert w.project.settings["advanced"]["timeLimit"] == 60
        assert w.project.settings["common"]["material_filter_radius"] is None
        assert lines and "Max iterations 300 -> 5" in lines[0]
        assert w.is_dirty()


def test_formulation_panel():
    with scratch() as d:
        w = opened("L-bracket", d)                                            # BM1: no Density, no Yield_Strength
        f = formulation_panel.FormulationPanel(w, w.project)
        model = f.objective.model()
        enabled = {model.item(i).text(): model.item(i).isEnabled() for i in range(model.rowCount())}
        objectives = {q.shown: q for q in f.quantities.values() if q.objective}   # this physics' objectives
        assert set(enabled) == set(objectives)
        for shown_name, q in objectives.items():                               # greyed exactly when a column is missing
            assert enabled[shown_name] == all(c in f.columns for c in q.needs), shown_name
        assert enabled["Compliance"] and enabled["Stress (p-norm)"]
        assert not enabled["Mass"] and not enabled["Failure factor (p-norm)"]
        assert f.reason("Mass") == "needs a Density column in the database"
        assert f.reason("Cost") == "needs a Density and a Cost column in the database"
        f.rows[2]["widgets"]["bound"].set_value(0.2)
        row = f.add_row(None)
        row["widgets"]["quantity"].setCurrentText("Material volume fraction")
        row["widgets"]["material"].setCurrentIndex(1)
        row["widgets"]["bound"].set_value(0.05)
        f.remove_row(f.rows[0])
        assert f.problems() == [] and f.tau_box.isVisibleTo(f) and f.tau.text() == "0.5"
        f.tau.setText("0.3")                                                    # one tau, written into all of them
        f.apply(w.project)
        assert [(c["expression"], c["bound"]) for c in w.project.formulation["constraints"]] == [
            ("MaterialVolumeFraction(1, tau=0.3)", 0.15), ("MaterialVolumeFraction(2, tau=0.3)", 0.2),
            ("MaterialVolumeFraction(1, tau=0.3)", 0.05)]
        assert w.project.validate() == [] and w.is_dirty()
        f = formulation_panel.FormulationPanel(w, w.project)
        f.add_row(None)
        f.tau.setText("0")
        assert f.problems() == ["constraint 4: choose a quantity", "tau must be a number greater than 0"]
        w.project.formulation["constraints"][0]["expression"] = "MaterialVolumeFraction(1, tau=0.7)"   # by hand
        assert "the material volume fraction constraints use different tau (0.3, 0.7)" in w.project.validate()[0]
    with scratch() as d:
        w = opened("Airplane bracket", d)
        f = formulation_panel.FormulationPanel(w, w.project)
        model = f.rows[0]["widgets"]["quantity"].model()
        assert all(model.item(i).isEnabled() for i in range(model.rowCount()))
        assert [r["widgets"]["unit"].text() for r in f.rows] == ["t", "USD"]   # the bounds 4e7 and 7.5e8, in mmKS
        assert not f.tau_box.isVisibleTo(f) and not f.rows[0]["widgets"]["material"].isVisibleTo(f)  # Mass, Cost
        f.rows[0]["widgets"]["quantity"].setCurrentText("Material volume fraction")
        assert f.tau_box.isVisibleTo(f) and f.rows[0]["widgets"]["material"].isVisibleTo(f)
        f.remove_row(f.rows[0])
        assert not f.tau_box.isVisibleTo(f)
    with scratch() as d:
        w = opened("U-bracket", d)                                            # bound 0.0035000000000000005
        f = formulation_panel.FormulationPanel(w, w.project)
        bound = f.rows[0]["widgets"]["bound"]
        assert bound.text() == "0.0035" and bound.value() == 0.0035000000000000005
        assert f.apply(w.project) == []
        bound.setText("0.0035")                                                # retyped as shown: kept exactly
        assert bound.value() == 0.0035000000000000005
        bound.setText("0.00350")                                               # a typed change: replaced
        assert bound.value() == 0.0035
    assert formulation_panel.bound_unit("Compliance", "mmKS") == "N*mm" and formulation_panel.bound_unit("Mass", "none") == \
        "no unit system declared"


def test_formulation_mass_fraction_row():
    """A Mass fraction row reads 'Mass fraction of [material] <= [limit] of a solid part in that material':
    its choices are the materials' labels, then the heaviest and the lightest; it is saved with the material
    word, opens again with it chosen, and is named in words in the messages. A material volume fraction row
    keeps only the materials."""
    with scratch() as d:
        w = opened("U-bracket", d)                                            # 6Materials: labels A to F
        f = formulation_panel.FormulationPanel(w, w.project)
        row = f.add_row(None)
        wd = row["widgets"]
        wd["quantity"].setCurrentText("Mass fraction")
        choices = [wd["material"].itemText(i) for i in range(wd["material"].count())]
        assert choices == list("ABCDEF") + ["the heaviest", "the lightest"], choices
        assert wd["of"].isVisibleTo(f) and wd["of"].text() == "of" and wd["material"].isVisibleTo(f)
        assert wd["unit"].text() == "of a solid part in that material"
        assert wd["info"].isVisibleTo(f) and "made solid in one material" in wd["info"].toolTip()
        wd["material"].setCurrentText("the lightest")
        wd["bound"].set_value(0.4)
        lines = f.apply(w.project)
        assert w.project.formulation["constraints"][-1] == {"expression": "MassFraction(lightest)", "op": "<=",
                                                             "bound": 0.4}
        assert "Constraint: Mass fraction of the lightest <= 0.4 of a solid part in that material" in lines, lines
        assert w.project.validate() == []
        f = formulation_panel.FormulationPanel(w, w.project)
        assert f.rows[-1]["widgets"]["material"].currentText() == "the lightest"
        wd = f.rows[-1]["widgets"]
        wd["quantity"].setCurrentText("Material volume fraction")             # takes no words: only materials
        assert [wd["material"].itemText(i) for i in range(wd["material"].count())] == list("ABCDEF")
        wd["quantity"].setCurrentText("Mass")
        assert not wd["of"].isVisibleTo(f) and not wd["material"].isVisibleTo(f)


def test_a_new_constraint_row_starts_at_its_quantity_limit():
    """A new row starts at the limit its quantity declares (start_bound): 0.4 for the fractions, 1 for the
    failure factor, and for Mass 0.4 times the part solid in the lightest material (far above the low-mass
    warning); a quantity that declares none starts at 0 as before. A limit typed by the user is kept when
    the quantity changes, and an existing row keeps its own."""
    from lsr_mmto.formulation.quantities import part_volume
    from lsr_mmto.materials.databases import database_table
    with scratch() as d:
        w = opened("U-bracket", d)
        _, attributes, _, values = database_table(w.project.database_paths()[0])
        lightest = values[:, attributes.index("Density")].min() * part_volume(w.project.stl_path())
        f = formulation_panel.FormulationPanel(w, w.project)
        before = f.rows[0]["widgets"]["bound"].value()
        row = f.add_row(None)["widgets"]
        assert row["bound"].value() == 0.0                                    # no quantity yet
        for shown, start in (("Volume fraction", 0.4), ("Material volume fraction", 0.4), ("Mass fraction", 0.4),
                             ("Failure factor (p-norm)", 1.0), ("Mass", 0.4 * lightest), ("Compliance", 0.0)):
            row["quantity"].setCurrentText(shown)
            assert abs(row["bound"].value() - start) <= 1e-12 * max(start, 1), (shown, row["bound"].value(), start)
        row["quantity"].setCurrentText("Mass")
        assert w.project.formulation["constraints"][0]["expression"] == "Mass()"
        row["bound"].setText("0.002")                                          # typed: kept from now on
        row["quantity"].setCurrentText("Volume fraction")
        assert row["bound"].value() == 0.002
        f.rows[0]["widgets"]["quantity"].setCurrentText("Volume fraction")    # an existing row keeps its limit
        assert f.rows[0]["widgets"]["bound"].value() == before
        f.add_row(None)["widgets"]["quantity"].setCurrentText("Mass")
        f.apply(w.project)
        assert not any("lightest material" in x for x in w.project.warnings())


def test_materials_panel():
    from lsr_mmto.materials.databases import DATABASE_DIR
    with scratch() as d:
        w = opened("L-bracket", d)
        m = materials_panel.MaterialsPanel(w, w.project)
        assert m.table.rowCount() == 3 and m.table.item(0, 0).text() == "Mat1_Red"
        assert "latent dimension 2 recommended" in m.report.toPlainText()
        m.combo.setCurrentText("CustomAlloys20_3D")
        assert m.table.rowCount() == 20 and m.table.item(0, 0).text() == "A"
        lines = m.apply(w.project)
        vae = w.project.settings["vae"]
        assert w.project.database == "CustomAlloys20_3D" and vae["latentDim"] == 3 and vae["numEpochs"] == 100000
        assert lines[0] == "Database: CustomAlloys20_3D" and "Latent dimension 2 -> 3" in lines[1]
        m = materials_panel.MaterialsPanel(w, w.project)
        m.set_database(os.path.join(ROOT, DATABASE_DIR, "BM1_3Mats_2Attr.xlsx"))     # a shipped file: by name
        assert m.project.database == "BM1_3Mats_2Attr"
        own = shutil.copy(os.path.join(ROOT, DATABASE_DIR, "BM1_3Mats_2Attr.xlsx"), os.path.join(d, "mine.xlsx"))
        m.set_database(own)                                                    # any other: relative to the project
        assert m.project.database == "mine.xlsx"
        m.apply(w.project)
        assert w.project.settings["vae"]["latentDim"] == 2 and w.project.validate() == []
        shutil.copy(os.path.join(ROOT, DATABASE_DIR, "BM1_3Mats_2Attr.nt"), os.path.join(d, "trained.nt"))
        shutil.copy(own, os.path.join(d, "trained.xlsx"))                      # its own network: width 50
        w.project.settings["vae"]["vae_hiddenDim"] = 300
        m = materials_panel.MaterialsPanel(w, w.project)
        m.set_database(os.path.join(d, "trained.xlsx"))
        lines = m.apply(w.project)
        assert lines[-1] == "Network settings set to match the trained network: hidden layer width 50", lines
        assert w.project.settings["vae"]["vae_hiddenDim"] == 50 and w.project.validate() == []


def test_database_editor():
    """New database and Edit: one row per material (name, optional shown-as, colour, numbers), the units under
    the headers, bad cells marked and Save off until none; a block pasted from Excel; always saved in the
    project's folder (no folder to choose); a shipped database saved as a copy; a network set aside when
    the numbers change, kept when only a colour does."""
    from lsr_mmto.materials.databases import DATABASE_DIR, check_database, database_table, material_display
    from gui import database_editor as DE

    def editing(m, action, fill):
        real = DE.DatabaseEditor.exec_
        try:
            DE.DatabaseEditor.exec_ = fill
            return action()
        finally:
            DE.DatabaseEditor.exec_ = real

    def typed(e, rows):
        for r, row in enumerate(rows, start=1):
            while e.table.rowCount() <= r:
                e.add_material()
            for c, text in enumerate(row):
                if c != 2:
                    e.table.item(r, c).setText(text)

    with scratch() as d:
        w = opened("L-bracket", d)
        m = materials_panel.MaterialsPanel(w, w.project)
        with answers():
            assert m.new_database() is None                                   # not in a folder of its own yet
        assert w.save_project_as(os.path.join(d, "P", "P.mmto")) and w.own_folder()
        folder = os.path.join(d, "P", "materials")
        m = materials_panel.MaterialsPanel(w, w.project)
        seen = {}

        def new(e):
            assert e.where.text() == f"Saved in {folder}"
            assert not [b for b in e.findChildren(QtWidgets.QPushButton) if "Browse" in b.text()]
            assert [e.table.horizontalHeaderItem(c).text() for c in range(e.table.columnCount())] == \
                ["Name", "Shown as", "Colour", "Density", "Young's modulus", "Poisson's ratio"]
            assert [e.table.cellWidget(0, c).currentText() for c in (3, 4, 5)] == ["t/mm^3", "MPa", "-"]
            assert all(e.table.item(r, 0).text() == "" for r in (1, 2, 3))      # no placeholder names
            typed(e, [["Steel", "", "", "7.85e-9", "200000", "0.29"], ["Copper", "Cu", "", "8.96e-9", "117000", "0.6"],
                      ["Aluminum", "", "", "2.7e-9", "68900", "0.33"]])
            seen["bad"] = e.check()
            assert not e.save_button.isEnabled() and e.table.item(2, 5).background().color() == DE.BAD
            e.table.item(2, 5).setText("0.34")
            e.add_material()                                                    # an empty row to paste into
            e.table.setCurrentCell(4, 0)
            e.table.paste("Titanium\t\t\t4.43e-9\t113800\t0.34\nIron\t\t\t7.1e-9\t169000\t0.29")
            e._set_colour(1, "#123456")
            seen["good"] = e.check()
            return 1
        path = editing(m, m.new_database, new)
        e = DE.DatabaseEditor(w, folder, unit_system=w.project.units)          # Add / Remove property
        e._fill_property_menu()
        from lsr_mmto.materials.databases import ATTRIBUTES, SHOWN
        offered = [a.text() for a in e.add_property_menu.actions()]
        assert offered == [SHOWN[k] for k in ATTRIBUTES if k not in e.attributes]   # every property not there yet
        assert "Yield strength" in offered and "Cost" in offered and "Density" not in offered
        e.add_property("Cost")
        assert e.attributes == ["Density", "Youngs_Modulus", "Poissons_Ratio", "Cost"]
        assert e.table.cellWidget(0, 6).currentText() == "USD/t"
        e.table.item(1, 6).setText("1000")
        with answers(Box.No):
            assert not e.remove_property("Cost")                              # it holds a number: asked
        with answers(Box.Yes):
            assert e.remove_property("Cost") and e.attributes == ["Density", "Youngs_Modulus", "Poissons_Ratio"]
        e.table.setCurrentCell(1, 3)
        assert e.remove_property() and e.attributes == ["Youngs_Modulus", "Poissons_Ratio"]
        for key in ("Youngs_Modulus", "Poissons_Ratio"):
            e.remove_property(key)
        assert "Add at least one property" in e.check()
        assert seen["bad"] == ["Copper: Poisson's ratio: -1 < value < 0.5"] and seen["good"] == []
        assert path == os.path.join(folder, "my_materials.xlsx") and m.project.database == "materials/my_materials.xlsx"
        assert check_database(path) == ([], [])
        names, attributes, units, values = database_table(path)
        assert names == ["Steel", "Copper", "Aluminum", "Titanium", "Iron"] and units == ["t/mm^3", "MPa", "-"]
        colours, labels = material_display(path)
        assert colours[0] == "#123456" and labels[:2] == ["Steel", "Cu"]

        network = os.path.splitext(path)[0] + ".nt"
        open(network, "wb").write(b"trained")
        editing(m, m.edit_database, lambda e: (e._set_colour(2, "#ff8800"), 1)[1])   # a colour only
        assert os.path.isfile(network) and material_display(path)[0][1] == "#ff8800"

        def edit(e):
            assert e.name.isReadOnly() and e.where.text() == path
            assert e.table.item(2, 1).text() == "Cu" and e.table.item(1, 1).text() == ""
            e.table.item(1, 4).setText("210000")
            return 1
        editing(m, m.edit_database, edit)
        assert not os.path.exists(network) and open(network[:-3] + ".previous.nt", "rb").read() == b"trained"
        assert "set aside as my_materials.previous.nt" in m.report.toPlainText()
        assert database_table(path)[3][0, 1] == 210000

        shipped = os.path.join(ROOT, DATABASE_DIR, "BM1_3Mats_2Attr.xlsx")
        before = open(shipped, "rb").read()
        m.set_database(shipped)
        copy = editing(m, m.edit_database, lambda e: (e.table.item(1, 1).setText("Red"), 1)[1])
        assert copy == os.path.join(folder, "BM1_3Mats_2Attr copy.xlsx") and open(shipped, "rb").read() == before
        assert m.project.database == "materials/BM1_3Mats_2Attr copy.xlsx"
        assert material_display(copy)[1] == ["Red", "Mat2_Blue", "Mate3_Green"]


def test_import_database():
    """Import...: a table of materials made elsewhere (here a CSV) becomes a database. Columns named as a
    property LSR MMTO knows are read (a unit in brackets after the name too), the others are listed in one
    message, and the editor opens with what was read, a value that is not a number shown as a cell to
    correct; once saved in the project's folder the project uses it."""
    from lsr_mmto.materials.databases import check_database, database_table
    from gui import database_editor as DE
    with scratch() as d:
        w = opened("L-bracket", d)
        assert w.save_project_as(os.path.join(d, "P", "P.mmto")) and w.own_folder()
        table = os.path.join(d, "alloys.csv")
        open(table, "w").write("Alloy,Density [kg/m^3],E (GPa),Price,E0\n"
                               "Al 6061,2700,69,2.1,1\nTi-6Al-4V,4430,114,30,2\nSteel 304,8000,x,3,3\n")
        m = materials_panel.MaterialsPanel(w, w.project)
        said, seen = [], {}
        real_exec, real_info = DE.DatabaseEditor.exec_, QtWidgets.QMessageBox.information

        def review(e):
            seen["heads"] = [e.table.horizontalHeaderItem(c).text() for c in range(e.table.columnCount())]
            seen["units"] = [e.table.cellWidget(0, c).currentText() for c in (3, 4)]
            seen["problems"] = e.check()
            assert not e.save_button.isEnabled()
            e.table.item(3, 4).setText("193")                                # the value that was not a number
            assert e.check() == []
            return 1
        try:
            DE.DatabaseEditor.exec_ = review
            QtWidgets.QMessageBox.information = staticmethod(lambda _, title, text: said.append(text))
            path = m.import_database(table)
        finally:
            DE.DatabaseEditor.exec_, QtWidgets.QMessageBox.information = real_exec, real_info
        assert len(said) == 1 and "Not imported, as LSR MMTO has no property of that name: Price, E0." in said[0]
        assert seen["heads"] == ["Name", "Shown as", "Colour", "Density", "Young's modulus"]
        assert seen["units"] == ["kg/m^3", "GPa"] and seen["problems"] == ["Steel 304: Young's modulus: a number"]
        assert path == os.path.join(d, "P", "materials", "alloys.xlsx")
        assert m.project.database == "materials/alloys.xlsx" and check_database(path)[0] == []
        names, attributes, units, values = database_table(path)
        assert names == ["Al 6061", "Ti-6Al-4V", "Steel 304"] and attributes == ["Density", "Youngs_Modulus"]
        assert values[2].tolist() == [8000.0, 193.0]
        open(table, "w").write("Alloy,Price\nA,1\nB,2\n")                    # nothing LSR MMTO knows
        with answers():
            assert m.import_database(table) is None


def test_edit_a_database_with_an_unknown_column():
    """A database with a column LSR MMTO does not know (E0): the check says so in plain words, and Edit...
    opens it with that column left out and named in a note, and saves it without the column."""
    import openpyxl
    from lsr_mmto.materials.databases import check_database, database_table, write_database
    from gui import database_editor as DE
    with scratch() as d:
        w = opened("L-bracket", d)
        assert w.save_project_as(os.path.join(d, "P", "P.mmto")) and w.own_folder()
        path = write_database(os.path.join(d, "P", "materials", "extra.xlsx"), ["Density", "Youngs_Modulus"],
                              ["t/mm^3", "MPa"], ["A", "B"], [None, None], ["#ff0000", "#0000ff"],
                              [[7.8e-9, 200000.0], [2.7e-9, 70000.0]])
        book = openpyxl.load_workbook(path)
        sheet = book.worksheets[0]
        for row, value in enumerate(("E0", "-", 1.0, 2.0), start=1):
            sheet.cell(row=row, column=4, value=value)
        book.save(path)
        errors = check_database(path)[0]
        assert len(errors) == 1 and errors[0].startswith("D1: 'E0' is not a property LSR MMTO knows (Density, "
                                                         "Young's modulus, "), errors
        assert errors[0].endswith("Rename the column to one of them, or use Import..., which leaves out the "
                                  "columns it does not know.") and "lsr_mmto" not in errors[0]
        m = materials_panel.MaterialsPanel(w, w.project)
        m.set_database(path)
        seen = {}

        def look(e):
            seen["heads"] = [e.table.horizontalHeaderItem(c).text() for c in range(e.table.columnCount())]
            seen["note"] = e.left_out_note.text() if not e.left_out_note.isHidden() else ""
            return 1
        real = DE.DatabaseEditor.exec_
        try:
            DE.DatabaseEditor.exec_ = look
            assert m.edit_database() == path
        finally:
            DE.DatabaseEditor.exec_ = real
        assert seen["heads"] == ["Name", "Shown as", "Colour", "Density", "Young's modulus"]
        assert seen["note"] == ("Left out, as LSR MMTO has no property of that name: E0. Save writes the "
                                "database without them.")
        assert check_database(path)[0] == [] and database_table(path)[1] == ["Density", "Youngs_Modulus"]


def test_error_in_the_app_keeps_it_running():
    """An error in what a button does is shown and written to the log; the app keeps running."""
    shown = []
    real = Box.critical
    with scratch() as d:
        w = opened("L-bracket", d)
        G.install_error_handler(w)
        Box.critical = staticmethod(lambda *a, **k: shown.append(a[2]) or Box.Ok)
        try:
            QtCore.QTimer.singleShot(0, lambda: 1 / 0)                            # an error inside the app
            assert _wait(lambda: shown, 10)
        finally:
            Box.critical = real
            sys.excepthook = sys.__excepthook__
        assert "ZeroDivisionError" in shown[0] and "still running" in shown[0]
        assert "Error: ZeroDivisionError" in w.message_text.toPlainText()
        assert w.isEnabled() and w.save_project()                             # still working


def test_vae_tab():
    """With a trained network in use the VAE settings are its own and greyed out; the tab says which
    network the database has and shows the VAE checks as you edit. No "?" button on our panels."""
    from lsr_mmto.materials.databases import DATABASE_DIR
    with scratch() as d:
        w = opened("L-bracket", d)
        s = settings_panel.SettingsPanel(w, w.project)
        assert not (s.windowFlags() & QtCore.Qt.WindowContextHelpButtonHint)
        assert s.vae_status.text() == "Network: BM1_3Mats_2Attr.nt, shipped." and s.vae_checks.text() == ""
        latent = s.rows[("vae", "latentDim")][0]
        assert not latent.widget.isEnabled() and not s.rows[("vae", "saveNet")][0].widget.isEnabled()
        assert s.apply(w.project) == []
        s.rows[("vae", "use_pretrained_vae")][0].widget.setChecked(False)     # train: a shipped database refuses
        assert latent.widget.isEnabled() and "save it over the shipped" in s.vae_checks.text()
        latent.widget.setText("4")
        assert "latentDim is 4; the database's effective dimension" in s.vae_checks.text()

        w.project.settings["vae"]["latentDim"] = 4                            # a file that does not fit its network
        s = settings_panel.SettingsPanel(w, w.project)
        assert s.rows[("vae", "latentDim")][0].get() == 2
        assert s.apply(w.project) == ["Settings: Latent dimension 4 -> 2",
                                      "Latent dimension: set to the trained network's"]

        shutil.copy(os.path.join(ROOT, DATABASE_DIR, "BM1_3Mats_2Attr.xlsx"), os.path.join(d, "ggd.xlsx"))
        w.project.database = "ggd.xlsx"                                       # one of yours, no network yet
        s = settings_panel.SettingsPanel(w, w.project)
        assert s.vae_status.text() == "Network: ggd.nt, none yet. The first run trains ggd.nt."
        use = s.rows[("vae", "use_pretrained_vae")][0].widget
        assert not use.isEnabled() and not use.isChecked()                  # shown unticked: nothing to use yet
        assert s.rows[("vae", "latentDim")][0].widget.isEnabled()
        assert s.apply(w.project) == [] and w.project.settings["vae"]["use_pretrained_vae"] is True   # kept
    f = formulation_panel.FormulationPanel(G.MMTOWindow(), Project.load(os.path.join(PROJECTS, "beam.mmto")))
    f.rows[0]["widgets"]["bound"].setText("abc")
    assert f.problems() == ["constraint 1: the bound is not a number"]


# ---------------------------------------------------------------------- supports and loads
from gui import supports_and_loads_panel as SP  # noqa: E402


def test_code_line():
    """A code line gives the same selection as a script; anything that is not a selection is refused."""
    for name in NAMES:
        for s in Project.load(os.path.join(PROJECTS, f"{name}.mmto")).selections.values():
            assert SP.parse_selection(repr(s)).to_dict() == s.to_dict(), s
    assert SP.parse_selection('rows(face="xmax", axis="y", n=8, end="max")').to_dict() == \
        {"type": "rows", "face": "xmax", "axis": "y", "n": 8, "end": "max"}
    for bad, says in (('__import__("os").system("dir")', "is not a selection type"),
                      ('box(lo=[0, 0, 0], hi=open("x"))', "open is not a selection type"),
                      ('face(**{"side": "xmin"})', "** arguments"), ('function("f.py:g")', "not a selection type"),
                      ("2 + 2", "write a selection"), ("face(", "not a selection"),
                      ("sphere([0, 0, 0], radius=r)", "only numbers, text, lists and selections"),
                      ("sphere([0, 0, 0], radius=math.pi)", "only numbers, text, lists and selections"),
                      ("sphere([0, 0, 0], radius=abs(-2))", "abs is not a selection type"),
                      ("sphere([0, 0, 0], radius=9**9**9)", "only numbers, text, lists and selections")):
        try:
            SP.parse_selection(bad)
        except ValueError as e:
            assert says in str(e), (bad, e)
        else:
            raise AssertionError(f"accepted {bad}")


def test_arithmetic_in_number_boxes():
    """Every number of the Selection form, the Code line and the support or load form takes plain
    arithmetic (38/2, 0.5*76, 2*pi*5, inside a point too); the selection or value keeps the number."""
    import math
    from lsr_mmto.problem import selections as S
    from gui.panel_widgets import number
    assert number("38/2") == 19 and number("0.5*76") == 38 and number("2*pi*5") == 10 * math.pi
    assert number("-(1 + 2)**2") == -9 and number(" 4 ") == 4
    for bad in ("x", "pi()", "math.pi", "1/0", "[1, 2]", "'5'", "True", "9**9**9", "1e999", ""):
        try:
            number(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad!r}")
    assert SP.parse_selection("sphere(center=[0, 38/2, 0], radius=2*pi*5)").to_dict() == \
        S.sphere([0, 19.0, 0], 10 * math.pi).to_dict()
    assert SP.parse_selection("plane('y', 76/2, limits={'x': [0, 80/2]})").to_dict() == \
        S.plane("y", 38.0, limits={"x": [0, 40.0]}).to_dict()
    assert SP.RegionDialog._value("At", "38/2") == 19 and SP.RegionDialog._value("Point", "0, 38/2, 0") == [0, 19, 0]
    assert SP.RegionDialog._value("Radius", "1.5h") == "1.5h" and SP.RegionDialog._value("Side", "xmax") == "xmax"
    with scratch() as d:
        w = opened("L-bracket", d)
        r = SP.RegionDialog(w, w.project)
        r.type.setCurrentText("Cylinder")
        r.inputs["center"].setText("52, 228/2, 1")
        r.inputs["axis"].setCurrentText("x")
        r.inputs["radius"].setText("10/2")
        r.inputs["along"][0].setText("44/2")
        r.inputs["along"][1].setText("61")
        assert r.selection().to_dict() == S.cylinder([52, 114.0, 1], "x", 5.0, along=[-30.0, 9]).to_dict()
        r.tabs.setCurrentIndex(1)                                             # the Code line keeps the numbers
        assert "114.0" in r.code.text() and "/" not in r.code.text()
        r.code.setText("box(lo=[0, 0, 0], hi=[200/10, 2*10, 1])")
        assert r.problems() == [] and r.selection().to_dict() == S.box([0, 0, 0], [20.0, 20, 1]).to_dict()
        r.code.setText("sphere([0, 0, 0], radius='2 + 1h')")                # a length in element sizes
        assert r.problems() == [] and r.selection().radius == "2 + 1h"
        r.tabs.setCurrentIndex(0)
        r.type.setCurrentText("Sphere")
        r.inputs["center"].setText("0, 0, 0")
        for typed, kept in (("2 + 2", 4), ("5.64 + 1h", "5.64 + 1h"), ("1.5h", "1.5h"), ("38/2 + 0.5 h", "38/2 + 0.5 h")):
            r.inputs["radius"].setText(typed)
            assert r.selection().radius == kept, typed
        r.inputs["radius"].setText("5 + x")
        assert r.problems() == ["Radius: '5 + x' is not a value; write e.g. 2.5 or 0, 38, 0, or a length in "
                                "element sizes h such as 5 + 1h"]
        b = SP.BCDialog(w, w.project)
        b.kind.setCurrentText("Force")
        b.inputs["total"][1][1].setText("-100/4")
        assert b.problems() == [] and b.bc()["total"] == [0.0, -25.0, 0.0]
        b.inputs["total"][1][1].setText("-100/x")
        assert b.problems() == ["Total force is not a number"]


def test_region_and_bc_forms():
    from lsr_mmto.problem import selections as S
    with scratch() as d:
        w = opened("L-bracket", d)
        r = SP.RegionDialog(w, w.project)
        shown = [r.type.itemText(i) for i in range(r.type.count())]
        assert "Picked faces" not in shown and "Rows on a face" in shown     # made with the mouse: not offered
        r.type.setCurrentText("Rows on a face")                              # fixed choices: dropdowns
        assert r.inputs["face"].count() == 6 and not r.inputs["face"].isEditable()
        assert r.form.labelForField(r.inputs["n"]).text() == "Number of rows"
        r.inputs["face"].setCurrentText("Right (x max)")
        r.inputs["axis"].setCurrentText("y")
        r.inputs["n"].setText("8")
        assert r.selection().to_dict() == S.rows("xmax", "y", 8).to_dict()
        r.type.setCurrentText("Closest node")
        assert isinstance(r.inputs["boundary"], QtWidgets.QCheckBox)
        r.type.setCurrentText("Cylinder")                                    # the ends: both or neither
        r.inputs["center"].setText("52, 114, 1")
        r.inputs["radius"].setText("5")
        assert r.selection().along is None
        assert r.form.labelForField(r._along_rows[0]).text() == "x from"      # axis x: coordinates
        r.inputs["along"][0].setText("22")
        assert r.problems() == ["the two ends: give both, or leave both empty"]
        r.inputs["along"][1].setText("61")
        assert r.selection().along == [-30.0, 9.0]                            # x 22 to 61, from x = 52
        assert SP.describe(r.selection()).endswith("x from 22 to 61")
        r.inputs["axis"].setCurrentText("1, 1, 0")                            # tilted: distances
        assert r.form.labelForField(r._along_rows[0]).text() == "From (distance)"
        assert r.selection().along == [22.0, 61.0]
        r.type.setCurrentText("Box")
        assert r.inputs["pad"].currentText() == "Automatic" and r.inputs["pad"].isEditable()
        r.inputs["lo"].setText("0, 38, 0")                                   # a point without brackets
        r.inputs["hi"].setText("[2, 40, 1]")
        assert r.problems() == [] and r.selection().to_dict() == S.box([0, 38, 0], [2, 40, 1]).to_dict()
        r.tabs.setCurrentIndex(1)                                             # the form, as a code line
        assert r.code.text() == "box(lo=[0.0, 38.0, 0.0], hi=[2.0, 40.0, 1.0], pad='auto')"
        r.name.setText("clamp")
        assert r.problems() == ["there is already a selection named clamp"]
        for name in NAMES:                                                    # every existing region, on the form
            p = Project.load(os.path.join(PROJECTS, f"{name}.mmto"))
            for k, s in p.selections.items():
                e = SP.RegionDialog(w, p, k)
                got = e.from_form() if e.tabs.currentIndex() == 0 else e.selection()
                assert got.to_dict() == s.to_dict(), (name, k, got, s)
        for name in NAMES:                                                    # the forms with no edits change nothing
            p = Project.load(os.path.join(PROJECTS, f"{name}.mmto"))
            for bc in p.bcs:
                assert SP.BCDialog(w, p, bc).bc() == bc, bc
        b = SP.BCDialog(w, w.project)
        b.kind.setCurrentText("Pressure")
        b.inputs["direction"][1][1].set_value(-1.0)
        b.inputs["pressure"][1].set_value(75)
        assert b.problems() == [] and b.bc() == {"kind": "traction", "on": "clamp", "pressure": 75.0,
                                                 "direction": [0.0, -1.0, 0.0], "area": "bbox"}
        assert SP.field_unit("stress", "mmKS") == "MPa" and SP.field_unit("area", "mmKS") == "mm^2"
        texts = [SP.describe(s) for s in w.project.selections.values()]      # the tables, in words
        assert texts[1] == "Rows on a face: Face Right (x max), Counted along y, Number of rows 8", texts
        assert SP.bc_values(w.project.bcs[0]) == "Held in x, y, z"
        code_talk = ("PyTO", "Python", "pad", "bbox", "inner_radius", "1e-12", "(t0", "xmax", "None")
        for i in range(r.type.count()):                                      # every region type, every label
            r.type.setCurrentIndex(i)
            for label in r.tabs.widget(0).findChildren(QtWidgets.QLabel):
                assert not any(x in label.text() for x in code_talk), label.text()
        for k in range(b.kind.count()):                                       # every support and load
            b.kind.setCurrentIndex(k)
            for label in b.findChildren(QtWidgets.QLabel):
                assert not any(x in label.text() for x in code_talk), label.text()


def test_supports_panel():
    with scratch() as d:
        w = opened("L-bracket", d)
        w.show()
        panel = w.open_supports_and_loads_panel()
        panel.preview()
        mesh = w.preview_mesh()
        _, nodes_of = w.project.supports_and_loads(mesh)
        assert panel.counts == {k: len(v) for k, v in nodes_of.items()} == {"clamp": 114, "corner": 16}
        assert "The supports hold the part against the loads." in panel.report.text()
        with answers():
            assert not panel.delete_region("clamp") and "clamp" in w.project.selections   # used by fixed 1
        panel.rename("corner", "tip")
        assert w.project.bcs[1]["on"] == "tip" and "tip" in w.project.selections
        panel.delete_bc(0)                                                    # no support: free to move
        panel.preview()
        assert "free to move or rotate" in panel.report.text()
        assert panel.delete_region("clamp") and "clamp" not in w.project.selections
        panel.set_keep_solid("tip", True)
        assert w.project.keep_solid_names == ["tip"] and w.is_dirty()


def test_picks_are_adopted():
    """An Apply in PyTO-AD's Structural Loads window adds to the project what loading a file with the same
    picks adds; a torque is refused; PyTO-AD's lists are left empty."""
    picks = {"constraint_data": [{"type": "Fixed X", "triangles": [12, 13]}],
             "force_data": [{"type": "force_xyz", "triangles": [], "points": [[0.0, 0.0, 0.0]], "regions": [],
                             "force_x": 0.0, "force_y": -1.0, "force_z": 0.0, "load_set": 0}]}
    with scratch() as d:
        w = opened("L-bracket", d)
        w.show()
        expected = w.project.copy()
        expected.adopt_picks(json.loads(json.dumps(picks)))
        w.open_structural_loads_window()
        w.constraint_data = list(picks["constraint_data"])
        w.force_data = list(picks["force_data"]) + [{"type": "torque", "triangles": [1], "torque": 1.0}]
        w.update_LivVar("structural_loads.forces_applied", True)
        for _ in range(5):
            APP.processEvents()
        assert w.project.to_dict() == expected.to_dict()
        assert w.constraint_data == [] and w.force_data == []
        assert "Added: Fixed support on gui_support1 (x)" in w.message_text.toPlainText()
        lw = w.structural_loads_window                                       # a force on a face, as clicked
        w.select_triangles(10)
        lw.load_type.setCurrentText("Force")
        lw.force_spinboxes["X"].setValue(1.0)
        lw.on_apply_clicked()
        for _ in range(5):
            APP.processEvents()
        assert not [n for n in w.plotter.actors if "_text_" in str(n)]       # no label left behind as a dot
        assert w.project.bcs[-1] == {"kind": "force", "on": "gui_load2", "total": [1.0, 0.0, 0.0]}
        w.selected_points = [[0.0, 0.0, 0.0]]                                  # picked, not applied
        w.update_point_markers()
        panel = w.open_supports_and_loads_panel()
        panel.clear_preview()
        assert w.selected_points == [] and "selected_points" not in w.plotter.actors
        assert "A torque is not one of LSR MMTO's loads" in w.message_text.toPlainText()


def test_pick_window_clear_picks():
    """The Pick window has no Box select nodes; Clear picks (and a right click) removes every face and node
    picked; picks not applied go when the window closes. A project with a Dragged box opens, shows and runs."""
    from lsr_mmto.problem import selections as S
    with scratch() as d:
        w = opened("L-bracket", d)
        w.show()
        w.open_structural_loads_window()
        pick = w.structural_loads_window
        buttons = [b.text() for b in pick.findChildren(QtWidgets.QPushButton)]
        assert "Box select nodes" not in buttons and "Clear picks" in buttons
        pick.selection_combo.setCurrentText("Node")
        assert not [b for b in pick.findChildren(QtWidgets.QPushButton) if "Box" in b.text()]

        def picked():
            w.select_triangles(10)
            w.update_highlights()
            w.selected_points = [[0.0, 0.0, 0.0]]
            w.update_point_markers()
            assert w.picked_count() and "selected_points" in w.plotter.actors
        picked()
        pick.clear_button.click()
        assert w.picked_count() == 0 and w.selected_points == [] and "selected_points" not in w.plotter.actors
        picked()
        w.on_right_button_press(None, None)                                   # a right click: the same
        assert w.picked_count() == 0 and "selected_points" not in w.plotter.actors
        picked()
        pick.parentWidget().close()                                           # the strip's x: picks go
        APP.processEvents()
        assert w.picked_count() == 0 and w.selected_points == [] and "selected_points" not in w.plotter.actors

        p = w.project                                                         # a Dragged box, made earlier
        p.add_selection("dragged", S.halfspaces([[1, 0, 0], [-1, 0, 0]], [[20, 0, 0], [0, 0, 0]]))
        p.bcs.append({"kind": "fixed", "on": "dragged", "dofs": "xyz"})
        w.save_project()
        assert w.open_project(w.path) and w.project.selections["dragged"].TYPE == "halfspaces"
        panel = w.open_supports_and_loads_panel()
        r = panel.edit_region("dragged")
        assert r.kind() == "halfspaces" and r.from_form().to_dict() == w.project.selections["dragged"].to_dict()
        r.reject()
        panel.preview()
        assert panel.counts["dragged"] > 0 and "Error" not in panel.report.text()
        mesh = w.preview_mesh()
        bcs, _ = w.project.supports_and_loads(mesh)
        assert w.project.supports_and_loads_error(mesh, bcs) is None


def test_docked():
    """Supports and loads and the Pick window sit on the right edge of the window, not over the 3D view;
    a second click raises the open one; Close (or a new project) removes the strip."""
    docks = lambda w: [d for d in w.findChildren(QtWidgets.QDockWidget) if d.isVisible()]  # noqa: E731
    with scratch() as d:
        w = opened("L-bracket", d)
        w.show()
        panel = w.open_supports_and_loads_panel()
        assert w.open_supports_and_loads_panel() is panel and len(docks(w)) == 1
        assert w.dockWidgetArea(panel.parent()) == QtCore.Qt.RightDockWidgetArea
        w.open_structural_loads_window()
        assert [x.windowTitle() for x in docks(w)] == ["Supports and loads", w.structural_loads_window.windowTitle()]
        w.structural_loads_window.close()                                    # its own Close button
        APP.processEvents()
        assert [x.windowTitle() for x in docks(w)] == ["Supports and loads"]
        w.open_project(w.path)                                               # the strips go with the project
        APP.processEvents()
        assert docks(w) == []
        panel = w.open_supports_and_loads_panel()
        panel.parent().close()                                               # the strip's x
        APP.processEvents()
        assert docks(w) == [] and w.open_supports_and_loads_panel().isVisible()


def test_add_flow():
    """Add offers both ways; an empty list says what to do; New shape makes a selection without leaving the
    dialog; a pick takes the list's place and Apply brings the list back with the new entry selected;
    each warning is said once."""
    from lsr_mmto.problem import selections as S
    with scratch() as d:
        w = opened("L-bracket", d)
        w.show()
        panel = w.open_supports_and_loads_panel()
        add = [b for b in panel.findChildren(QtWidgets.QPushButton) if b.text() == "Add"][0]
        assert [a.text() for a in add.menu().actions()] == ["Click on the part...", "Enter by hand..."]
        assert not panel.empty.isVisibleTo(panel) and not panel.regions.isVisibleTo(panel)   # Places: closed
        b = SP.BCDialog(panel, w.project)                                    # New shape, then OK
        real = SP.RegionDialog.exec_
        try:
            SP.RegionDialog.exec_ = lambda self: (self.type.setCurrentText("Face of the part"), 1)[1]
            b.new_shape()
        finally:
            SP.RegionDialog.exec_ = real
        assert b.on.currentText() == "selection1" and "selection1" not in w.project.selections   # not yet: on OK
        assert b.new_regions["selection1"].to_dict() == S.face("xmin").to_dict()
        w.project.bcs.clear()
        panel.refresh()
        assert panel.empty.isVisibleTo(panel) and not panel.bcs.isVisibleTo(panel)
        front = lambda: [t.tabText(t.currentIndex()) for t in w.findChildren(QtWidgets.QTabBar)  # noqa: E731
                         if t.isVisible() and t.count() > 1]
        panel.pick()                                                          # the round trip
        APP.processEvents()
        pick = w.structural_loads_window
        assert front() == ["Pick"]                                            # in the list's place
        assert w.open_supports_and_loads_panel() is panel                     # the list's tab: no second copy
        assert front() == ["Supports and loads"]
        w.open_structural_loads_window()                                      # the same one, to the front
        assert w.structural_loads_window is pick and front() == ["Pick"]
        w.select_triangles(10)
        pick.load_type.setCurrentText("Force")
        pick.force_spinboxes["Y"].setValue(-2.0)
        pick.on_apply_clicked()
        for _ in range(5):
            APP.processEvents()
        assert front() == [] and panel.isVisible() and w.project.bcs[-1]["kind"] == "force"   # Pick closed
        assert panel._selected(panel.bcs) == len(w.project.bcs) - 1
        assert not hasattr(w, "structural_loads_window")                     # gone, and nothing points at it
        G.UnitsWindow(w, w.settings).apply_settings()                         # once crashed on the deleted Pick
        text = w.problems.toPlainText()
        assert text.count("Warning: the database gives Young's modulus") == 1
        w.apply_panel(type("P", (), {"apply": lambda self, p: ["changed"]})())   # a change: shown twice? no
        assert w.problems.toPlainText().count("Warning: the database gives Young's modulus") == 1


def test_scale():
    """The Ruler tick shows and hides the scales; the region form says how big the part is; the point
    under the mouse is read off the part."""
    with scratch() as d:
        w = opened("L-bracket", d)
        assert w.part_extent_text() == "The part spans x 0 to 200, y 0 to 200, z 0 to 1 mm."
        assert w._rulers == []                                                # off by default
        assert w.extent.text() == "Part: x 0 to 200 \u00b7 y 0 to 200 \u00b7 z 0 to 1 mm"   # always shown
        w.ruler_box.setChecked(True)
        assert len(w._rulers) == 3                                            # x, y, the origin: a plate has no z
        assert w.open_project(w.path) and len(w._rulers) == 3                 # kept on for the next project
        w.ruler_box.setChecked(False)
        assert w._rulers == [] and not [n for n in w.plotter.actors if str(n).startswith("mmto_origin")]
        r = SP.RegionDialog(w.open_supports_and_loads_panel(), w.project)
        assert any(label.text() == w.part_extent_text() for label in r.findChildren(QtWidgets.QLabel))
    assert G.MMTOWindow.length_format(200) == "%.0f" and G.MMTOWindow.length_format(0.029) % 0.016 == "0.0160"
    assert G.MMTOWindow.length_format(20) % 16 == "16.0" and G.MMTOWindow.length_format(0.1) % 0.05 == "0.050"
    import types
    with scratch() as d:                                                      # 0 on a small part, the rest as before
        w = opened("L-bracket", d)
        for corners, text in (([[-1e-12, 0.002, 0], [0.029, 0.016, 0.001]],
                               "x 0 to 0.0290, y 0.0020 to 0.0160, z 0 to 0.0010"),     # a hair below 0
                              ([[-10, 0, 0], [116, 38, 1]], "x -10 to 116, y 0 to 38, z 0 to 1")):
            vectors = np.array([corners], dtype=float)
            w.stl_geom = types.SimpleNamespace(mesh=types.SimpleNamespace(vectors=vectors))
            assert w.part_extent_text() == f"The part spans {text} mm.", w.part_extent_text()
        w.stl_geom = None


def test_snap_steps_follow_the_part():
    """Snap offers off and three round steps (1, 2 or 5 times a power of ten, at or below) of a thousandth,
    a hundredth and a tenth of the part's largest extent, with the length unit; they change with the part,
    and a point snaps to the step chosen."""
    import pyvista as pv
    assert G.snap_steps(0.029) == [0.00002, 0.0002, 0.002] and G.snap_steps(200) == [0.2, 2.0, 20.0]
    assert G.snap_steps(0.3) == [0.0002, 0.002, 0.02] and G.snap_steps(50) == [0.05, 0.5, 5.0]
    assert G.snap_steps(0) == []
    with scratch() as d:
        w = G.MMTOWindow()
        w.resize(1500, 900)
        w.show()                                                              # shown before a part, as it opens
        assert w.open_project(shutil.copy(os.path.join(PROJECTS, "L-bracket.mmto"), d))
        texts = [w.snap.itemText(i) for i in range(w.snap.count())]
        assert texts == ["Snap: off", "Snap: 0.2 mm", "Snap: 2 mm", "Snap: 20 mm"]
        w.snap.setCurrentIndex(1)
        assert w.snap.currentData() == 0.2 and np.allclose(w.snapped([119.76, 80.02, 1.0]), [119.8, 80.0, 1.0])
        small = os.path.join(d, "small.stl")
        pv.Box(bounds=(0, 0.029, 0, 0.01, 0, 0.001)).triangulate().save(small)
        with answers(Box.Yes):
            assert w.replace_part(small)
        texts = [w.snap.itemText(i) for i in range(w.snap.count())]
        assert texts == ["Snap: off", "Snap: 0.00002 mm", "Snap: 0.0002 mm", "Snap: 0.002 mm"]
        assert w.snap.currentIndex() == 1                                     # still the finest
        for _ in range(5):
            APP.processEvents()
        need = w.snap.fontMetrics().horizontalAdvance(w.snap.currentText()) + w.snap.iconSize().width() + 30
        assert w.snap.width() >= need                                         # the longer step not cut short
        assert np.allclose(w.snapped([0.01637, 0.0, 0.0]), [0.01636, 0.0, 0.0])
        assert w.extent.text() == "Part: x 0 to 0.0290 \u00b7 y 0 to 0.0100 \u00b7 z 0 to 0.0010 mm"    # a zero is 0


def test_selection_form_extent_follows_the_part_and_units():
    """The Selection form's "The part spans ..." line is written again when the part is replaced and when
    the units change, while the form stays open."""
    import pyvista as pv
    real_exec = G.UnitsWindow.exec_
    with scratch() as d:
        w = opened("L-bracket", d)
        panel = w.open_supports_and_loads_panel()
        panel.add_region()
        form = panel.stack.currentWidget().findChild(SP.RegionDialog)
        assert form.extent.text() == "The part spans x 0 to 200, y 0 to 200, z 0 to 1 mm."
        small = os.path.join(d, "small.stl")
        pv.Box(bounds=(0, 50, 0, 20, 0, 2)).triangulate().save(small)
        with answers(Box.Yes):
            assert w.replace_part(small)
        assert form.extent.text() == "The part spans x 0 to 50.0, y 0 to 20.0, z 0 to 2.0 mm."
        try:
            def apply_ips(self):
                self.settings.update_settings("IPS", self.settings.temperature_unit, self.settings.angle_unit)
                return 1
            G.UnitsWindow.exec_ = apply_ips
            w.open_units_window()
        finally:
            G.UnitsWindow.exec_ = real_exec
        assert form.extent.text() == "The part spans x 0 to 50.0, y 0 to 20.0, z 0 to 2.0 in."
        form.reject()


def test_region_preview_clears():
    """In the region form, Clear takes the preview off the part, any edit does too, and so does closing."""
    with scratch() as d:
        w = opened("L-bracket", d)
        panel = w.open_supports_and_loads_panel()
        on_part = lambda: "mmto_region" in w.plotter.actors                       # noqa: E731
        r = SP.RegionDialog(panel, w.project, preview=panel.preview_one)
        r.type.setCurrentText("Flat cut")
        r.inputs["axis"].setCurrentText("x")
        r.inputs["value"].setText("12")
        r.show_preview()
        assert on_part() and "nodes" in r.result.text()
        r.inputs["value"].setText("20")                                       # an edit: the old preview goes
        assert not on_part() and r.result.text() == ""
        r.show_preview()
        assert on_part()
        r.clear_form()                                                        # the Clear button
        assert not on_part() and r.inputs["value"].text() == ""
        r.show_preview()
        r.type.setCurrentText("Box")                                          # another type: goes too
        assert not on_part()
        r.type.setCurrentText("Flat cut")
        r.inputs["value"].setText("20")
        r.show_preview()
        r.reject()                                                            # Cancel: nothing left behind
        assert not on_part()


def test_inside_stl_move_by():
    """Inside an STL has a Move by row (x, y, z with the length unit and an (i)); the selection keeps the move,
    the project file and the code line only when it is not zero, and the preview draws the moved shape."""
    from lsr_mmto.problem import selections as S
    airplane = Project.load(os.path.join(PROJECTS, "Airplane bracket.mmto"))
    collar = airplane.selections["collar"]
    with scratch() as d:
        w = opened("L-bracket", d)
        r = SP.RegionDialog(w, airplane, "collar")
        box = r.inputs["shift"]
        assert r.form.labelForField(box.parentWidget()).text() == "Move by" and box.text() == "0, 0, 0"
        row = [x for x in box.parentWidget().children() if isinstance(x, QtWidgets.QWidget)]
        assert any(isinstance(x, QtWidgets.QLabel) and x.text() == "mm" for x in row)
        assert any(isinstance(x, SP.Info) and x.toolTip().startswith("Moves the shape before nodes") for x in row)
        assert r.from_form().to_dict() == collar.to_dict() and "shift" not in collar.to_dict()
        assert SP.describe(collar).startswith("Inside an STL: File ") and "Move by" not in SP.describe(collar)
        box.setText("10, 0, 0")
        moved = r.selection()
        assert moved.shift == [10.0, 0.0, 0.0] and moved.to_dict()["shift"] == [10.0, 0.0, 0.0]
        assert SP.describe(moved).endswith("Move by 10, 0, 0")
        r.tabs.setCurrentIndex(1)
        assert r.code.text().endswith("shift=[10.0, 0.0, 0.0])")
        assert SP.parse_selection(r.code.text()).to_dict() == moved.to_dict()
        mesh = w.preview_mesh()
        here = G.place_outline(collar, mesh, airplane.base_dir)
        there = G.place_outline(moved, mesh, airplane.base_dir)
        assert np.allclose(np.array(there.bounds) - np.array(here.bounds), [10, 10, 0, 0, 0, 0])


def test_click_a_hole():
    """Click a hole in the Cylinder form, then a click on the inside wall of a hole: the wall is grown as a
    Facet pick grows, and the form gets the hole's centre, axis, radius and ends; Tolerance Automatic takes
    the wall's nodes half an element outside the radius. A flat face fills nothing."""
    import pyvista as pv
    with scratch() as d:
        stl = os.path.join(d, "washer.stl")                           # a plate (a washer) with a hole of
        pv.Disc(center=(40, 30, 0), inner=8, outer=20, r_res=2, c_res=48).extrude(   # radius 8 at x 40, y 30,
            (0, 0, 10), capping=True).triangulate().clean().compute_normals(          # through z 0 to 10, its
            auto_orient_normals=True).save(stl)                                     # faces facing out
        w = G.MMTOWindow()
        assert w.new_project(stl, os.path.join(d, "washer"))
        w.show()
        geom = w.stl_geom
        v = np.asarray(geom.mesh.vectors, dtype=float)
        n = np.asarray(geom.tri_normals, dtype=float)
        c = v.mean(axis=1)
        on_wall = (np.abs(np.hypot(c[:, 0] - 40, c[:, 1] - 30) - 8) < 0.5) & (np.abs(n[:, 2]) < 0.1)
        wall, top = int(np.flatnonzero(on_wall)[0]), int(np.flatnonzero(n[:, 2] > 0.99)[0])

        class Picker:                                                         # a click on the part
            def __init__(self, cell):
                self.cell = cell

            def GetCellId(self):
                return self.cell
        panel = w.open_supports_and_loads_panel()
        r = panel.add_region()
        r.type.setCurrentText("Cylinder")
        button = next(b for b in r.findChildren(QtWidgets.QPushButton) if b.text() == "Click a hole")
        button.click()
        assert "Click the inside wall" in r.result.text()
        w.on_left_button_press(c[top].tolist(), Picker(top))                  # a flat face: said so
        assert "on a flat face" in r.result.text() and r.inputs["radius"].text() == ""
        geom.tri_highlight[5] = True                                          # a pick in the Pick window, kept
        button.click()
        w.on_left_button_press(c[wall].tolist(), Picker(wall))
        centre = [float(x) for x in r.inputs["center"].text().split(",")]
        assert np.allclose(centre[:2], [40, 30], rtol=0.01) and abs(centre[2]) < 1e-6
        assert r.inputs["axis"].currentText() == "z"
        assert abs(float(r.inputs["radius"].text()) - 8) < 0.08              # within 1 %, the hole's own
        assert [float(b.text()) for b in r.inputs["along"]] == [0.0, 10.0]
        assert r.inputs["inner_radius"].text() == "0"
        assert "nodes on the run's mesh" in r.result.text() and "mmto_region" in w.plotter.actors
        assert "elements across" not in r.result.text()                     # a hole many elements across
        bore = r.selection()
        radius = float(r.inputs["radius"].text())
        from lsr_mmto.problem.mesh import make_mesh
        p = w.project
        for n_dof in (p.n_dof, 4 * p.n_dof):                                # the wall nodes on two mesh sizes
            mesh = make_mesh(os.path.abspath(p.stl_path()), n_dof, p.single_layer, p.unknowns_per_node())
            xyz = np.asarray(mesh.node_xyz, dtype=float)
            pad = 0.5 * float(np.min(mesh.elem_size))                       # Tolerance Automatic
            boundary = np.asarray(mesh.get_boundary_nodes())
            wall_nodes = boundary[np.hypot(xyz[boundary, 0] - 40, xyz[boundary, 1] - 30) < radius + pad - 1e-9]
            taken = p.evaluate(bore, mesh, "nodes")
            assert len(wall_nodes) and set(wall_nodes) <= set(taken), n_dof
            assert np.hypot(xyz[taken, 0] - 40, xyz[taken, 1] - 30).max() <= radius + pad + 1e-9
        assert np.flatnonzero(geom.tri_highlight).tolist() == [5]
        w.on_left_button_press(c[wall].tolist(), Picker(wall))                # one fill per click of the button
        assert "Copied" in panel.stack.currentWidget().copied.text()
        r.reject()

    from gui.supports_and_loads_panel import hole_from_wall                   # a tilted hole
    axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    tube = pv.Cylinder(center=(5, 6, 7), direction=axis, radius=3, height=4, resolution=60, capping=False)
    tube = tube.triangulate().compute_normals(cell_normals=True, point_normals=False)
    corners = tube.points[tube.faces.reshape(-1, 4)[:, 1:]]
    point, got, radius, start, end, outside = hole_from_wall(tube.cell_data["Normals"], corners)
    assert np.allclose(got, axis) and abs(radius - 3) < 0.03 and abs(end - start - 4) < 1e-6
    assert np.allclose(point + 2 * axis, [5, 6, 7])                         # the middle of the tube
    assert outside and not hole_from_wall(-tube.cell_data["Normals"], corners)[5]   # normals out: a round face


def test_filled_numbers_to_the_parts_precision():
    """The numbers Click a hole fills in keep the part's own precision: under 1e-6 x its largest extent is 0;
    on a 200 mm part the values are as they were."""
    from gui.supports_and_loads_panel import _number_text
    assert _number_text(-2.4e-08, 0.22) == "0" and _number_text(-0.00876, 0.22) == "-0.00876"
    assert _number_text(0.0028199999489, 0.22) == "0.00282" and _number_text(-0.0325626, 0.22) == "-0.0325626"
    for v in (40.0, 30.000000049, 7.999999699, 0.0, 10.0, -52.5, 123.456789, 0.25, 1e-3):
        assert _number_text(v, 200.0) == format(round(v, 9) + 0.0, ".6g"), v


def test_click_a_hole_on_the_rotor():
    """Click a hole on a brake rotor saved in metres: every triangle of the inside wall of the bore, the six
    bolt holes and the six rim notches gives its centre and radius (within 1 %); a bolt hole, under two element
    sizes in radius on the run's mesh, gets the note that it is only a few elements across, the bore does not;
    Surface nodes only is ticked, and the bore with it has no node off the mesh surface (without it, 12);
    a click on a flat face says so; the outside of the rim fills a tube about it, which takes the nodes of the
    rim, not the whole part."""
    from gui.supports_and_loads_panel import hole_from_wall
    from lsr_mmto.problem import selections as S
    with scratch() as d:
        w = G.MMTOWindow()
        assert w.new_project(os.path.join(ROOT, "tests", "data", "brake_rotor.stl"), os.path.join(d, "rotor"))
        w.show()
        geom = w.stl_geom
        n = np.asarray(geom.tri_normals, dtype=float)
        v = np.asarray(geom.mesh.vectors, dtype=float)
        c = v.mean(axis=1)
        bore = np.array([0.0, -0.00876])
        side = np.flatnonzero(np.abs(n[:, 2]) < 0.1)                         # the walls across the disc
        rim = side[np.abs(np.hypot(*(c[side, :2] - bore).T) - 0.1096) < 0.0005]
        walls = np.setdiff1d(side, rim)
        found, first = {}, {}
        for t in walls:
            centre, axis, r, start, end, outside = hole_from_wall(*w.wall_triangles(int(t)))
            assert not outside and np.array_equal(axis, [0, 0, 1]), t
            known = min((0.0141, 0.00282, 0.0115), key=lambda k: abs(k - r))
            assert abs(r - known) < 0.01 * known, (t, r)
            assert np.all(np.abs(np.hypot(*(v[t, :, :2] - centre[:2]).T) - r) < 0.01 * r), t   # on that circle
            found.setdefault(known, set()).add(tuple(np.round(centre[:2], 4)))
            first.setdefault(known, int(t))
        assert {k: len(cs) for k, cs in found.items()} == {0.0141: 1, 0.00282: 6, 0.0115: 6}
        assert np.allclose(next(iter(found[0.0141])), bore, atol=0.01 * 0.0141)

        class Picker:                                                         # a click on the part
            def __init__(self, cell):
                self.cell = cell

            def GetCellId(self):
                return self.cell
        panel = w.open_supports_and_loads_panel()
        r = panel.add_region()
        r.type.setCurrentText("Cylinder")
        button = next(b for b in r.findChildren(QtWidgets.QPushButton) if b.text() == "Click a hole")
        flat = int(np.flatnonzero(n[:, 2] < -0.99)[0])
        button.click()
        w.on_left_button_press(c[flat].tolist(), Picker(flat))
        assert "That click was on a flat face, not the inside wall of a hole" in r.result.text()
        assert r.inputs["radius"].text() == ""
        for known in (0.00282, 0.0141):                                      # a bolt hole, the bore
            button.click()
            w.on_left_button_press(c[first[known]].tolist(), Picker(first[known]))
            radius = float(r.inputs["radius"].text())
            assert abs(radius - known) < 0.01 * known and r.inputs["inner_radius"].text() == "0"
            h = float(np.max(w.preview_mesh().elem_size))
            note = f"This hole is only {2 * radius / h:.1f} elements across on the run's mesh"
            assert "nodes on the run's mesh" in r.result.text()
            assert (note in r.result.text()) == (radius < 2 * h) == (known == 0.00282), r.result.text()
            assert r.inputs["boundary"].isChecked()
        assert r.inputs["center"].text().split(", ")[1] == "-0.00876"          # the bore's y, to the part's
        mesh = w.preview_mesh()                                               # precision; at 30000 dofs
        assert w.project.n_dof == 30000
        surface = set(np.asarray(mesh.get_boundary_nodes()).tolist())
        bore_on_surface = r.selection()
        bore_through = S.from_dict(dict(bore_on_surface.to_dict(), boundary=False))
        assert not set(w.project.evaluate(bore_on_surface, mesh, "nodes").tolist()) - surface
        assert len(set(w.project.evaluate(bore_through, mesh, "nodes").tolist()) - surface) == 12
        button.click()
        w.on_left_button_press(c[rim[0]].tolist(), Picker(int(rim[0])))      # the outside of the rim: a tube
        radius, inner = r.inputs["radius"].text(), r.inputs["inner_radius"].text()
        assert radius.endswith(" + 1h") and inner.endswith(" - 1h") and radius.split()[0] == inner.split()[0]
        assert abs(float(radius.split()[0]) - 0.1096) < 0.001
        assert r.inputs["center"].text().split(", ")[0] == "0"                 # the fit's noise in x is gone
        mesh = w.preview_mesh()
        xyz = np.asarray(mesh.node_xyz, dtype=float)
        h = float(np.max(mesh.elem_size))
        taken = w.project.evaluate(r.selection(), mesh, "nodes")
        out = np.hypot(*(xyz[taken, :2] - bore).T)
        assert len(taken) and len(taken) < 0.2 * mesh.num_nodes
        assert out.min() >= 0.1096 - 1.6 * h and out.max() <= 0.1096 + 1.6 * h
        r.reject()


def test_empty_inner_radius_is_zero():
    """An empty Inner radius is its default, 0 (a solid cylinder), on the form and the code line; an empty
    Radius, which has no default, is still refused."""
    with scratch() as d:
        w = opened("L-bracket", d)
        panel = w.open_supports_and_loads_panel()
        r = panel.add_region()
        r.type.setCurrentText("Cylinder")
        r.inputs["center"].setText("52, 114, 1")
        r.inputs["radius"].setText("5")
        r.inputs["inner_radius"].setText("")
        bore = r.selection()
        assert bore.inner_radius == 0 and not r.problems()
        r.tabs.setCurrentIndex(1)
        assert "inner_radius=0.0" in r.code.text()
        r.tabs.setCurrentIndex(0)
        r.inputs["radius"].setText("")
        assert any("Radius: no value" in e for e in r.problems())
        r.reject()


def test_clear_empties_the_form():
    """Clear in the Selection form puts every field back to its default and removes the form's preview."""
    with scratch() as d:
        w = opened("L-bracket", d)
        w.show()
        panel = w.open_supports_and_loads_panel()
        r = panel.add_region()
        r.type.setCurrentText("Cylinder")
        r.inputs["center"].setText("52, 114, 1")
        r.inputs["radius"].setText("5")
        r.inputs["inner_radius"].setText("2")
        r.inputs["along"][0].setText("0")
        r.inputs["along"][1].setText("3")
        r.show_preview()
        assert "mmto_region" in w.plotter.actors
        clear = next(b for b in r.findChildren(QtWidgets.QPushButton) if b.text() == "Clear")
        assert clear.toolTip() == "Empty the form and remove its preview"
        clear.click()
        assert "mmto_region" not in w.plotter.actors and r.result.text() == ""
        assert r.kind() == "cylinder" and r.code.text() == ""
        assert [r.inputs[f].text() for f in ("center", "radius", "inner_radius")] == ["", "", "0"]
        assert [b.text() for b in r.inputs["along"]] == ["", ""]
        assert r.inputs["pad"].currentText() == "Automatic" and r.inputs["axis"].currentText() == "x"
        r.reject()


def test_enter_previews():
    """Enter in the Selection form and in the support or load form checks what was typed and shows it on
    the part; the window stays open, a mistake shows in its message line; only OK saves and closes."""
    from PyQt5 import QtTest
    enter = lambda box: QtTest.QTest.keyClick(box, QtCore.Qt.Key_Return)   # noqa: E731
    with scratch() as d:
        w = opened("L-bracket", d)
        w.show()
        panel = w.open_supports_and_loads_panel()
        r = panel.add_region()
        ok = lambda dialog: next(b for b in dialog.findChildren(QtWidgets.QPushButton)   # noqa: E731
                                 if b.text() == "OK")
        assert not ok(r).isDefault()
        r.type.setCurrentText("Cylinder")
        r.inputs["center"].setText("52, 114, 1")
        r.inputs["radius"].setText("5")
        enter(r.inputs["radius"])
        assert r.isVisible() and "nodes on the run's mesh" in r.result.text() and "mmto_region" in w.plotter.actors
        r.inputs["axis"].setCurrentText("1, 1, 0")                            # an edit: the preview goes
        assert "mmto_region" not in w.plotter.actors
        enter(r.inputs["axis"].lineEdit())                                    # in an editable list too
        assert r.isVisible() and "mmto_region" in w.plotter.actors
        assert r.inputs["axis"].findText("1, 1, 0") < 0                       # Enter adds no choice
        r.inputs["radius"].setText("five")
        enter(r.inputs["radius"])
        assert r.isVisible() and "#b00000" in r.result.text() and "'five'" in r.result.text()
        assert "mmto_region" not in w.plotter.actors
        r.inputs["radius"].setText("5")
        r.name.setText("bore")
        ok(r).click()                                                         # OK: saved and closed
        assert "bore" in w.project.selections and panel.stack.count() == 1

        b = panel.add_bc()
        assert not ok(b).isDefault()
        b.kind.setCurrentText("Force")
        b.on.setCurrentText("bore")
        b.inputs["total"][1][1].setText("-10/2")
        enter(b.inputs["total"][1][1])
        assert b.isVisible() and "nodes on the run's mesh" in b.result.text()
        assert {"mmto_region", "mmto_arrow_region"} <= set(w.plotter.actors)  # the load, with its arrow
        b.inputs["total"][1][0].setText("x")
        enter(b.inputs["total"][1][0])
        assert b.isVisible() and "Total force is not a number" in b.result.text()
        b.inputs["total"][1][0].setText("0")
        n = len(w.project.bcs)
        ok(b).click()
        assert len(w.project.bcs) == n + 1 and w.project.bcs[-1] == {"kind": "force", "on": "bore",
                                                                     "total": [0.0, -5.0, 0.0]}
        assert panel.stack.count() == 1 and "mmto_arrow_region" not in w.plotter.actors


def test_pages():
    """Forms opened from Supports and loads are pages of its strip, one at a time: Enter by hand, then
    New selection over it; OK goes back one page each time. Combined selections follow their parts by name."""
    from lsr_mmto.problem import selections as S
    with scratch() as d:
        w = opened("L-bracket", d)
        w.show()
        panel = w.open_supports_and_loads_panel()
        home = panel.stack.currentWidget()
        b = panel.add_bc()                                                    # a page, not a window
        assert panel.stack.count() == 2 and b.isVisible() and not home.isVisible()
        r = b.new_shape()                                                     # New selection: a page over it
        assert panel.stack.count() == 3 and panel.stack.currentWidget().mouse.text().startswith("Hover")
        panel.show_mouse("x 1   y 2   z 0 mm")
        assert panel.stack.currentWidget().mouse.text() == "Mouse on the part: x 1   y 2   z 0 mm"
        r.type.setCurrentText("Face of the part")
        r.accept()
        assert panel.stack.count() == 2 and b.on.currentText() == "selection1"     # back to the support/load form
        b.kind.setCurrentText("Fixed support")
        b.accept()
        assert panel.stack.count() == 1 and panel.stack.currentWidget() is home
        assert w.project.bcs[-1] == {"kind": "fixed", "on": "selection1", "dofs": "xyz"}
        assert w.project.selections["selection1"].to_dict() == S.face("xmin").to_dict()

        r = panel.add_region()                                                # Combined selections: by name, live
        r.type.setCurrentText("Combined selections")
        r.inputs["a"].setCurrentText("clamp")
        r.inputs["b"].setCurrentText("corner")
        r.name.setText("both")
        r.accept()
        assert w.project.selections["both"].to_dict() == S.union(S.selection("clamp"), S.selection("corner")).to_dict()
        mesh = w.preview_mesh()
        n = len(w.project.evaluate("both", mesh))
        w.project.selections["corner"] = S.face("xmax")                       # an edit: the combination follows
        assert len(w.project.evaluate("both", mesh)) != n
        panel.rename("clamp", "grip")                                         # a rename: carried through
        assert S.names_in(w.project.selections["both"]) == ["grip", "corner"]
        with answers():
            assert not panel.delete_region("corner")                          # used by both: refused
        r = panel.edit_region("both")                                         # opens on the form, as made
        assert r.kind() == "combined" and r.inputs["a"].currentText() == "grip"
        r.name.setText("grip2")
        w.project.selections["grip"] = S.union(S.selection("grip2"))              # a loop: refused
        assert any("cannot contain itself" in e for e in r.problems())
        r.reject()
        w.project.selections["grip"] = S.face("xmin")

        r = panel.add_region()                                                # Flat cut: optional limits
        r.type.setCurrentText("Flat cut")
        r.inputs["value"].setText("12")
        assert not r.inputs["limits"]["x"][1].isVisibleTo(r)                  # no x limits on a cut across x
        r.inputs["limits"]["y"][0][0].setText("0")
        assert any("give both" in e for e in r.problems())
        r.inputs["limits"]["y"][0][1].setText("40")
        assert r.selection().to_dict() == S.plane("x", 12, limits={"y": [0, 40]}).to_dict()
        r.reject()

        panel.report.setText("<span style='color:#b00000'>Error: old</span>")   # a stale line goes
        w.preview_on = False
        panel.set_keep_solid("corner", True)
        assert panel.report.text() == ""


def test_keep_solid_on_picked_faces():
    """Keep solid on a selection that picks nodes only (rows, picked faces): the elements touching it."""
    with scratch() as d:
        w = opened("L-bracket", d)
        mesh = w.preview_mesh()
        nodes = w.project.evaluate("corner", mesh, "nodes")
        elems = w.project.evaluate("corner", mesh, "elements")                 # rows: nodes only, once an error
        assert len(elems) and np.isin(np.asarray(mesh.elemArray)[elems], nodes).any(axis=1).all()
        w.project.keep_solid_names = ["corner", "clamp"]
        lines, _ = w.draw_supports_and_loads()
        assert not [line for line in lines if "Error" in line]
        assert [n for n in w.plotter.actors if str(n).startswith("mmto_keep_")]


def test_outline_one_row_measure():
    """A selection's shape is drawn with its nodes; clicking a row shows only that one; Measure gives the
    distance between two clicks; the mouse lookup never runs while a button is held or one is running."""
    from lsr_mmto.problem import selections as S
    with scratch() as d:
        w = opened("L-bracket", d)
        w.show()
        mesh = w.preview_mesh()
        for sel in (S.cylinder([52, 114, 1], "x", 5, along=[-30, 9]), S.box([0, 0, 0], [10, 10, 1]),
                    S.sphere([10, 10, 0], 3), S.plane("x", 12, limits={"y": [0, 40]}), S.face("xmin"),
                    S.near_face("ymax", 5.0)):
            shape = G.place_outline(sel, mesh)
            assert shape is not None and shape.n_points, sel
        assert G.place_outline(S.rows("xmax", "y", 8), mesh) is None          # no shape of its own
        lo, hi = np.array(G.place_outline(S.cylinder([52, 114, 1], "x", 5, along=[-30, 9]), mesh).bounds).reshape(3, 2).T
        assert abs(lo[0] - 22) < 1e-6 and abs(hi[0] - 61) < 1e-6              # x 22 to 61, as typed

        panel = w.open_supports_and_loads_panel()
        panel.bcs.selectRow(1)                                                # the force on corner, alone
        shown = [n for n in w.plotter.actors if str(n).startswith("mmto_")]
        assert "mmto_corner" in shown and "mmto_arrow1" in shown and "mmto_clamp" not in shown
        assert panel.report.text() == "corner: 16 nodes"
        panel.places_header.setChecked(True)
        panel.regions.selectRow(0)                                            # the place clamp, alone
        assert "mmto_clamp" in w.plotter.actors and "mmto_corner" not in w.plotter.actors

        class Picker:                                                         # a click on the part
            def GetCellId(self):
                return 3
        w.measure.setChecked(True)
        assert w.measured.text() == "Measure: click the first point"
        w.on_left_button_press([0.0, 0.0, 1.0], Picker())
        w.on_left_button_press([30.0, 40.0, 1.0], Picker())
        assert w.measured.text().startswith("Distance 50 mm") and "measure_line" in w.plotter.actors
        assert w.selected_triangles == [] if hasattr(w, "selected_triangles") else True   # no picking meanwhile
        w.measure.setChecked(False)
        assert w.measured.text() == "" and "measure_line" not in w.plotter.actors

        calls = []
        w.show_point_under_mouse = lambda position=None: calls.append(position)
        w._looking = True
        w._look_up_mouse()                                                     # one running: skipped
        w._looking = False
        w._mouse_at = (5, 5)
        w._look_up_mouse()
        assert calls == [(5, 5)]


def _wait(condition, seconds=600):
    import time
    end = time.time() + seconds
    while not condition() and time.time() < end:
        APP.processEvents()
        time.sleep(0.05)
    APP.processEvents()
    return condition()


def test_run():
    """Run: a real short optimization from the strip; live plot and design; results in a folder of its
    own with the tabs at the end; Stop ends a run early and keeps its design; nothing closes mid-run."""
    with scratch() as d:
        w = opened("L-bracket", d)
        w.show()
        w.project.n_dof, w.project.single_layer = 3000, True
        w.project.settings["common"]["maxIterations"] = 3
        assert w.save_project()
        r = w.open_run_panel()
        assert r is w.run_window and r.windowTitle() == "Current run"
        assert not hasattr(r, "spec")                                         # PyTO-AD's formulation: not here
        assert r.start_optimization()
        assert _wait(lambda: not r.optimization_running), "the run did not finish"
        assert r.status.text().startswith("Run finished after 3 iterations"), r.status.text()
        assert os.path.isfile(os.path.join(r.save_dir, "run_data.npz"))
        assert sorted(os.listdir(r.save_dir)) == ["run_config.yaml", "run_data.config.json", "run_data.npz"]
        from lsr_mmto.results import load_results
        rec = load_results.load(os.path.join(r.save_dir, "run_data"))             # each element's material is the
        z, zr = rec.z_solid(), np.asarray(rec.z_real)                         # real one nearest its latent point
        nearest = np.argmin(((z[:, None, :] - zr[None, :, :]) ** 2).sum(-1), axis=1)
        assert np.array_equal(nearest, rec.material_indices[rec.solid])
        assert [r.tabs.tabText(i) for i in range(r.tabs.count())] == ["Progress", "Log", "Convergence",
                                                                      "Latent space", "Summary"]
        assert "result_design" in w.plotter.actors and not w.part_actor().GetVisibility()
        assert w.view == "design" and w.design_button.isChecked() and w.design_button.isEnabled()
        w.part_button.click()                                                 # Part, in the top row
        assert w.part_actor().GetVisibility() and "result_design" not in w.plotter.actors
        assert w.part_button.isChecked()
        panel = w.open_supports_and_loads_panel()
        w.design_button.click()                                               # Design again
        assert w.view == "design"
        panel.preview()                                                       # a setup action: the part
        assert w.view == "setup" and w.part_actor().GetVisibility()
        w.clear_preview()                                                     # Clear keeps the design
        assert w.show_design_view() and "result_design" in w.plotter.actors
        first = r.save_dir

        w.project.settings["common"]["maxIterations"] = 50                   # Stop: early, design kept
        assert w.save_project() and r.start_optimization()
        assert r.save_dir != first                                            # a folder per run
        assert [r.tabs.tabText(i) for i in range(r.tabs.count())] == ["Progress", "Latent space", "Log"]
        r.tabs.setCurrentWidget(r.latent_canvas)                              # the live latent space
        assert _wait(lambda: r.latent_figure.axes and r.latent_figure.axes[0].get_title().startswith("Iteration"))
        assert _wait(lambda: len(r.history["objective"]) >= 2)
        with answers():
            assert not w.open_project(w.path)                                 # not while it runs
            r.parentWidget().close()
        assert G.in_open_strip(r)                                             # the strip stays
        r.stop_optimization()
        assert _wait(lambda: not r.optimization_running)
        assert r.status.text().startswith("Run stopped"), r.status.text()
        assert r.tabs.indexOf(r.latent_canvas) < 0                            # the run's own tab follows
        assert os.path.isfile(os.path.join(r.save_dir, "run_data.npz"))

        w.project.formulation["objective"] = None                            # a project that cannot run
        assert not r.start_optimization() and "Cannot start" in r.status.text()


def test_network_warning_continue_or_cancel():
    """A network that reproduces the database less closely than the project's warning limit: before the
    optimization the window asks, with each property's error; Cancel leaves no run folder behind,
    Continue runs and keeps the errors with the run."""
    import torch
    from lsr_mmto.materials.databases import DATABASE_DIR, DATABASE_VAE, VAEParams
    from lsr_mmto.materials.encoder import MaterialEncoder
    asked, saved = [], Box.exec_

    def click(choice):
        def exec_(box):
            asked.append(box.text())
            next(b for b in box.buttons() if b.text().replace("&", "") == choice).click()
            return 0
        return exec_

    with scratch() as d:
        w = opened("L-bracket", d)
        xlsx = shutil.copy(os.path.join(ROOT, DATABASE_DIR, "BM1_3Mats_2Attr.xlsx"), d)
        vae = VAEParams()
        for k, v in DATABASE_VAE["BM1_3Mats_2Attr"].items():
            setattr(vae, k, v)
        torch.manual_seed(0)                                                 # the same untrained network
        np.random.seed(0)
        enc = MaterialEncoder(vae)
        enc.readExcel(xlsx)
        torch.save(enc.vaeNet.state_dict(), xlsx[:-5] + ".nt")             # untrained
        w.project.database = os.path.basename(xlsx)
        w.project.n_dof = 3000
        w.project.settings["vae"].update(use_pretrained_vae=True, latentDim=2, vae_hiddenDim=50)
        assert w.save_project()
        r = w.open_run_panel()
        try:
            Box.exec_ = click("Cancel")
            assert r.start_optimization()
            assert _wait(lambda: not r.optimization_running)
            assert len(asked) == 1 and "Young's modulus" in asked[0] and "Run anyway?" in asked[0], asked
            assert r.status.text().startswith("Not run: cancelled at the material network warning"), \
                r.status.text()
            assert not os.path.exists(r.save_dir)

            Box.exec_ = click("Continue")
            assert r.start_optimization()
            assert _wait(lambda: len(r.history["objective"]) >= 1)
            r.stop_optimization()
            assert _wait(lambda: not r.optimization_running)
        finally:
            Box.exec_ = saved
        assert len(asked) == 2 and r.status.text().startswith("Run stopped"), r.status.text()
        assert "network_errors_percent" in open(os.path.join(r.save_dir, "run_config.yaml")).read()


def test_export_and_run_script():
    """File > Export as script writes the project as Python in its folder, after a save when there are
    changes; the script edited and run with File > Run script opens the project it saved; a script that
    fails, or saves nothing, says so and leaves the project as it was; the script runs from any folder."""
    import subprocess
    with scratch() as d:
        w = opened("L-bracket", d)
        w.project.n_dof = 4000                                                # a change: saved first
        with answers(Box.Save):
            path = w.export_script()
        assert path and os.path.isfile(path) and not w.is_dirty()
        assert any("Script written" in line for line in w.message_text.toPlainText().splitlines())
        code = open(path).read()
        assert "from lsr_mmto import Project" in code and "total=[0.0, -4.0, 0.0]" in code and "n_dof=4000" in code
        with open(path, "w") as fh:
            fh.write(code.replace("total=[0.0, -4.0, 0.0]", "total=[0.0, -8.0, 0.0]"))
        assert w.run_script(path)
        assert _wait(lambda: w.script_process is None, 300)
        assert w.project.bcs[1]["total"] == [0.0, -8.0, 0.0], w.message_text.toPlainText()[-500:]
        assert w.path == os.path.join(d, "L-bracket.mmto") and not w.is_dirty()

        for text, said in (("raise ValueError('a mistake in the script')", "stopped with an error"),
                           ("print('nothing saved')", "saved no project")):
            bad = os.path.join(d, "other.py")
            with open(bad, "w") as fh:
                fh.write(text + "\n")
            assert w.run_script(bad)
            assert _wait(lambda: w.script_process is None, 120)
            log = w.message_text.toPlainText()
            assert said in log and (said != "stopped with an error" or "a mistake in the script" in log)
            assert w.project.bcs[1]["total"] == [0.0, -8.0, 0.0]

        elsewhere = tempfile.mkdtemp()                                        # not the repository folder
        try:
            done = subprocess.run([sys.executable, path], cwd=elsewhere, capture_output=True, text=True,
                                  timeout=300)
            assert done.returncode == 0, done.stderr[-1000:]
        finally:
            shutil.rmtree(elsewhere, ignore_errors=True)


def test_latent_views():
    """The latent space in 1, 2, 3 and 4 dimensions: the run's tab, Compare's and the live one draw it;
    1D as counts along the line with every name readable, 4D through a fixed projection whose title
    says for how many elements it shows the right nearest material; the live axes only widen."""
    from types import SimpleNamespace
    from gui import results_panel as R
    from gui import run_panel as RN
    from lsr_mmto.results import run_plots
    rng = np.random.default_rng(0)
    for n_z in (1, 2, 3, 4):
        zr = rng.normal(size=(6, n_z)) * 2
        z = zr[rng.integers(0, 6, 800)] + rng.normal(size=(800, n_z)) * 0.2
        run = SimpleNamespace(z_solid=lambda z=z: z, z_real=zr, material_labels=list("ABCDEF"),
                              material_names=list("ABCDEF"))
        fig = run_plots.latent_scatter(run)
        ax = fig.axes[0]
        if n_z == 1:
            assert ax.get_ylabel() == "elements" and sum(p.get_height() for p in ax.patches) == 800
            assert sorted(t.get_text() for t in ax.texts) == list("ABCDEF")
            fig.canvas.draw()                                                 # no two names overlap
            boxes = [t.get_window_extent() for t in ax.texts]
            assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:])
        if n_z == 4:
            assert "PCA projection from 4D" in ax.get_title() and "% of the elements" in ax.get_title()
        assert R.latent_side_by_side({"a": run, "b": run}).axes                # Compare draws it too
        import matplotlib.pyplot as plt
        plt.close(fig)
    with scratch() as d:
        w = opened("L-bracket", d)
        r = w.open_run_panel()
        r.project = w.project
        r._start_live_latent()
        assert r.tabs.tabText(1) == "Latent space"
        for n_z in (1, 3, 4):
            r.latent_pca, r.latent_limits = None, None
            zr = rng.normal(size=(3, n_z))
            for it, spread in ((1, 3.0), (2, 0.5)):
                frame = RN.latent_frame({"x_phys": np.ones(200), "z": zr[rng.integers(0, 3, 200)] * spread,
                                         "z_real": zr, "iteration": it}, np.arange(200))
                r.update_latent_plot(frame)
                if it == 1:
                    first = [np.array(l) for l in r.latent_limits]
            assert all((a <= b).all() for a, b in zip(r.latent_limits[0:1], first[0:1]))   # widened only
            assert r.latent_figure.axes[0].get_title().startswith("Iteration 2")
        r._end_live_latent()
        assert r.tabs.indexOf(r.latent_canvas) < 0


def test_snap_copy_outline_colour_new_database():
    """Snap rounds Measure and a copied point; a click on the part with a form open copies the point;
    outlines are magenta whatever the place."""
    from lsr_mmto.problem import selections as S

    class Picker:
        def GetCellId(self):
            return 2
    with scratch() as d:
        w = opened("L-bracket", d)
        w.show()
        w.snap.setCurrentIndex(w.snap.findText("Snap: 2 mm"))
        assert list(w.snapped([119.76, 80.02, 1.0])) == [120.0, 80.0, 0.0]
        w.measure.setChecked(True)
        w.on_left_button_press([0.2, 0.3, 1.0], Picker())
        w.on_left_button_press([29.7, 40.4, 1.0], Picker())
        assert w.measured.text().startswith("Distance 50 mm")                 # (0, 0) to (30, 40), snapped
        w.measure.setChecked(False)

        panel = w.open_supports_and_loads_panel()
        panel.add_region()                                                    # a form open: a click copies
        w.on_left_button_press([119.76, 80.02, 1.0], Picker())
        assert APP.clipboard().text() == "120, 80, 0"
        assert panel.stack.currentWidget().copied.text() == "Copied 120, 80, 0: paste it with Ctrl+V"
        panel.stack.currentWidget().findChild(SP.RegionDialog).reject()

        w.draw_outline("mmto_outline", S.box([0, 0, 0], [10, 10, 1]), SP.LOAD_COLOUR)
        colour = w.plotter.actors["mmto_outline"].GetProperty().GetColor()
        assert [round(c * 255) for c in colour] == [196, 0, 196]               # magenta, not the red given




def test_convert_units():
    """Convert to <units> units (Materials), on a copy of the Airplane bracket project (mmKS, its database in
    kg/m^3, Pa and USD/kg): the button shows only while the units differ; the window lists each column with its
    factor and the limits; OK writes the project's own copy with the values times the factors exactly and the
    limits converted, saves the project, and leaves no unit warning; the network file is byte-identical and
    still gives the materials back as closely; the shipped database is untouched. Unticked, the limits stay."""
    from lsr_mmto.materials.databases import DATABASE_DIR, database_table
    from lsr_mmto.materials.encoder import network_reconstruction
    shipped = os.path.join(DATABASE_DIR, "CustomAlloys20_3D")
    before = {ext: open(shipped + ext, "rb").read() for ext in (".xlsx", ".nt")}
    _, attributes, _, old = database_table(shipped + ".xlsx")
    errors = [r["percent_of_range"] for r in network_reconstruction(shipped + ".xlsx", shipped + ".nt")]
    factors = np.array([{"Density": 1e-12, "Youngs_Modulus": 1e-06, "Cost": 1000.0}.get(a, 1.0) for a in attributes])
    real, seen = materials_panel.ConvertUnitsDialog.exec_, []
    for tick in (True, False):
        with scratch() as d:
            w = opened("Airplane bracket", d)
            assert w.save_project_as(os.path.join(d, "A", "A.mmto"))
            bounds = [c["bound"] for c in w.project.formulation["constraints"]]
            expected = [new for _, _, _, new in w.project.unit_conversion()[1]] if tick else bounds
            m = materials_panel.MaterialsPanel(w, w.project)
            assert not m.convert_button.isHidden() and m.convert_button.text() == "Convert to mmKS units..."

            def answer(dialog):
                seen.append([label.text().strip() for label in dialog.findChildren(QtWidgets.QLabel)])
                dialog.limits.setChecked(tick)
                return 1
            materials_panel.ConvertUnitsDialog.exec_ = answer
            try:
                lines = m.convert_units()
            finally:
                materials_panel.ConvertUnitsDialog.exec_ = real
            assert seen[-1][1:] == ["Density: kg/m^3 -> t/mm^3, x 1e-12", "Young's modulus: Pa -> MPa, x 1e-06",
                                    "Cost: USD/kg -> USD/t, x 1000", "Limits written for the old numbers:",
                                    "Mass <= 4e+07 -> 4e-05 t", "Cost <= 7.5e+08 -> 0.75 USD"], seen[-1]
            p, folder = w.project, os.path.join(d, "A", "materials")
            assert p.database == "materials/CustomAlloys20_3D.xlsx" and sorted(os.listdir(folder)) == \
                ["CustomAlloys20_3D.nt", "CustomAlloys20_3D.xlsx"]
            _, _, units, new = database_table(os.path.join(folder, "CustomAlloys20_3D.xlsx"))
            stored = np.vectorize(lambda v: float("%.16g" % v))(old * factors)   # as an Excel file keeps them
            assert units == ["t/mm^3", "MPa", "USD/t", "MPa"] and np.array_equal(new, stored)
            assert [c["bound"] for c in p.formulation["constraints"]] == expected and (expected[0] == 4e7 / 1e12) == tick
            assert not any("the database gives" in x for x in p.warnings()) and m.convert_button.isHidden()
            assert open(os.path.join(folder, "CustomAlloys20_3D.nt"), "rb").read() == before[".nt"]
            assert not w.is_dirty() and Project.load(w.path).formulation == p.formulation
            assert lines[0].startswith("Database converted to mmKS units") and \
                any(line.startswith("Limits converted: Mass <= 4e+07 -> 4e-05 t") for line in lines) == tick
            again = [r["percent_of_range"] for r in network_reconstruction(os.path.join(folder, "CustomAlloys20_3D.xlsx"),
                                                                            os.path.join(folder, "CustomAlloys20_3D.nt"))]
            assert np.allclose(again, errors, atol=0.05), (max(again), max(errors))
            w.apply_panel(m)                                            # OK of the panel: nothing more changes
            assert p.database == "materials/CustomAlloys20_3D.xlsx" and not w.is_dirty()
    assert {ext: open(shipped + ext, "rb").read() for ext in (".xlsx", ".nt")} == before


def test_examples_open_as_templates():
    """An example opens as a template: Save asks for a project folder of one's own and copies everything
    into it, and the example's file never changes. In a project folder, a database chosen (a shipped one
    too) is copied into materials/, and New database writes there."""
    original = os.path.join(PROJECTS, "L-bracket.mmto")
    before = open(original).read()
    w = G.MMTOWindow()
    assert w.open_project(original) and w.template and not w.is_dirty()
    assert w.windowTitle() == "L-bracket (example) - LSR MMTO"
    real = G.ProjectFolderDialog.exec_
    with scratch() as d:
        try:
            def fill(dialog):
                assert dialog.stl is None and dialog.name.text() == "L-bracket"       # Save As: no STL asked
                dialog.location.setText(d)
                dialog.name.setText("My bracket")
                return 1
            G.ProjectFolderDialog.exec_ = fill
            w.project.settings["common"]["maxIterations"] = 7
            assert w.save_project()
        finally:
            G.ProjectFolderDialog.exec_ = real
        folder = os.path.join(d, "My bracket")
        assert w.path == os.path.join(folder, "My bracket.mmto") and not w.template and not w.is_dirty()
        assert w.windowTitle() == "My bracket - LSR MMTO" and w.own_folder()
        assert sorted(os.listdir(os.path.join(folder, "materials"))) == ["BM1_3Mats_2Attr.nt",
                                                                         "BM1_3Mats_2Attr.xlsx"]
        assert open(original).read() == before                                # the example is untouched
        assert w.database_folder() == os.path.join(folder, "materials")
        m = materials_panel.MaterialsPanel(w, w.project)
        m.combo.setCurrentText("6Materials_AF_3Attr")                         # a shipped one: copied in at OK
        w.apply_panel(m)
        assert w.project.database == "materials/6Materials_AF_3Attr.xlsx"
        assert os.path.isfile(os.path.join(folder, "materials", "6Materials_AF_3Attr.nt"))
        assert w.project.validate() == [] and "Material database: materials/6Materials_AF_3Attr.xlsx" in \
            w.facts.toolTip()


def test_part_facts_pca_live_colours_training():
    """The part's facts behind the (i), not over the 3D view; the PCA breakdown behind the Materials
    panel's (i), and its Values | Network error switch; the live design by density; the training's epochs in the Run status line."""
    from gui import draw_design
    s = draw_design.strength([0.1, 0.55, 1.0])
    assert np.allclose(s, [0.25, 0.625, 1.0])                                # pale to full colour
    assert draw_design.faded(np.array([[255.0, 0, 0]] * 3), s).tolist() == [[255, 191, 191], [255, 96, 96],
                                                                         [255, 0, 0]]
    with scratch() as d:
        w = opened("L-bracket", d)
        assert "geometry_info" not in w.plotter.actors
        facts = w.facts.toolTip().splitlines()
        assert facts[0] == "Part: LBracket_BM1.STL" and facts[1].startswith("Volume: ")
        assert facts[2].startswith("Size: ") and facts[3] == "Material database: BM1_3Mats_2Attr"

        panel = materials_panel.MaterialsPanel(w, w.project)
        tip = panel.latent_info.toolTip()
        assert "direction 1: " in tip and "The recommended latent dimension is the smallest whole number" in tip
        # Values | Network error: the same table, each cell the network's error with both values behind it
        assert panel.show_values.isChecked() and panel.show_errors.isEnabled()
        value = panel.table.item(0, 2).text()
        panel.show_errors.click()
        cells = [panel.table.item(i, j) for i in range(panel.table.rowCount())
                 for j in range(2, panel.table.columnCount())]
        assert cells and all(c.text().endswith("%") for c in cells)
        assert cells[0].toolTip().startswith("Database: ") and "\nNetwork: " in cells[0].toolTip()
        panel.show_values.click()
        assert panel.table.item(0, 2).text() == value

        w.project.n_dof, w.project.single_layer = 3000, True
        mesh = w.preview_mesh()
        n = int(mesh.num_elems)
        x = np.full(n, 0.5)
        x[:10] = 0.05                                                         # almost empty: not drawn
        w.set_design(lambda win: draw_design.live(win, mesh, x, np.zeros(n, dtype=int), ["#ff0000"]))
        blocks = w.plotter.actors["result_design"].GetMapper().GetInput()
        assert blocks.n_cells == 6 * (n - 10) and not w.part_actor().GetVisibility()

        r = w.open_run_panel()
        r.optimization_running, r.optimizing, r.epochs = True, False, 30000
        try:
            r._training_progress("[STD] Epoch 5000: reconLoss 1.0e-03, klLoss 2.0e-01, loss 1.0e-03")
            assert r.status.text() == "Training the material network: epoch 5,000 / 30,000"
            r.optimizing = True                                               # the optimization has started
            r._training_progress("[STD] Epoch 10000: reconLoss 1.0e-03")
            assert r.status.text().endswith("epoch 5,000 / 30,000")
        finally:
            r.optimization_running = False


def test_results_compare_delete():
    """Results: the project's runs, newest first; a run shown as each of Material, Density, Stress and
    Solid; two runs compared side by side; another place's run opened for looking only; Delete, in
    Results and in the Run strip, removes a run's folder."""
    from gui import draw_design
    with scratch() as d:
        w = opened("L-bracket", d)
        w.show()
        w.project.n_dof, w.project.single_layer = 3000, True
        w.project.settings["common"]["maxIterations"] = 2
        assert w.save_project_as(os.path.join(d, "P", "P.mmto")) and w.own_folder()
        r = w.open_run_panel()
        folders = []
        for _ in range(2):
            assert r.start_optimization()
            assert _wait(lambda: not r.optimization_running), "the run did not finish"
            assert r.status.text().startswith("Run finished"), r.status.text()
            folders.append(r.save_dir)
        assert all(os.path.dirname(f) == os.path.join(d, "P", "results") for f in folders)
        assert r.results_button.isVisibleTo(r) and r.delete_button.isVisibleTo(r)

        r.results_button.click()                                              # Open in Results
        res = w.results_window
        assert res.folders() == folders[::-1] and res.current == folders[1]
        assert [res.tabs.tabText(i) for i in range(res.tabs.count())] == ["Convergence", "Latent space",
                                                                          "Summary"]
        shown = [res.field.itemText(i) for i in range(res.field.count())]
        assert shown == ["Material", "Density", "Stress", "Solid", "Smooth (exported)"], shown   # the views a
        assert not res.field.model().item(4).isEnabled()                      # structural run saved; not exported
        for field in shown[:-1]:
            res.field.setCurrentText(field)
            assert "result_design" in w.plotter.actors and w.view == "design", field
            assert bool(w.design_bars) == (field in ("Density", "Stress")), (field, w.design_bars)

        for i in range(2):
            res.list.item(i).setCheckState(QtCore.Qt.Checked)
        assert res.compare_button.isEnabled() and res.compare()
        assert {"result_design", "result_design_2"} <= set(w.plotter.actors)
        cw = res.compare_window                                               # a window of its own
        assert cw.isWindow() and [cw.tabs.tabText(i) for i in range(cw.tabs.count())] == \
            ["Convergence", "Latent space", "Summary"]
        assert not any(isinstance(cw.tabs.widget(i), QtWidgets.QLabel) for i in range(cw.tabs.count()))
        assert "Iterations" in cw.tabs.widget(2).findChild(QtWidgets.QTextBrowser).toPlainText()
        cw.close()                                                            # closing it ends the comparison
        assert res.comparing is None and "result_design_2" not in w.plotter.actors
        assert res.compare()

        moved = shutil.copytree(os.path.join(d, "P"), os.path.join(d, "Q"))  # a project zipped and sent
        other = os.path.join(moved, "results", os.path.basename(folders[0]))
        assert res.add_other(other) == other and res.current == other
        assert res.tabs.count() == 3 and not res.delete_button.isEnabled()  # for looking only
        assert res.select(folders[0])
        with answers(Box.Yes):
            res.delete_run()
        assert not os.path.exists(folders[0]) and res.folders() == [folders[1], other]
        with answers(Box.Yes):
            r.delete_run()                                                    # the run the strip just made
        assert not os.path.exists(folders[1]) and res.folders() == [other]
        assert w.view == "setup" and w.part_actor().GetVisibility()


def test_load_arrow_outside_the_part():
    """A load's arrow is drawn outside the part whichever way it points: pushing in, its tip on the face;
    pulling away, it starts on the face (it used to sit inside the part, hidden)."""
    from lsr_mmto.problem import selections as S
    with scratch() as d:
        w = opened("L-bracket", d)
        w.project.n_dof, w.project.single_layer = 3000, True
        w.project.add_selection("right end", S.face("xmax"))
        x_face = float(np.max(np.asarray(w.preview_mesh().node_xyz)[:, 0]))
        for fx, outside in ((5.0, lambda b: b[0] >= x_face - 1e-6), (-5.0, lambda b: b[0] >= x_face - 1e-6)):
            w.project.bcs = [bc for bc in w.project.bcs if bc.get("on") != "right end"]
            w.project.add_bc("force", "right end", total=[fx, 0.0, 0.0])
            w.draw_supports_and_loads()
            i = len(w.project.bcs) - 1
            bounds = w.plotter.actors[f"mmto_arrow{i}"].GetBounds()          # xmin, xmax, ...
            assert outside(bounds), (fx, bounds, x_face)                       # no part of it inside x < face
            assert bounds[1] > x_face + 1.0, (fx, bounds)                      # and it reaches out beyond it


def test_stop_while_training():
    """Stop while the material network trains: the run ends there, no network is saved and the run's
    folder is removed; the status says so."""
    import openpyxl
    with scratch() as d:
        w = opened("L-bracket", d)
        w.show()
        from lsr_mmto.materials.databases import write_database
        m = materials_panel.MaterialsPanel(w, w.project)
        path = write_database(os.path.join(d, "fresh.xlsx"), ["Youngs_Modulus", "Poissons_Ratio"], ["Pa", "-"],
                              ["Soft", "Mid", "Stiff"], [None] * 3, ["#ff0000", "#00ff00", "#0000ff"],
                              [[1e6, 0.30], [2e6, 0.32], [4e6, 0.35]])
        m.set_database(path)
        w.apply_panel(m)
        w.project.settings["vae"]["numEpochs"] = 200000
        w.project.settings["vae"]["maxAttributeErrorPercent"] = 1e-9          # it would train for minutes
        w.project.n_dof, w.project.single_layer = 3000, True
        assert w.project.validate() == [], w.project.validate()
        assert w.save_project()
        r = w.open_run_panel()
        assert r.start_optimization()
        assert _wait(lambda: r.status.text().startswith("Training the material network"), 300), r.status.text()
        r.stop_optimization()
        assert _wait(lambda: not r.optimization_running, 120)
        assert r.status.text() == "Stopped while training the material network. Nothing was saved."
        assert not os.path.exists(os.path.splitext(path)[0] + ".nt")         # no half-trained network
        assert not os.path.exists(r.save_dir)                                 # no empty run folder


def test_export_stl():
    """Export STL of the design shown: design.stl and a file per material used, in the run's folder, all closed;
    the pieces add up to their whole. Results' Show list has Smooth (exported), greyed with a tip until the
    run's folder holds the files; after the export it is chosen and the pieces are drawn, and can be pulled
    apart. What the folder holds decides: a Results panel opened later finds the files, and finds none once
    they are gone."""
    import trimesh
    from gui import results_panel
    from gui.results_panel import SMOOTH
    with scratch() as d:
        w = opened("L-bracket", d)
        w.show()
        w.project.n_dof, w.project.single_layer = 3000, True
        w.project.settings["common"]["maxIterations"] = 40                 # converged enough: not grey
        assert w.save_project_as(os.path.join(d, "P", "P.mmto"))
        r = w.open_run_panel()
        assert r.start_optimization() and _wait(lambda: not r.optimization_running)
        res = w.open_results_panel(r.save_dir)
        w.update_top_row()
        assert res.current == r.save_dir and w.shown_run()[0] == r.save_dir and w.export_button.isEnabled()
        smooth = res.field.model().item(res.field.findText(SMOOTH))
        assert not smooth.isEnabled() and smooth.toolTip() == "Export STL first" and not res.apart.isVisible()
        assert w.export_stl()                                                 # Export STL, in the top row
        assert _wait(lambda: not w.exporting, 300)
        assert "STL files saved" in w.banner.text() and "Open folder" in w.banner.text()
        folder, report = w.exported_last
        assert folder == r.save_dir and not report.from_regions                # the smooth surface, as before
        assert os.path.isfile(os.path.join(folder, "design.stl")) and report.whole_closed
        assert report.pieces and all(p.closed and os.path.isfile(p.path) for p in report.pieces), report.lines()
        assert abs(report.sum_off) < 1e-6                                     # no gap, no overlap
        assert abs(report.whole_volume / report.design_volume - 1) < 0.3      # PyTO-AD's own bound
        for p in report.pieces:
            t = trimesh.load(p.path, force="mesh")
            assert t.is_watertight and t.volume > 0, p.label
        assert res.field.currentText() == SMOOTH and smooth.isEnabled() and res.apart.isVisible()
        assert any(str(n).startswith("result_smooth") for n in w.plotter.actors)
        res.apart.setChecked(True)
        assert w.view == "design" and any(str(n).startswith("result_smooth") for n in w.plotter.actors)
        assert w.shown_run()[0] == r.save_dir                                 # Export STL works on this view too
        res.close()
        res = results_panel.ResultsWindow(w)                                  # a new panel: the folder decides
        res.select(r.save_dir)
        smooth = res.field.model().item(res.field.findText(SMOOTH))
        assert smooth.isEnabled() and res.field.currentText() == "Material"
        res.field.setCurrentText(SMOOTH)
        assert any(str(n).startswith("result_smooth") for n in w.plotter.actors)
        res.field.setCurrentText("Material")
        for f in os.listdir(folder):
            if f.lower().endswith(".stl"):
                os.remove(os.path.join(folder, f))
        res.select(r.save_dir)
        assert not res.field.model().item(res.field.findText(SMOOTH)).isEnabled()


def test_earlier_run_after_scale_part():
    """A run made before Scale part is drawn and exported at the size it was run at: its mesh is the rebuilt one
    scaled back (the run's element size, the same nodes), Export STL gives the same extent, and the window's own
    mesh of the scaled part is left as it is. A part changed in another way still cannot show the run."""
    import trimesh
    from stl import mesh as stl_mesh
    from gui import draw_design
    from lsr_mmto.results import load_results, stl_export
    with scratch() as d:
        w = opened("L-bracket", d)
        w.project.n_dof, w.project.single_layer = 3000, True
        w.project.settings["common"]["maxIterations"] = 40                 # a design with voids to export
        assert w.save_project_as(os.path.join(d, "P", "P.mmto"))
        r = w.open_run_panel()
        assert r.start_optimization() and _wait(lambda: not r.optimization_running)
        folder = r.save_dir

        def exported_extent(mesh):
            report = stl_export.export(load_results.load(os.path.join(folder, "run_data")), mesh, folder,
                                       materials=False)
            assert report.whole_closed, report.lines()
            return trimesh.load(os.path.join(folder, "design.stl"), force="mesh").bounds

        run = load_results.load(os.path.join(folder, "run_data"))
        before = draw_design.run_mesh(w, folder, run)
        nodes, extent = np.array(before.node_xyz, dtype=float), exported_extent(before)
        assert w.scale_part(2.0)
        after = draw_design.run_mesh(w, folder, load_results.load(os.path.join(folder, "run_data")))
        assert after is not before and np.allclose(after.elem_size, run.elem_size, rtol=1e-12, atol=0)
        assert np.allclose(after.node_xyz, nodes, rtol=1e-12, atol=1e-12 * np.abs(nodes).max())
        assert np.allclose(exported_extent(after), extent, rtol=1e-9, atol=1e-9 * np.abs(extent).max())
        own = draw_design.mesh_of(w, w.project)                              # the scaled part's own mesh
        assert np.allclose(own.elem_size, 2 * np.asarray(run.elem_size), rtol=1e-12, atol=0)

        part = stl_mesh.Mesh.from_file(w.project.stl_path())                 # stretched along x only
        part.vectors[:, :, 0] *= 1.5
        part.save(w.project.stl_path())
        draw_design.forget_meshes(w.project.stl_path())
        w._mesh, w._mesh_key = None, None
        try:
            draw_design.run_mesh(w, folder, load_results.load(os.path.join(folder, "run_data")))
            raise AssertionError("a run was shown on a part of another shape")
        except draw_design.MeshError as e:
            assert "its part has changed since the run" in str(e), e


def test_pieces_pulled_apart_in_a_row():
    """Pull the material pieces apart: the pieces side by side along x, largest first and in place, each next
    one moved by the part's width and a tenth more, so no two overlap; the legend names them."""
    import types
    import pyvista as pv
    from gui import draw_design
    with scratch() as d:
        boxes = {"small": (0, 4, 0, 10, 0, 1), "large": (0, 10, 0, 6, 0, 1), "middle": (4, 10, 6, 10, 0, 1)}
        pieces = []
        for k, (label, bounds) in enumerate(boxes.items()):
            path = os.path.join(d, f"design_{label}.stl")
            pv.Box(bounds=bounds).triangulate().save(path)
            pieces.append((path, ["#ff0000", "#00ff00", "#0000ff"][k], label))
        window = types.SimpleNamespace(plotter=pv.Plotter(off_screen=True), design_legend=[])
        together = draw_design.smooth(window, pieces)
        assert [m.bounds for m in together] == [pv.read(p).bounds for p in (pieces[1][0], pieces[0][0], pieces[2][0])]
        window.design_legend.clear()
        apart = draw_design.smooth(window, pieces, apart=True)
        xs = [(m.bounds[0], m.bounds[1]) for m in apart]
        assert xs[0] == (0.0, 10.0)                                           # the largest stays in place
        assert np.allclose([lo for lo, _ in xs[1:]], [11.0, 26.0])            # moved by 1 x 11 and 2 x 11
        for i in range(len(apart)):
            for j in range(i + 1, len(apart)):
                a, b = np.reshape(apart[i].bounds, (3, 2)), np.reshape(apart[j].bounds, (3, 2))
                assert np.any((a[:, 1] < b[:, 0]) | (b[:, 1] < a[:, 0])), (i, j)   # apart along some axis
        assert [label for _, label, _ in window.design_legend[0]] == ["large", "small", "middle"]
        window.design_legend.clear()                                          # one piece of two materials
        draw_design.smooth(window, pieces[:1], absent=[("#00ff00", "other")])
        assert [(label, share) for _, label, share in window.design_legend[0]] == [("small", 1.0), ("other", 0.0)]
        window.plotter.close()


def test_constraints_named_in_the_plots():
    """The Progress plot of Current run names each constraint in the Formulation panel's words with its
    limit; the two-run plot of Results names them as the runs do (both names when the runs differ)."""
    import types
    from gui import results_panel
    with scratch() as d:
        w = opened("L-bracket", d)
        r = w.open_run_panel()
        r.project = w.project
        r.constraint_names = r._constraint_names()
        assert r.constraint_names[0].startswith("Material volume fraction of ") and \
            r.constraint_names[0].endswith("<= 0.15"), r.constraint_names
        r.max_iterations, r.started = 10, 0.0
        r.update_history_plot({"objective": [1.0, 0.9], "constraints": [[0.1, 0.2, 0.3], [0.0, 0.1, 0.2]],
                               "iteration": 1})
        legend = [t.get_text() for t in r.history_figure.axes[1].get_legend().get_texts()]
        assert legend == r.constraint_names
        r.close()

    def run(labels):
        return types.SimpleNamespace(history={"J_phys": [1.0, 0.5], "constraints": [[0.0] * len(labels)] * 2},
                                     constraint_spec=[[k] for k in labels], constraint_label=lambda i: labels[i])
    fig = results_panel.convergence_overlaid({"a": run(["Mass", "Cost"]), "b": run(["Mass", "Volume"])})
    texts = [t.get_text() for t in fig.legends[0].get_texts()]
    assert texts == ["a", "b", "Mass", "Cost / Volume"], texts


def test_material_legend():
    """The legend at the top left of the 3D view: each material present with its share, largest first; while a
    run goes, by density among the elements drawn; on a finished run's Material view, by solid elements; two
    runs compared, a left and a right column. The database decides whether it shows: a database of several
    materials keeps it when the design settles on one of them ("Red  100%"), a database of one material has
    none. Hidden on the Part view and on Density; it follows the light and dark view."""
    import types
    from gui import draw_design
    with scratch() as d:
        w = opened("L-bracket", d)
        w.show()
        w.project.n_dof, w.project.single_layer = 3000, True
        mesh = w.preview_mesh()
        n = int(mesh.num_elems)
        x = np.zeros(n)
        x[:100], x[100:150], x[150:160] = 1.0, 0.8, 0.05                      # the last ones too faint to count
        material = np.zeros(n, dtype=int)
        material[100:] = 2
        colours, labels = ["#ff0000", "#00ff00", "#0000ff"], ["Red", "Green", "Blue"]
        w.set_design(lambda win: draw_design.live(win, mesh, x, material, colours, labels))
        APP.processEvents()
        assert w.legend.isVisible() and w.legend.geometry().topLeft() == QtCore.QPoint(12, 12)
        (column,) = w.design_legend
        assert [r[:2] for r in column] == [("#ff0000", "Red"), ("#0000ff", "Blue"), ("#00ff00", "Green")]
        assert np.allclose([r[2] for r in column], [100 / 140, 40 / 140, 0])
        text = w.legend.text()
        assert "Red" in text and "71%" in text and "29%" in text and "Green" not in text
        assert text.index("Red") < text.index("Blue")
        w.set_dark_view(True)
        assert w.legend.property("dark") is True
        w.set_dark_view(False)
        w.show_setup_view()
        assert not w.legend.isVisible()                                       # the Part view
        w.set_design(lambda win: draw_design.live(win, mesh, x, np.zeros(n, dtype=int), colours, labels))
        assert w.legend.isVisible()                                           # settled on one material
        assert "Red" in w.legend.text() and "100%" in w.legend.text() and "Blue" not in w.legend.text()
        w.set_design(lambda win: draw_design.live(win, mesh, x, np.zeros(n, dtype=int), colours[:1], labels[:1]))
        assert not w.legend.isVisible()                                       # a database of one material
        solid = np.zeros(n, dtype=bool)
        solid[:30] = True
        indices = np.where(np.arange(n) < 10, 1, 0)
        run = types.SimpleNamespace(material_colors=colours, material_labels=labels, material_indices=indices,
                                    solid=solid, x_cont=x, result_info={})
        w.set_design(lambda win: draw_design.record(win, run, mesh, "Material"))
        assert w.legend.isVisible() and w.design_legend[0][0][1] == "Red" and np.isclose(w.design_legend[0][0][2], 2 / 3)
        w.set_design(lambda win: draw_design.record(win, run, mesh, "Density"))
        assert not w.legend.isVisible()
        other = types.SimpleNamespace(**{**vars(run), "material_indices": np.full(n, 2)})

        def both(win):
            draw_design.record(win, run, mesh, "Material", name="design")
            draw_design.record(win, other, mesh, "Material", shift=100.0, name="design_2")
        w.set_design(both, wide=True)
        assert w.legend.isVisible() and len(w.design_legend) == 2
        assert "<i>left</i>" in w.legend.text() and "<i>right</i>" in w.legend.text() and "Blue" in w.legend.text()


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


def test_show_list_is_the_runs_own():
    """The Results panel's Show list holds the views the shown runs saved, so a run of another physics
    shows its own results and never another physics' Stress."""
    import types
    from gui import draw_design
    structural = types.SimpleNamespace(result_info={"von_mises": {"view": "Stress"}})
    other = types.SimpleNamespace(result_info={"mean_value": {"view": "Value"}})
    assert draw_design.views(structural) == ("Material", "Density", "Stress", "Solid")
    assert draw_design.views(other) == ("Material", "Density", "Value", "Solid")
    assert draw_design.views(structural, other) == ("Material", "Density", "Stress", "Value", "Solid")
    assert draw_design.views() == ("Material", "Density", "Solid")

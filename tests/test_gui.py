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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils import project_paths  # noqa: E402

project_paths.bootstrap()

from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _quiet(kind, context, message):
    """Qt's messages, less the ones the off-screen mode prints for every 3D view it cannot draw."""
    if not any(s in message for s in ("OpenGL", "propagateSizeHints")):
        sys.stderr.write(message + "\n")


QtCore.qInstallMessageHandler(_quiet)

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
    assert {c["name"] for c in pyto_ad.MainWindow.BUTTON_CONFIG} >= set(G.SIDEBAR) - set(G.PANELS) - \
        {"Supports and loads"}
    assert all(hasattr(pyto_ad.MainWindow, m) for m in ("update_LivVar", "_raise_if_open",
                                                         "start_box_node_selection"))
    assert all(hasattr(pyto_ad.StructuralLoadsWindow, m) for m in ("redraw_loads", "refresh_applied_list"))
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
        assert "is not an LSR MMTO project (.mmto)" in G.mmto_file_error(pyto)
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


# ---------------------------------------------------------------------- Materials, Formulation, Settings
from gui import panels  # noqa: E402


def opened(name, d):
    w = G.MMTOWindow()
    assert w.open_project(shutil.copy(os.path.join(PROJECTS, f"{name}.mmto"), d))
    return w


def test_panels_with_no_edits_change_nothing():
    """OK with no edits, on every project and every panel: the project is as it was, to the last digit
    (the U-bracket's mass bound is 0.0035000000000000005)."""
    assert list(G.MMTOWindow().sidebar_buttons) == list(G.SIDEBAR)
    with scratch() as d:
        for name in NAMES:
            w = opened(name, d)
            for cls in (panels.MaterialsPanel, panels.FormulationPanel, panels.SettingsPanel):
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
                if isinstance(self, panels.SettingsPanel):
                    self.rows[("common", "maxIterations")][0].set(5)
                return 0
            QtWidgets.QDialog.exec_ = edit_then_cancel
            w.open_panel("Settings")
        finally:
            QtWidgets.QDialog.exec_ = real
        assert w.project.settings["common"]["maxIterations"] == before != 5 and not w.is_dirty()


def test_settings_panel_covers_every_setting():
    from project import DEFAULT_SETTINGS, VAE_PARAMS  # type: ignore
    keys = panels.all_setting_keys()
    want = {(s, k) for s, v in DEFAULT_SETTINGS.items() for k in v} | {("vae", k) for k in VAE_PARAMS} | \
        {("mesh", "n_dof"), ("mesh", "single_layer")}
    assert len(keys) == len(set(keys)) and set(keys) == want, set(keys) ^ want
    assert all(k in panels.LABELS and k in panels.HELP for _, k in keys)


def test_settings_panel_edits():
    with scratch() as d:
        w = opened("L-bracket", d)
        s = panels.SettingsPanel(w, w.project)
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
        f = panels.FormulationPanel(w, w.project)
        model = f.objective.model()
        enabled = {model.item(i).text(): model.item(i).isEnabled() for i in range(model.rowCount())}
        assert enabled == {"Compliance": True, "Mass": False, "Stress (p-norm)": True,
                           "Failure factor (p-norm)": False}
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
        f = panels.FormulationPanel(w, w.project)
        f.add_row(None)
        f.tau.setText("0")
        assert f.problems() == ["constraint 4: choose a quantity", "tau must be a number greater than 0"]
        w.project.formulation["constraints"][0]["expression"] = "MaterialVolumeFraction(1, tau=0.7)"   # by hand
        assert "the material volume fraction constraints use different tau (0.3, 0.7)" in w.project.validate()[0]
    with scratch() as d:
        w = opened("Airplane bracket", d)
        f = panels.FormulationPanel(w, w.project)
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
        f = panels.FormulationPanel(w, w.project)
        bound = f.rows[0]["widgets"]["bound"]
        assert bound.text() == "0.0035" and bound.value() == 0.0035000000000000005
        assert f.apply(w.project) == []
        bound.setText("0.0035")                                                # retyped as shown: kept exactly
        assert bound.value() == 0.0035000000000000005
        bound.setText("0.00350")                                               # a typed change: replaced
        assert bound.value() == 0.0035
    assert panels.bound_unit("Compliance", "mmKS") == "N*mm" and panels.bound_unit("Mass", "none") == \
        "no unit system declared"


def test_materials_panel():
    from databases import DATABASE_DIR  # type: ignore
    with scratch() as d:
        w = opened("L-bracket", d)
        m = panels.MaterialsPanel(w, w.project)
        assert m.table.rowCount() == 3 and m.table.item(0, 0).text() == "Mat1_Red"
        assert "latent dimension 2 recommended" in m.report.toPlainText()
        m.combo.setCurrentText("CustomAlloys20_3D")
        assert m.table.rowCount() == 20 and m.table.item(0, 0).text() == "A"
        lines = m.apply(w.project)
        vae = w.project.settings["vae"]
        assert w.project.database == "CustomAlloys20_3D" and vae["latentDim"] == 3 and vae["numEpochs"] == 100000
        assert lines[0] == "Database: CustomAlloys20_3D" and "Latent dimension 2 -> 3" in lines[1]
        m = panels.MaterialsPanel(w, w.project)
        m.set_database(os.path.join(ROOT, DATABASE_DIR, "BM1_3Mats_2Attr.xlsx"))     # a shipped file: by name
        assert m.project.database == "BM1_3Mats_2Attr"
        own = shutil.copy(os.path.join(ROOT, DATABASE_DIR, "BM1_3Mats_2Attr.xlsx"), os.path.join(d, "mine.xlsx"))
        m.set_database(own)                                                    # any other: relative to the project
        assert m.project.database == "mine.xlsx"
        m.apply(w.project)
        assert w.project.settings["vae"]["latentDim"] == 2 and w.project.validate() == []


def test_vae_tab():
    """With a trained network in use the VAE settings are its own and greyed out; the tab says which
    network the database has and shows the VAE checks as you edit. No "?" button on our panels."""
    from databases import DATABASE_DIR  # type: ignore
    with scratch() as d:
        w = opened("L-bracket", d)
        s = panels.SettingsPanel(w, w.project)
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
        s = panels.SettingsPanel(w, w.project)
        assert s.rows[("vae", "latentDim")][0].get() == 2
        assert s.apply(w.project) == ["Settings: Latent dimension 4 -> 2",
                                      "Latent dimension: set to the trained network's"]

        shutil.copy(os.path.join(ROOT, DATABASE_DIR, "BM1_3Mats_2Attr.xlsx"), os.path.join(d, "ggd.xlsx"))
        w.project.database = "ggd.xlsx"                                       # one of yours, no network yet
        s = panels.SettingsPanel(w, w.project)
        assert s.vae_status.text() == "Network: ggd.nt, none yet. The first run trains ggd.nt."
        use = s.rows[("vae", "use_pretrained_vae")][0].widget
        assert not use.isEnabled() and not use.isChecked()                  # shown unticked: nothing to use yet
        assert s.rows[("vae", "latentDim")][0].widget.isEnabled()
        assert s.apply(w.project) == [] and w.project.settings["vae"]["use_pretrained_vae"] is True   # kept
    f = panels.FormulationPanel(G.MMTOWindow(), Project.load(os.path.join(PROJECTS, "beam.mmto")))
    f.rows[0]["widgets"]["bound"].setText("abc")
    assert f.problems() == ["constraint 1: the bound is not a number"]


# ---------------------------------------------------------------------- supports and loads
from gui import supports as SP  # noqa: E402


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
                      ("2 + 2", "only numbers, text, lists and selections"), ("face(", "not a selection")):
        try:
            SP.parse_selection(bad)
        except ValueError as e:
            assert says in str(e), (bad, e)
        else:
            raise AssertionError(f"accepted {bad}")


def test_region_and_bc_forms():
    import selections as S  # type: ignore
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
        assert r.problems() == ["there is already a place named clamp"]
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
    """Add offers both ways; an empty list says what to do; New shape makes a place without leaving the
    dialog; a pick takes the list's place and Apply brings the list back with the new entry selected;
    each warning is said once."""
    import selections as S  # type: ignore
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
        assert b.on.currentText() == "place1" and "place1" not in w.project.selections   # not yet: on OK
        assert b.new_regions["place1"].to_dict() == S.face("xmin").to_dict()
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
        text = w.message_text.toPlainText()
        assert text.count("Warning: the database gives Youngs Modulus") == 1
        w.apply_panel(type("P", (), {"apply": lambda self, p: ["changed"]})())   # a change: said again? no
        assert w.message_text.toPlainText().count("Warning: the database gives Youngs Modulus") == 1


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
    assert G.MMTOWindow.length_format(200) == "%.0f" and G.MMTOWindow.length_format(0.0285) % 0.0164 == "0.016"


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
        r.clear_shown()                                                       # the Clear button
        assert not on_part()
        r.show_preview()
        r.type.setCurrentText("Box")                                          # another type: goes too
        assert not on_part()
        r.type.setCurrentText("Flat cut")
        r.inputs["value"].setText("20")
        r.show_preview()
        r.reject()                                                            # Cancel: nothing left behind
        assert not on_part()


def test_pages():
    """Forms opened from Supports and loads are pages of its strip, one at a time: Enter by hand, then
    New place over it; OK goes back one page each time. Combined places follow their parts by name."""
    import selections as S  # type: ignore
    with scratch() as d:
        w = opened("L-bracket", d)
        w.show()
        panel = w.open_supports_and_loads_panel()
        home = panel.stack.currentWidget()
        b = panel.add_bc()                                                    # a page, not a window
        assert panel.stack.count() == 2 and b.isVisible() and not home.isVisible()
        r = b.new_shape()                                                     # New place: a page over it
        assert panel.stack.count() == 3 and panel.stack.currentWidget().mouse.text().startswith("Hover")
        panel.show_mouse("x 1   y 2   z 0 mm")
        assert panel.stack.currentWidget().mouse.text() == "Mouse on the part: x 1   y 2   z 0 mm"
        r.type.setCurrentText("Face of the part")
        r.accept()
        assert panel.stack.count() == 2 and b.on.currentText() == "place1"     # back to the support/load form
        b.kind.setCurrentText("Fixed support")
        b.accept()
        assert panel.stack.count() == 1 and panel.stack.currentWidget() is home
        assert w.project.bcs[-1] == {"kind": "fixed", "on": "place1", "dofs": "xyz"}
        assert w.project.selections["place1"].to_dict() == S.face("xmin").to_dict()

        r = panel.add_region()                                                # Combined places: by name, live
        r.type.setCurrentText("Combined places")
        r.inputs["a"].setCurrentText("clamp")
        r.inputs["b"].setCurrentText("corner")
        r.name.setText("both")
        r.accept()
        assert w.project.selections["both"].to_dict() == S.union(S.place("clamp"), S.place("corner")).to_dict()
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
        w.project.selections["grip"] = S.union(S.place("grip2"))              # a loop: refused
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
    """Keep solid on a place that picks nodes only (rows, picked faces): the elements touching it."""
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
    """A place's shape is drawn with its nodes; clicking a row shows only that one; Measure gives the
    distance between two clicks; the mouse lookup never runs while a button is held or one is running."""
    import selections as S  # type: ignore
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

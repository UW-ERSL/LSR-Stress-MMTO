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
give are greyed out with the reason, a new database brings its VAE settings. About half a minute.

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

from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402

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
    assert {c["name"] for c in pyto_ad.MainWindow.BUTTON_CONFIG} >= set(G.SIDEBAR) - set(G.PANELS)
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
    from project import DEFAULT_SETTINGS, SETTING_DOCS, VAE_PARAMS  # type: ignore
    keys = panels.all_setting_keys()
    want = {(s, k) for s, v in DEFAULT_SETTINGS.items() for k in v} | {("vae", k) for k in VAE_PARAMS} | \
        {("mesh", "n_dof"), ("mesh", "single_layer")}
    assert len(keys) == len(set(keys)) and set(keys) == want, set(keys) ^ want
    assert all(k in panels.LABELS and (k in SETTING_DOCS or k in panels.MESH_DOCS) for _, k in keys)


def test_settings_panel_edits():
    with scratch() as d:
        w = opened("L-bracket", d)
        s = panels.SettingsPanel(w, w.project)
        row, label, reset = s.rows[("common", "maxIterations")]
        assert not label.font().bold() and not reset.isEnabled()
        row.set(5)
        s._mark("common", "maxIterations")
        assert label.font().bold() and reset.isEnabled()
        assert s.rows[("group", "Run length")][0].text() == "Run length"
        s.rows[("advanced", "timeLimit")][0].set(60)
        s._mark("advanced", "timeLimit")
        assert s.rows[("group", "Run length")][0].text() == "Run length  (1 changed)"
        s.rows[("common", "material_filter_radius")][0].none.setChecked(True)  # same as the density filter
        s.rows[("mesh", "n_dof")][0].widget.setText("abc")
        assert s.problems() == ["Mesh dofs (n_dof) is not a number"]
        s.reset("mesh", "n_dof")
        assert s.problems() == []
        lines = s.apply(w.project)
        assert w.project.settings["common"]["maxIterations"] == 5 and w.project.n_dof == 30000
        assert w.project.settings["advanced"]["timeLimit"] == 60
        assert w.project.settings["common"]["material_filter_radius"] is None
        assert lines and "maxIterations 300 -> 5" in lines[0]
        assert w.is_dirty()


def test_formulation_panel():
    with scratch() as d:
        w = opened("L-bracket", d)                                            # BM1: no Density, no Yield_Strength
        f = panels.FormulationPanel(w, w.project)
        model = f.objective.model()
        enabled = {model.item(i).text(): model.item(i).isEnabled() for i in range(model.rowCount())}
        assert enabled == {"Compliance": True, "Mass": False, "StressPNorm": True, "FailureFactorPNorm": False}
        assert f.reason("Mass").startswith("Mass needs a Density column; the database has Youngs_Modulus")
        assert f.reason("Cost") == ("Cost needs a Density and a Cost column; the database has Youngs_Modulus, "
                                    "Poissons_Ratio")
        f.rows[2]["widgets"]["bound"].set_value(0.2)
        row = f.add_row(None)
        row["widgets"]["quantity"].setCurrentText("MaterialVolumeFraction")
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
        f.rows[0]["widgets"]["quantity"].setCurrentText("MaterialVolumeFraction")
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
        assert lines[0] == "Database: CustomAlloys20_3D" and "latentDim 2 -> 3" in lines[1]
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
        assert s.apply(w.project) == ["Settings: latentDim 4 -> 2", "latentDim: set to the trained network's"]

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

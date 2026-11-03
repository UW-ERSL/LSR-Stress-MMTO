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
from gui import formulation_panel, materials_panel, settings_panel  # noqa: E402


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
                      ("2 + 2", "only numbers, text, lists and selections"), ("face(", "not a selection")):
        try:
            SP.parse_selection(bad)
        except ValueError as e:
            assert says in str(e), (bad, e)
        else:
            raise AssertionError(f"accepted {bad}")


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
        w.snap.setCurrentIndex([t for t, _ in G.SNAPS].index("Snap: 1"))
        assert list(w.snapped([119.76, 80.02, 1.0])) == [120.0, 80.0, 1.0]
        w.measure.setChecked(True)
        w.on_left_button_press([0.2, 0.3, 1.0], Picker())
        w.on_left_button_press([29.7, 40.4, 1.0], Picker())
        assert w.measured.text().startswith("Distance 50 mm")                 # (0, 0) to (30, 40), snapped
        w.measure.setChecked(False)

        panel = w.open_supports_and_loads_panel()
        panel.add_region()                                                    # a form open: a click copies
        w.on_left_button_press([119.76, 80.02, 1.0], Picker())
        assert APP.clipboard().text() == "120, 80, 1"
        assert panel.stack.currentWidget().copied.text() == "Copied 120, 80, 1: paste it with Ctrl+V"
        panel.stack.currentWidget().findChild(SP.RegionDialog).reject()

        w.draw_outline("mmto_outline", S.box([0, 0, 0], [10, 10, 1]), SP.LOAD_COLOUR)
        colour = w.plotter.actors["mmto_outline"].GetProperty().GetColor()
        assert [round(c * 255) for c in colour] == [196, 0, 196]               # magenta, not the red given




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
        for field in draw_design.FIELDS:
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
    the pieces add up to their whole; they are drawn on the part, and can be pulled apart."""
    import trimesh
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
        assert w.export_stl()                                                 # Export STL, in the top row
        assert _wait(lambda: not w.exporting, 300)
        assert "STL files saved" in w.banner.text() and "Open folder" in w.banner.text()
        folder, report = res.smooth
        assert folder == r.save_dir and not report.whole_error, report.whole_error
        assert os.path.isfile(os.path.join(folder, "design.stl")) and report.whole_closed
        assert report.pieces and all(p.closed and os.path.isfile(p.path) for p in report.pieces), report.lines()
        assert abs(report.sum_off) < 1e-6                                     # no gap, no overlap
        assert abs(report.whole_volume / report.design_volume - 1) < 0.3      # PyTO-AD's own bound
        for p in report.pieces:
            t = trimesh.load(p.path, force="mesh")
            assert t.is_watertight and t.volume > 0, p.label
        assert any(str(n).startswith("result_smooth") for n in w.plotter.actors)
        res.apart.setChecked(True)
        assert w.view == "design" and any(str(n).startswith("result_smooth") for n in w.plotter.actors)

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

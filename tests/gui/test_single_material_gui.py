"""
test_single_material_gui.py
===========================
The window with one material in the design, off-screen. Materials: Single material or Multi material, the
database that fits each (a misfit is an error in the report), Fix every element to with multi material, and
OK handing back exactly what was chosen. Settings: every setting of the latent points greyed out with one
material, the VAE settings too with single material, their values kept. Formulation: Material volume fraction
greyed out, with the reason beside a row that has it. The database editor saves a database of one material.
A run of one material: no Latent space tab, None for its latent dimension, and Compare says why it has no
latent space to compare.

    python -m pytest tests/gui/test_single_material_gui.py
"""
import contextlib
import os
import shutil
import sys
import tempfile

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform.startswith("win"):                         # off-screen Qt on Windows finds no fonts itself
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from PyQt5 import QtCore, QtWidgets  # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
QtCore.qInstallMessageHandler(lambda kind, context, message: None if "OpenGL" in message or
                              "propagateSizeHints" in message else sys.stderr.write(message + "\n"))

from gui import formulation_panel, main_window as G, materials_panel, settings_panel  # noqa: E402

DATABASE = os.path.join(ROOT, "data", "databases", "6Materials_AF_3Attr.xlsx")
ROW = 2                                                     # the material each test makes single


@contextlib.contextmanager
def scratch():
    """A folder one level below the repository, like projects/, so STL paths are written the same."""
    d = tempfile.mkdtemp(prefix="_test_gui_", dir=ROOT)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def opened(d):
    """The U-bracket (the 6-material database) open in a window, from a copy in d."""
    w = G.MMTOWindow()
    assert w.open_project(shutil.copy(os.path.join(ROOT, "projects", "U-bracket.mmto"), d))
    return w


def one_material_database(folder):
    """A database of one material, the 6-material database's row ROW, written in folder; returns its path."""
    from openpyxl import Workbook
    from lsr_mmto.materials.databases import database_table
    names, attributes, units, values = database_table(DATABASE)
    wb = Workbook()
    ws = wb.active
    ws.append(["Attributes"] + list(attributes))
    ws.append(["Material/Units"] + list(units))
    ws.append([str(names[ROW])] + [float(v) for v in values[ROW]])
    path = os.path.join(folder, "one_material.xlsx")
    wb.save(path)
    return path


def names():
    from lsr_mmto.materials.databases import database_table
    return [str(n) for n in database_table(DATABASE)[0]]


def test_materials_window_single_or_multi_material():
    """Multi material, as every project before this choice: Fix every element to, on the top row, lists None and
    the database's materials, and the material chosen is the project's pinned material after OK. Single material
    with the same 6-material database: the list becomes Material:, with every material and no None; the
    database's first is chosen unless one was chosen for single material before, the report names it, and OK
    hands back the single material and that material; no Network error tab. Back to multi material: nothing is
    fixed, and back to single material: its material again. A database of one material works for single
    material and is an error for multi material; a fixed material that a newly chosen database does not have is
    no longer fixed, and the report says so."""
    with scratch() as d:
        w = opened(d)
        m = materials_panel.MaterialsPanel(w, w.project)
        assert m.multi_button.isChecked() and not m.single_button.isChecked()
        assert m.fix_row.isVisibleTo(m) and m.tabs.isTabVisible(1) and m.fix_label.text() == "Fix every element to:"
        assert [m.fix_combo.itemData(i) for i in range(m.fix_combo.count())] == [None] + names()
        assert m.fix_combo.itemText(0) == materials_panel.NOTHING_FIXED and m.fix_combo.currentIndex() == 0
        m.fix_combo.setCurrentIndex(1 + ROW)
        label = m.fix_combo.itemText(1 + ROW)                                 # as the list shows it
        assert m.apply(w.project) == [f"Materials in the design: Multi material, every element fixed to {label}"]
        assert (w.project.material_source, w.project.pinned_material) == ("network", names()[ROW])
        assert w.project.validate() == []

        m = materials_panel.MaterialsPanel(w, w.project)
        assert m.fix_combo.currentData() == names()[ROW]                       # the fixed material, shown again
        m.single_button.setChecked(True)
        assert m.fix_label.text() == "Material:" and m.fix_row.isVisibleTo(m) and not m.tabs.isTabVisible(1)
        assert [m.fix_combo.itemData(i) for i in range(m.fix_combo.count())] == names()
        assert m.project.pinned_material == names()[0] and m.fix_combo.currentIndex() == 0
        m.fix_combo.setCurrentIndex(ROW)
        assert m.report.toPlainText().startswith(f"{label}, at its database values; no network."), \
            m.report.toPlainText()
        assert not m.latent_info.isVisibleTo(m)
        m.multi_button.setChecked(True)
        assert m.project.pinned_material == names()[ROW] and m.fix_combo.currentIndex() == 1 + ROW   # its own
        m.single_button.setChecked(True)
        assert m.project.pinned_material == names()[ROW]                       # its choice, back again
        vae = dict(w.project.settings["vae"])
        assert m.apply(w.project) == [f"Materials in the design: Single material: {label}"]
        assert w.project.settings["vae"] == vae                                # no network: its settings stay
        assert (w.project.material_source, w.project.pinned_material) == ("one_database_material", names()[ROW])
        assert w.project.validate() == [], w.project.validate()

        m = materials_panel.MaterialsPanel(w, w.project)
        assert m.single_button.isChecked() and m.fix_combo.currentData() == names()[ROW]
        m.set_database(one_material_database(d))
        assert m.project.pinned_material == names()[ROW]                       # the same material, by name
        lines = m.apply(w.project)
        assert lines[0] == "Database: one_material.xlsx" and w.project.validate() == [], (lines, w.project.validate())

        m = materials_panel.MaterialsPanel(w, w.project)
        m.multi_button.setChecked(True)
        assert "Error: Multi material: one_material.xlsx has one material, and the network needs at least two" \
            in m.report.toPlainText()
        m.project.pinned_material = "no such material"                         # a fixed material another lacks
        m.set_database(DATABASE)
        assert "no such material is not in this database, so no material is fixed now." in m.report.toPlainText()
        assert m.project.pinned_material is None and m.fix_combo.currentIndex() == 0


def test_materials_window_keeps_each_choice_and_changes_nothing_on_its_own():
    """Multi material with a fixed material, then single material, then multi again: the fixed material is back.
    A single material the newly chosen database does not have: its first material is chosen, and the report
    says so. Single material on a database of one material with no material named: opening and OK change
    nothing. Analyse starts on the project's material."""
    from gui import run_panel
    with scratch() as d:
        w = opened(d)
        m = materials_panel.MaterialsPanel(w, w.project)
        m.fix_combo.setCurrentIndex(1 + ROW)
        m.single_button.setChecked(True)
        m.fix_combo.setCurrentIndex(1)
        m.multi_button.setChecked(True)
        assert m.project.pinned_material == names()[ROW]
        m.single_button.setChecked(True)
        assert m.project.pinned_material == names()[1]
        m.set_database(one_material_database(d))
        assert m.project.pinned_material is None and m.fix_combo.currentData() == names()[ROW]   # its only one
        assert f"{names()[1]} is not in this database, so its first material is chosen." in m.report.toPlainText()

        w.project.database = one_material_database(d)
        w.project.material_source, w.project.pinned_material = "one_database_material", None
        m = materials_panel.MaterialsPanel(w, w.project)
        assert m.fix_combo.currentText() and m.project.pinned_material is None
        assert m.apply(w.project) == [] and w.project.pinned_material is None

        w.project.database = DATABASE
        w.project.pinned_material = names()[ROW]
        chosen = []
        real = run_panel.AnalyseDialog.exec_
        run_panel.AnalyseDialog.exec_ = lambda dialog: chosen.append(dialog.material.currentIndex()) or 0
        try:
            w.open_run_panel().analyse()
        finally:
            run_panel.AnalyseDialog.exec_ = real
        assert chosen == [ROW]


def test_settings_greyed_with_one_material():
    """With a fixed material, every setting that reads the latent points is greyed out with the reason, the
    others and the VAE settings are not; with single material, the VAE settings are greyed out too and the VAE
    line says there is no network. OK changes nothing."""
    from lsr_mmto.problem.settings import BY_KEY
    with scratch() as d:
        w = opened(d)
        w.project.pinned_material = names()[ROW]
        s = settings_panel.SettingsPanel(w, w.project)
        for (section, key), entry in s.rows.items():
            if section == "group":
                continue
            row, label, reset = entry
            if section == "vae":                                                # greyed by the trained network
                assert label.toolTip() != settings_panel.ONE_MATERIAL, key       # only, not for one material
                continue
            latent = BY_KEY[key].reads_latent_points
            assert row.widget.isEnabled() != latent and label.isEnabled() != latent, key
            if latent:
                assert label.toolTip() == settings_panel.ONE_MATERIAL and not reset.isVisibleTo(s), key
        assert s.apply(w.project) == []

        w.project.database = os.path.relpath(one_material_database(d), w.project.base_dir).replace("\\", "/")
        w.project.material_source, w.project.pinned_material = "one_database_material", None
        s = settings_panel.SettingsPanel(w, w.project)
        for (section, key), entry in s.rows.items():
            if section == "vae":
                row, label, reset = entry
                assert not row.widget.isEnabled() and label.toolTip() == settings_panel.ONE_MATERIAL, key
        assert s.vae_status.text() == settings_panel.NO_NETWORK and s.vae_checks.text() == ""
        assert s.rows[("common", "maxIterations")][0].widget.isEnabled()
        assert s.apply(w.project) == []


def test_formulation_greys_material_volume_fraction_with_one_material():
    """With one material, Material volume fraction is greyed out in the list, resting the mouse on it says why,
    and a row that already has it shows the reason beside it."""
    with scratch() as d:
        w = opened(d)
        w.project.constraint("MaterialVolumeFraction(1)", "<=", 0.3)
        f = formulation_panel.FormulationPanel(w, w.project)
        assert f.reason("MaterialVolumeFraction") is None
        w.project.pinned_material = names()[ROW]
        f = formulation_panel.FormulationPanel(w, w.project)
        combo = f._quantity_combo("constraint")
        item = next(combo.model().item(i) for i in range(combo.count())
                    if combo.itemData(i) == "MaterialVolumeFraction")
        assert not item.isEnabled() and item.toolTip() == "Not available: one material fills the design"
        row = next(e for e in f.rows if e["widgets"]["quantity"].currentData() == "MaterialVolumeFraction")
        assert row["widgets"]["why"].text() == "one material fills the design"


def test_database_editor_saves_one_material():
    """A database of one material is a database the editor saves, and its one row cannot be removed."""
    from gui import database_editor as DE
    with scratch() as d:
        w = opened(d)
        e = DE.DatabaseEditor(w, d, path=one_material_database(d), unit_system=w.project.units)
        assert e.check() == [] and e.save_button.isEnabled()
        e.table.setCurrentCell(1, 0)
        e.remove_material()
        assert e.table.rowCount() == 2 and e.check() == []


def test_the_live_design_of_a_run_of_one_material():
    """A run of one material started in the window: no live Latent space tab, and the live design draws with
    this run's material labels, whether it is the session's first run or follows a run of another database
    (the labels were once set only for a run of several materials)."""
    import numpy as np
    from lsr_mmto.materials.databases import material_display
    with scratch() as d:
        w = opened(d)
        drawn = []
        w.set_design = lambda make, show=False: drawn.append(make)          # the design view, not drawn here
        run_window = w.open_run_panel()
        for earlier in (None, "BM1_3Mats_2Attr"):
            if earlier:                                                       # a run of another database first
                run_window.project = w.project.copy(_database=earlier)
                run_window._start_live_latent()
            run_window.project = w.project.copy()
            run_window.project.pinned_material = names()[ROW]
            run_window.colours = run_window._material_colours()
            run_window._start_live_latent()
            tabs = [run_window.tabs.tabText(i) for i in range(run_window.tabs.count())]
            assert "Latent space" not in tabs, tabs
            assert run_window.latent_names == material_display(DATABASE)[1]
            n = int(w.project.mesh().num_elems)
            run_window.update_visualization({"x": np.ones(n), "material": np.full(n, ROW)}, 1)
            assert drawn
        run_window.parentWidget().close()
        w.close()


@pytest.fixture(scope="module")
def single_run():
    """A run of one database material, a few iterations on a coarse mesh, read back; and a run of the network."""
    from lsr_mmto.problem.project import Project, run
    from lsr_mmto.results import load_results
    d = tempfile.mkdtemp(prefix="_test_gui_", dir=ROOT)
    out = []
    for single in (True, False):
        p = Project.load(os.path.join(ROOT, "projects", "U-bracket.mmto"))
        p.n_dof, p.settings["common"]["maxIterations"] = 3000, 2
        if single:
            p.database = one_material_database(d)
            p.material_source = "one_database_material"
        out.append(load_results.load(os.path.join(run(p, save_dir=os.path.join(d, str(single)), verbose=False),
                                                  "run_data")))
    yield out
    shutil.rmtree(d, ignore_errors=True)


def test_a_run_of_one_material_has_no_latent_space(single_run):
    """Its tabs are Convergence and Summary, its latent dimension reads None (one material), its summary names the
    material every element is, the latent plot says in words that there is none, and Compare puts in the Latent
    space tab why there is none."""
    from gui import results_panel
    from gui.run_panel import result_tabs
    from lsr_mmto.results import compare, run_plots, tables
    one, several = single_run
    assert tables.one_material_text(one) == f"{names()[ROW]}, at its database values"
    assert tables.one_material_text(several) is None
    with pytest.raises(ValueError, match="one material, so it has no latent space"):
        run_plots.latent_scatter(one)
    tabs = QtWidgets.QTabWidget()
    result_tabs(tabs, one)
    assert [tabs.tabText(i) for i in range(tabs.count())] == ["Convergence", "Summary"]
    result_tabs(tabs, several)
    assert [tabs.tabText(i) for i in range(tabs.count())] == ["Convergence", "Latent space", "Summary"]
    assert tables.latent_dimension_text(one) == "None (one material)" and tables.latent_dimension_text(several) != \
        "None (one material)"
    assert "None (one material)" in str(compare.table({"one": one, "several": several}))
    window = results_panel.CompareWindow(None, {"one": one, "several": several}, lambda: None)
    latent = window.tabs.widget([window.tabs.tabText(i) for i in range(window.tabs.count())].index("Latent space"))
    assert latent.text() == results_panel.NO_LATENT_SPACE.format("one")

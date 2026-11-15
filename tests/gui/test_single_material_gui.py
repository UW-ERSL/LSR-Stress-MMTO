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
    """Multi material, as every project before this choice: Fix every element to lists None and the database's
    materials, and the material chosen is the project's pinned material after OK. Single material with that
    6-material database: an error in the report, no Fix every element to and no Network error tab. With a
    database of one material: the report says it is used at its database values, and OK hands back the single
    material and the database. Multi material with a database of one material is an error; a fixed material that
    a newly chosen database does not have is no longer fixed, and the report says so."""
    with scratch() as d:
        w = opened(d)
        m = materials_panel.MaterialsPanel(w, w.project)
        assert m.multi_button.isChecked() and not m.single_button.isChecked()
        assert m.fix_row.isVisibleTo(m) and m.tabs.isTabVisible(1)
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
        report = m.report.toPlainText()
        assert report.startswith("Error: Single material: 6Materials_AF_3Attr.xlsx has 6 materials"), report
        assert not m.fix_row.isVisibleTo(m) and not m.tabs.isTabVisible(1)
        assert m.project.pinned_material is None
        one = one_material_database(d)
        m.set_database(one)
        assert m.report.toPlainText().startswith("1 material, at its database values; no network."), \
            m.report.toPlainText()
        assert not m.latent_info.isVisibleTo(m)
        vae = dict(w.project.settings["vae"])
        lines = m.apply(w.project)
        assert lines == ["Database: one_material.xlsx", "Materials in the design: Single material"], lines
        assert w.project.settings["vae"] == vae                                # no network: its settings stay
        assert (w.project.material_source, w.project.pinned_material) == ("one_database_material", None)
        assert w.project.validate() == [], w.project.validate()

        m = materials_panel.MaterialsPanel(w, w.project)
        assert m.single_button.isChecked()
        m.multi_button.setChecked(True)
        assert "Error: Multi material: one_material.xlsx has one material, and the network needs at least two" \
            in m.report.toPlainText()
        m = materials_panel.MaterialsPanel(w, w.project)                       # a fixed material another lacks
        m.multi_button.setChecked(True)
        m.project.pinned_material = "no such material"
        m.set_database(DATABASE)
        assert "no such material is not in this database, so no material is fixed now." in m.report.toPlainText()
        assert m.project.pinned_material is None and m.fix_combo.currentIndex() == 0


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

"""
test_thermal_walk_through.py
============================
A thermal project, the physics the window has never shown, taken through the window the way a user
would: saved and opened again, Run, its Results views and Summary, Analyse, Analyse this design,
Compare, Export STL, and a new database in the editor. A physics that is not structural must go through
every step with nothing of structural's showing: no stress, no failure factor, no force or stress units.
A run and an analysis whose solver did not converge say so in red under Current run (the lines themselves
stood in for here). About ten seconds.

    python tests/gui/test_thermal_walk_through.py        (or: python -m pytest tests)
"""
import os
import shutil
import sys
import tempfile

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform.startswith("win"):                         # off-screen Qt on Windows finds no fonts itself
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests", "physics"))           # its helpers, from test_thermal_physics.py

from PyQt5 import QtCore, QtWidgets  # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _quiet(kind, context, message):
    """Qt's messages, less the ones the off-screen mode prints for every 3D view it cannot draw."""
    if not any(s in message for s in ("OpenGL", "propagateSizeHints", "makeCurrent")):
        sys.stderr.write(message + "\n")


QtCore.qInstallMessageHandler(_quiet)

from test_thermal_physics import _held_and_heated  # noqa: E402

Box = QtWidgets.QMessageBox


def _wait(condition, seconds=600):
    import time
    end = time.time() + seconds
    while not condition() and time.time() < end:
        APP.processEvents()
        time.sleep(0.05)
    APP.processEvents()
    return condition()


def _summary_text(tabs):
    """The text of the Summary tab of a Results or Compare window."""
    for i in range(tabs.count()):
        if tabs.tabText(i) == "Summary":
            return tabs.widget(i).findChild(QtWidgets.QTextBrowser).toPlainText()
    raise AssertionError("no Summary tab")


def test_a_thermal_project_through_the_window(monkeypatch):
    from gui import database_editor, formulation_panel, main_window as G, materials_panel, run_panel, settings_panel
    from gui import supports_and_loads_panel as SP
    from gui.results_panel import SMOOTH
    from lsr_mmto.problem.project import Project
    from lsr_mmto.results.load_results import load

    def analyse_with(dialog):                              # the Analyse window: the first material, saved
        dialog.material.setCurrentIndex(0)
        dialog.save.setChecked(True)
        return QtWidgets.QDialog.Accepted
    monkeypatch.setattr(run_panel.AnalyseDialog, "exec_", analyse_with)
    from lsr_mmto.results import tables                    # the red line at the end of a run and an analysis,
    said = lambda run: [f"Not reliable: a test line for {run.analysed_material or 'the run'}."]   # noqa: E731
    monkeypatch.setattr(tables, "solver_warning", said)    # as if their solver had not converged
    for name in ("question", "warning", "information"):
        monkeypatch.setattr(Box, name, staticmethod(lambda *a, _n=name, **k: Box.Yes if _n == "question" else Box.Ok))

    d = tempfile.mkdtemp(prefix="_test_thermal_", dir=ROOT)
    try:
        p = _held_and_heated("Heated bracket")
        p.units = "mmKS"
        p.settings["common"]["maxIterations"] = 3
        path = os.path.join(d, "T", "Heated bracket.mmto")
        p.save(path)                                                 # saved: its part and database copied in
        q = Project.load(path)
        assert q.physics == "thermal" and q.bcs == p.bcs and q.units == "mmKS"
        assert q.validate() == []

        w = G.MMTOWindow()
        w.show()
        assert w.open_project(path) and w.project.physics == "thermal"

        assert w.open_supports_and_loads_panel() is not None          # its own kinds, in its own units
        for bc in w.project.bcs:
            assert SP.BCDialog(w, w.project, bc).bc() == bc, bc
        b = SP.BCDialog(w, w.project)
        assert [b.kind.itemText(i) for i in range(b.kind.count())] == ["Fixed temperature", "Heat source"]
        b.kind.setCurrentText("Heat source")
        b.inputs["total"][1].set_value(2.5)
        assert b.problems() == [] and b.bc()["total"] == 2.5
        assert SP.field_unit("heat", "mmKS") == "mW" and SP.field_unit("temperature", "IPS") == "°R"
        assert "Total heat" in SP.bc_values(w.project.bcs[1]), SP.bc_values(w.project.bcs[1])
        b.close()
        for cls in (materials_panel.MaterialsPanel, formulation_panel.FormulationPanel,
                    settings_panel.SettingsPanel):                    # each panel: OK with no edits
            assert cls(w, w.project).apply(w.project) == [], cls.__name__
        assert not w.is_dirty()

        r = w.open_run_panel()                                        # Run
        assert r.start_optimization()
        assert _wait(lambda: not r.optimization_running), "the run did not finish"
        assert r.status.text().startswith("Run finished after 3 iterations"), r.status.text()
        assert "<span style='color:#b00000'>Not reliable: a test line for the run.</span>" in r.status.text()
        run_folder = r.save_dir
        ran = load(run_folder)
        assert ran.physics == "thermal" and list(ran.results) == ["temperature"]

        res = w.open_results_panel(run_folder)                        # Results: its views
        shown = [res.field.itemText(i) for i in range(res.field.count())]
        assert shown == ["Material", "Density", "Temperature", "Solid", SMOOTH], shown
        for view in shown[:-1]:
            res.field.setCurrentText(view)
            assert "result_design" in w.plotter.actors and w.view == "design", view
            assert not res.component.isVisible() and not res.deformed.isVisible(), view
        res.field.setCurrentText("Temperature")                       # at the nodes: smooth over the faces
        drawn = w.plotter.actors["result_design"].GetMapper().GetInput()
        assert np.isclose(np.asarray(drawn.point_data["value"]).max(), ran.results["temperature"].max())
        assert any("Temperature" in t and "K" in t for t in w.design_bars), w.design_bars

        summary = _summary_text(res.tabs)                             # Summary: its own row and units
        assert "Max temperature" in summary and "mmKS: mm, K, mW, mW/(mm*K), t" in summary, summary
        assert "failure" not in summary.lower() and "MPa" not in summary

        assert w.analyse()                                            # Analyse: the solid part, saved
        assert _wait(lambda: not w.run_going_now()), "the analysis did not finish"
        assert r.status.text().startswith("Analysis of the solid part in"), r.status.text()
        assert "<span style='color:#b00000'>Not reliable: a test line for " in r.status.text()
        assert "a test line for the run" in w.message_text.toPlainText()          # and in the messages panel
        solid = r.save_dir
        assert load(solid).results["temperature"].shape == ran.results["temperature"].shape

        res = w.open_results_panel(run_folder)                        # Analyse this design
        assert res.select(run_folder) and res.analyse_design()
        assert _wait(lambda: not w.run_going_now()), "the design analysis did not finish"
        design = w.run_window.save_dir
        assert load(design).config["analysis"]["design_of"] == os.path.basename(run_folder)

        res = w.results_window                                        # Compare: the run and its analysis
        assert res.compare(), res.status.text()
        cw = res.compare_window
        text = _summary_text(cw.tabs)
        assert "Thermal compliance (mW*K)" in text and "failure" not in text.lower(), text
        cw.close()

        assert res.select(run_folder)                                 # Export STL
        w.update_top_row()
        assert w.export_stl()
        assert _wait(lambda: not w.exporting, 300)
        folder, report = w.exported_last
        assert folder == run_folder and os.path.isfile(os.path.join(folder, "design.stl"))

        e = database_editor.DatabaseEditor(w, d, unit_system=w.project.units, project=w.project)
        assert e.attributes == ["Density", "Conductivity"]            # a new database: what thermal reads
        assert [e.table.horizontalHeaderItem(c).text() for c in range(e.table.columnCount())][-2:] == \
            ["Density", "Thermal conductivity"]
        e.close()

        assert w.save_project()                                       # saved and opened again
        assert w.open_project(path) and w.project.physics == "thermal" and w.project.bcs == p.bcs
        w.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)



def test_temperatures_and_heat_are_picked_on_the_part():
    """Click on the part, on a thermal project, opens the Pick window of temperatures and heat: Type offers
    what thermal's kinds take (Temperature, Total Heat; not Heat Flux, which no kind takes), each value in the
    project's units. A click on the part selects its face, a right click clears it, and Apply adds the pick to
    the project at once, saying so in the project's units and leaving no label behind. A thermoelastic project
    offers both picking windows, one line each, each opening its own; a project switch or a change of units
    closes them."""
    from types import SimpleNamespace
    from gui import main_window as G
    d = tempfile.mkdtemp(prefix="_test_thermal_", dir=ROOT)
    try:
        p = _held_and_heated("Picked bracket")
        p.units = "mmKS"
        path = os.path.join(d, "T", "Picked bracket.mmto")
        p.save(path)
        w = G.MMTOWindow()
        w.show()
        assert w.open_project(path)
        panel = w.open_supports_and_loads_panel()
        menu = [b for b in panel.findChildren(QtWidgets.QPushButton) if b.text() == "Add"][0].menu()
        panel.fill_add_menu(menu)
        assert [a.text() for a in menu.actions()] == ["Click on the part...", "Enter by hand..."]
        assert w.pick_windows() == [G.ThermalPickWindow]
        panel.pick()
        pick = w.thermal_loads_window
        assert isinstance(pick, G.ThermalPickWindow) and pick.windowTitle() == "Pick"
        assert w.picking_window() is pick
        assert [pick.thermal_type.itemText(i) for i in range(pick.thermal_type.count())] == ["Temperature",
                                                                                           "Total Heat"]
        labels = [label.text() for label in pick.findChildren(QtWidgets.QLabel)]
        assert "Temperature (K)" in labels and "Total Heat (mW)" in labels, labels

        def click(cell):                                                    # as the mouse on the part does
            w.on_left_button_press(None, SimpleNamespace(GetCellId=lambda: cell, GetActor=lambda: w.part_actor()))

        def selected():
            return sum(bool(t) for t in w.stl_geom.tri_highlight)

        click(10)
        assert selected() > 0
        w.on_right_button_press(None, None)                                 # a right click clears it
        assert selected() == 0
        before = len(w.project.bcs)
        for kind, value, spin in (("Temperature", 310.0, "temp_value_spin"), ("Total Heat", 4.0,
                                                                              "total_heat_value_spin")):
            if not in_open_strip(w, "thermal_loads_window"):
                panel.pick()
            pick = w.thermal_loads_window
            click(10)
            pick.thermal_type.setCurrentText(kind)
            getattr(pick, spin).setValue(value)
            pick.on_apply_clicked()
            for _ in range(5):
                APP.processEvents()
        assert [(bc["kind"], bc.get("value", bc.get("total"))) for bc in w.project.bcs[before:]] == \
            [("fixed_temperature", 310.0), ("heat_source", 4.0)], w.project.bcs[before:]
        text = w.message_text.toPlainText()
        assert "(4 mW)" in text and " W to " not in text, text[-300:]
        assert not [n for n in w.plotter.actors if "_text_" in str(n)]       # no label left behind as a dot

        w.project.physics = "thermoelastic"                                   # both windows, one line each
        panel.fill_add_menu(menu)
        sub = menu.actions()[0].menu()
        assert menu.actions()[0].text() == "Click on the part" and sub is not None
        assert [a.text() for a in sub.actions()] == ["Supports and forces...", "Temperatures and heat..."]
        for action, attr in zip(sub.actions(), ("structural_loads_window", "thermal_loads_window")):
            action.trigger()
            assert in_open_strip(w, attr), attr
        w.close_picking_windows()
        panel.pick()
        assert w.save_project()                                               # so the switch asks nothing
        w.open_project(path)                                                  # a project switch closes it
        assert not in_open_strip(w, "thermal_loads_window")
        w.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def in_open_strip(w, attr):
    """Whether the window keeps an open picking window under attr."""
    from gui.main_window import in_open_strip as open_
    return open_(getattr(w, attr, None))

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))

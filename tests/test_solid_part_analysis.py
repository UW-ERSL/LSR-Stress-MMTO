"""
test_solid_part_analysis.py
===========================
Analyse (optimize/solid_part_analysis.py, the button beside Run): the solid part solved once in one
material of the database, with that material's database values.

- Its compliance is a direct solve's, with every element solid and the material's values, to 1e-10.
- A project that cannot run cannot be analysed either: the same checks refuse it.
- Not saved, nothing is written; saved, it is one new results entry, marked as an analysis, that
  loads back like a run.
- In the window: Analyse beside Run, its window, Results showing it with its label and no convergence
  or latent space, and a comparison of a run with an analysis that says so instead of plotting.

The thermal physics is analysed in tests/test_thermal_physics.py.

    python -m pytest tests/test_solid_part_analysis.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pytest
import torch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform.startswith("win"):                         # off-screen Qt on Windows finds no fonts itself
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lsr_mmto.optimize import solid_part_analysis as A  # noqa: E402
from lsr_mmto.problem.project import Project  # noqa: E402

PROJECTS = os.path.join(ROOT, "projects")


def _beam():
    """The rectangular beam on a coarse mesh, with compliance as its objective."""
    p = Project.load(os.path.join(PROJECTS, "rectangular beam.mmto"))
    p.n_dof = 3000
    p.formulation["objective"] = {"expression": "Compliance()", "sense": "minimize"}
    return p


def _values(monkeypatch):
    """The objective and constraint values the analysis computes, at full precision, as it computes them."""
    seen = []
    real = A._quantities

    def keep(*args):
        seen.append(real(*args))
        return seen[-1]
    monkeypatch.setattr(A, "_quantities", keep)
    return seen


def _files(folder):
    """{path: bytes} of every file under folder."""
    out = {}
    for base, _, names in os.walk(folder):
        for n in names:
            with open(os.path.join(base, n), "rb") as fh:
                out[os.path.relpath(os.path.join(base, n), folder)] = fh.read()
    return out


def test_compliance_is_a_direct_solve_of_the_solid_part(monkeypatch):
    """The analysis in each of two materials gives the compliance F . u of a solve with x_phys = 1 in every
    element and that material's Young's modulus and Poisson's ratio from the database, to 1e-10, and that
    solve's displacement u at each node, x, y and z, as its result."""
    from lsr_mmto.materials.databases import database_table
    from lsr_mmto.physics.available_physics import physics_module
    from lsr_mmto.pyto_ad_imports import MaterialModel, deflation, linear_solvers
    seen = _values(monkeypatch)
    p = _beam()
    p.complete_settings()
    names, attributes, _, values = database_table(p.database_paths()[0])
    mesh, mat_prop, bcs, _, _, _ = p.build(verbose=False)
    physics = physics_module(p.physics)
    fe = physics.make_solver(mesh, mat_prop, bcs, linear_solvers.Solvers.PARDISO, deflation.DeflationSolver(), None)
    advanced = p.settings["advanced"]
    n = mesh.num_elems
    for row in (0, len(names) - 1):
        E = values[row, attributes.index("Youngs_Modulus")]
        constant = advanced["use_constant_poissons_ratio"] or "Poissons_Ratio" not in attributes
        nu = advanced["constant_poissons_ratio"] if constant else values[row, attributes.index("Poissons_Ratio")]
        fe.set_material_fields(torch.full((n,), float(E), dtype=torch.float64),
                               torch.full((n,), float(nu), dtype=torch.float64))
        u = fe.solve(torch.ones(n, dtype=torch.float64), MaterialModel.SIMP)
        direct = float(torch.dot(fe.total_force, u))
        run = A.analyse_solid_part(p, row)
        name, value, limit = seen[-1][0]
        assert name == "Compliance" and limit is None
        assert abs(value - direct) <= 1e-10 * abs(direct), (row, value, direct)
        assert np.isclose(run.final_objective, direct, rtol=1e-6)            # the record keeps it as float32
        assert run.analysed_material is not None and run.n_latent == 0
        assert np.all(run.material_indices == row) and run.solid.all() and np.all(run.x_cont == 1)
        assert list(run.results) == ["von_mises", "displacement"] and run.results["von_mises"].shape == (n,)
        shown, u = run.results["displacement"], u.numpy().reshape(-1, 3)
        assert shown.shape == (mesh.num_nodes, 3) and run.result_info["displacement"]["where"] == "node"
        assert np.allclose(shown, u, rtol=0, atol=1e-6 * np.abs(u).max())         # kept as float32
        assert np.allclose(np.linalg.norm(shown, axis=1), np.linalg.norm(u, axis=1), rtol=0,
                           atol=1e-6 * np.abs(u).max())
    E0, E1 = (values[r, attributes.index("Youngs_Modulus")] for r in (0, len(names) - 1))
    assert E0 != E1, "the two materials should differ"


def test_a_project_that_cannot_run_is_not_analysed():
    p = _beam()
    p.formulation["objective"] = None
    with pytest.raises(ValueError, match="no objective"):
        A.analyse_solid_part(p, 0)
    with pytest.raises(ValueError, match="a row of the database"):
        A.analyse_solid_part(_beam(), 99)


def test_not_saved_writes_nothing_saved_is_one_entry_that_loads_back():
    from lsr_mmto.problem import project_folder
    from lsr_mmto.results import load_results
    with tempfile.TemporaryDirectory(dir=ROOT) as d:
        p = _beam()
        p.save(os.path.join(d, "beam.mmto"))
        p = Project.load(os.path.join(d, "beam.mmto"))
        before = _files(d)
        run = A.analyse_solid_part(p, 1)
        assert _files(d) == before, "an analysis not saved wrote into the project's folder"
        assert run.path == "" and run.analysed_material

        folder = project_folder.new_run_folder(p)
        saved = A.analyse_solid_part(p, 1, save_dir=folder)
        assert project_folder.runs(p) == [folder]
        assert sorted(os.listdir(folder)) == ["run_config.yaml", "run_data.config.json", "run_data.npz"]
        assert project_folder.analysed_material(folder) == saved.analysed_material == run.analysed_material
        again = load_results.load(os.path.join(folder, "run_data"))
        assert again.analysed_material == run.analysed_material
        assert np.array_equal(again.history["J_phys"], saved.history["J_phys"])     # the saved analysis comes
        assert np.array_equal(again.results["von_mises"], saved.results["von_mises"])   # back as it was saved
        assert np.array_equal(again.results["displacement"], saved.results["displacement"])   # a vector at each
        assert again.results["displacement"].shape == (int(p.mesh().num_nodes), 3)          # node, back as it was
        assert again.result_info == run.result_info and again.result_info["displacement"]["shape"] == "vector"
        assert np.array_equal(again.material_indices, run.material_indices)
        assert again.config["analysis"] == run.config["analysis"]
        assert project_folder.run_facts(folder)[1] is None                   # no iterations
        from lsr_mmto.results import tables
        text = str(tables.summary(again))
        assert "Analysis: " + again.analysed_material in text and "Iterations" not in text
        assert "Grey" not in text and "Latent dimension" not in text

        record = dict(np.load(os.path.join(folder, "run_data.npz")))              # as an older run wrote it:
        meta = json.loads(str(record["meta_json"]))                                 # no displacement, results
        meta["results"] = {"von_mises": {k: meta["results"]["von_mises"][k] for k in ("view", "title", "measures")}}
        record["meta_json"] = np.array(json.dumps(meta))                            # described without where
        del record["r_displacement"]                                                # and shape
        np.savez_compressed(os.path.join(folder, "run_data.npz"), **record)
        old = load_results.load(folder)
        assert list(old.results) == ["von_mises"] and old.results["von_mises"].shape == (old.n_elems,)
        assert old.result_info["von_mises"] == dict(meta["results"]["von_mises"], where="element", shape="scalar")


def test_stop_ends_it_and_nothing_is_written():
    with tempfile.TemporaryDirectory() as d:
        folder = os.path.join(d, "analysis")
        with pytest.raises(A.AnalysisStopped):
            A.analyse_solid_part(_beam(), 0, save_dir=folder, should_stop=lambda: True)
        assert os.listdir(d) == []


def test_a_quantity_of_the_latent_points_is_not_given(monkeypatch):
    """The L-bracket's constraints are material volume fractions, worked out from the latent points: the
    analysis gives the objective; its record holds no value for the constraints, and its Summary and Compare
    leave them out. Compared with a run that gives them, only the rows both have are shown: the constraints and
    what only an optimization has (grey elements, latent dimension, iterations) are left out."""
    import dataclasses
    from lsr_mmto.results import compare
    from lsr_mmto.results import tables
    seen = _values(monkeypatch)
    p = Project.load(os.path.join(PROJECTS, "L-bracket.mmto"))
    p.n_dof = 3000
    run = A.analyse_solid_part(p, 0)
    assert seen[-1][0][1] is not None and all(v is None for _, v, _ in seen[-1][1:])
    assert np.isfinite(run.final_objective) and np.all(np.isnan(run.history["constraints"]))
    text = str(tables.summary(run))
    assert run.objective_label in text and "volume fraction" not in text and "Constraint" not in text, text
    assert "volume fraction" not in tables.summary_details(run)
    assert "volume fraction" not in str(compare.table({"one": run, "two": run}))
    optimized = dataclasses.replace(run, config={k: v for k, v in run.config.items() if k != "analysis"},
                                    history=dict(run.history, constraints=[[-0.5, 0.0, 0.5]]))
    both = compare.table({"run": optimized, "analysis": run})
    heads = [str(r[0]) for t in both.tables[1:] for r in t.rows]
    assert not [h for h in heads if "used" in h or h in ("Grey (in between)", "Latent dimension", "Iterations",
                                                         "Time per iteration")], heads
    assert heads[0].startswith(run.objective_label) and "Solid" in heads and "Used" in heads and "Run time" in heads
    rows = [r for r in compare.table({"run": optimized, "again": optimized}).tables[1].rows if "used" in str(r[0])]
    assert len(rows) == 3 and all(isinstance(r[1], tables.Raw) for r in rows), rows


# ---------------------------------------------------------------------- in the window
def test_analyse_in_the_window(monkeypatch):
    """Analyse beside Run, with the same look; its window; not saved: shown first in Results, nothing
    written; saved: listed among the runs with its label; a run and an analysis compared: no convergence
    or latent space to plot, said in their place. Analyse this design in Results, for a run only: the
    analysis of its design is saved, shown, and ticked with its run, ready for Compare."""
    from PyQt5 import QtCore, QtWidgets
    APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])   # noqa: F841
    from gui import main_window as G
    from gui import results_panel, run_panel
    from lsr_mmto.problem import project_folder

    def wait(condition, seconds=600):
        import time
        end = time.time() + seconds
        while not condition() and time.time() < end:
            APP.processEvents()
            time.sleep(0.05)
        APP.processEvents()
        return condition()

    chosen = {}

    def exec_(dialog):                                     # the Analyse window, answered
        chosen["materials"] = [dialog.material.itemText(i) for i in range(dialog.material.count())]
        chosen["save ticked"] = dialog.save.isChecked()
        dialog.material.setCurrentIndex(chosen["row"])
        dialog.save.setChecked(chosen["save"])
        return QtWidgets.QDialog.Accepted
    monkeypatch.setattr(run_panel.AnalyseDialog, "exec_", exec_)

    d = tempfile.mkdtemp(prefix="_test_analyse_", dir=ROOT)
    try:
        w = G.MMTOWindow()
        w.show()
        assert w.open_project(shutil.copy(os.path.join(PROJECTS, "L-bracket.mmto"), d))
        w.project.n_dof, w.project.single_layer = 3000, True
        w.project.settings["common"]["maxIterations"] = 2
        assert w.save_project()
        w.update_top_row()
        assert w.analyse_button.text() == "Analyse" and w.analyse_button.isEnabled()
        assert w.analyse_button.objectName() == "analyseButton"
        row = w.run_button.parentWidget().layout()
        assert row.indexOf(w.analyse_button) == row.indexOf(w.run_button) + 1   # beside Run

        before = _files(d)
        chosen.update(row=1, save=False)
        assert w.analyse()
        assert w.run_going_now() and w.run_window.status.text().startswith("Analysing the solid part in")
        w.update_top_row()
        assert w.run_button.text() == "Stop" and not w.analyse_button.isVisible()   # the same Stop as a run
        assert w.progress_text.text().startswith("Analysing the solid part in")
        assert wait(lambda: not w.run_going_now()), "the analysis did not finish"
        w.update_top_row()
        assert w.run_button.text() == "Run" and w.analyse_button.isVisible()
        labels = chosen["materials"]
        assert len(labels) == 3 and not chosen["save ticked"]
        r, res = w.run_window, w.results_window
        assert r.status.text() == f"Analysis of the solid part in {labels[1]} done, not saved.", r.status.text()
        assert _files(d) == before, "an analysis not saved wrote into the project's folder"
        assert res.folders() == [results_panel.UNSAVED] and res.current == results_panel.UNSAVED
        assert res.list.item(0).text() == f"Analysis: {labels[1]} \u00b7 not saved"
        assert [res.tabs.tabText(i) for i in range(res.tabs.count())] == ["Summary"]
        assert not res.delete_button.isEnabled() and not res.folder_button.isEnabled()
        assert "result_design" in w.plotter.actors and w.view == "design"
        assert [res.field.itemText(i) for i in range(res.field.count())][:5] == ["Material", "Density", "Stress",
                                                                                  "Displacement", "Solid"]
        res.field.setCurrentText("Displacement")                              # an analysis's displacement too
        assert "result_design" in w.plotter.actors and res.component.isVisible() and res.deformed.isVisible()
        res.deformed.setChecked(True)
        assert "result_design" in w.plotter.actors and float(res.scale.text()) > 0
        res.deformed.setChecked(False)
        res.field.setCurrentText("Material")

        chosen.update(row=0, save=True)                                       # Stop: nothing is saved
        assert w.analyse()
        r.stop_optimization()
        assert wait(lambda: not w.run_going_now())
        assert r.status.text() == "The analysis was stopped. Nothing was saved.", r.status.text()
        assert project_folder.runs(w.project) == []

        assert r.start_optimization() and wait(lambda: not r.optimization_running), "the run did not finish"
        run_folder = r.save_dir
        chosen.update(row=2, save=True)
        assert w.analyse() and wait(lambda: not w.run_going_now())
        folders = project_folder.runs(w.project)
        assert len(folders) == 2 and run_folder in folders
        analysis = next(f for f in folders if f != run_folder)
        assert project_folder.analysed_material(analysis) == labels[2]
        assert results_panel.UNSAVED not in res.folders() and res.current == analysis
        texts = {res.list.item(i).data(QtCore.Qt.UserRole): res.list.item(i).text() for i in range(res.list.count())}
        assert texts[analysis].startswith(f"Analysis: {labels[2]} \u00b7 ")
        assert not texts[run_folder].startswith("Analysis")
        assert [res.tabs.tabText(i) for i in range(res.tabs.count())] == ["Summary"]

        for i in range(res.list.count()):
            res.list.item(i).setCheckState(QtCore.Qt.Checked)
        assert res.compare()
        cw = res.compare_window
        assert [cw.tabs.tabText(i) for i in range(cw.tabs.count())] == ["Convergence", "Latent space", "Summary"]
        for i in (0, 1):
            assert isinstance(cw.tabs.widget(i), QtWidgets.QLabel) and "not an optimization" in cw.tabs.widget(i).text()
        cw.close()

        for i in range(res.list.count()):                                    # Analyse this design: for a run
            res.list.item(i).setCheckState(QtCore.Qt.Unchecked)
        assert res.select(analysis) and not res.design_button.isEnabled()
        assert res.select(run_folder) and res.design_button.isEnabled()
        res.design_button.click()
        assert w.run_going_now() and r.status.text().startswith("Analysing the design of ")
        assert wait(lambda: not w.run_going_now()), "the analysis of the design did not finish"
        assert r.status.text().startswith("Analysis of the design of ") and "saved in the results" in r.status.text()
        design = next(f for f in project_folder.runs(w.project) if f not in folders)
        assert project_folder.analysed_material(design).startswith("design of ") and res.current == design
        assert sorted(res.ticked()) == sorted([design, run_folder]) and res.compare_button.isEnabled()
        assert "press Compare" in res.status.text()
        assert res.compare()
        res.compare_window.close()

        w.project.formulation["objective"] = None                            # a project that cannot run
        assert not w.analyse() and "Cannot start" in r.status.text()
        empty = G.MMTOWindow()                                                # no project: greyed
        empty.update_top_row()
        assert not empty.analyse_button.isEnabled()
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def _short_run(d):
    """The L-bracket run briefly on a coarse mesh in a project folder d, its database a copy of its own (so the
    test may change it), snapped to the real materials at the end. (project, run folder)."""
    from lsr_mmto.materials.databases import DATABASE_DIR
    from lsr_mmto.problem.project import run
    p = Project.load(os.path.join(PROJECTS, "L-bracket.mmto"))
    p.stl, p.base_dir = p.stl_path(), d
    os.makedirs(os.path.join(d, "materials"))
    for ending in (".xlsx", ".nt"):
        shutil.copy(os.path.join(DATABASE_DIR, "BM1_3Mats_2Attr" + ending), os.path.join(d, "materials"))
    p.database = "materials/BM1_3Mats_2Attr.xlsx"
    p.n_dof, p.single_layer = 3000, True
    p.settings["common"]["maxIterations"] = 4
    p.settings["advanced"]["snap_to_real_material"] = True
    folder = os.path.join(d, "results", "run")
    run(p, save_dir=folder, verbose=False)
    return p, folder


def test_analyse_this_design():
    """Analyse this design: a finished run's design as built (density above 0.5 solid, each element in its
    nearest database material), solved once under the project's supports and loads now. With the design the
    final check analysed (its solid elements) and the materials it used (the network's values of the real
    materials, written into the run's database), it gives the run's own final check: the stress and the
    displacement of every element and node, to 1 part in 10,000 of the largest. Twice the load gives twice the displacement; a project whose mesh
    is not the run's is refused, saying what differs; saved, its record says which run it analysed."""
    import openpyxl
    from lsr_mmto.results.load_results import load
    with tempfile.TemporaryDirectory() as d:
        p, folder = _short_run(d)
        ran = load(folder)
        npz = os.path.join(folder, "run_data.npz")                       # the final check's design: its solid set
        arrays = dict(np.load(npz, allow_pickle=True))
        arrays["x_cont"] = ran.solid.astype(arrays["x_cont"].dtype)
        np.savez_compressed(npz, **arrays)
        book = openpyxl.load_workbook(os.path.join(d, "materials", "BM1_3Mats_2Attr.xlsx"))
        sheet = book.worksheets[0]                                       # the final check's materials
        heads = [sheet.cell(row=1, column=c).value for c in range(2, sheet.max_column + 1)]
        for c, head in enumerate(heads, start=2):
            if head is None:
                continue
            for r, value in enumerate(ran.material_property(head), start=3):
                sheet.cell(row=r, column=c, value=float(value))
        book.save(os.path.join(d, "materials", "BM1_3Mats_2Attr.xlsx"))

        a = A.analyse_design(p, folder, save_dir=os.path.join(d, "results", "design check"))
        assert a.analysed_material == f"design of {ran.config['run']['timestamp'].replace('T', ' ')}"
        assert a.config["analysis"]["design_of"] == "run"
        assert np.array_equal(a.solid, ran.solid) and np.array_equal(a.material_indices, ran.material_indices)
        for name in ("von_mises", "displacement"):                       # to 1 part in 10,000 of the largest
            got, want = a.results[name], ran.results[name]               # value: the run decodes the materials
            assert np.abs(got - want).max() <= 1e-4 * np.abs(want).max(), name     # again, in its own precision

        twice = p.copy()                                                  # the loads changed: the results follow
        for bc in twice.bcs:
            if bc["kind"] == "force":
                bc["total"] = [2 * v for v in bc["total"]]
        b = A.analyse_design(twice, folder)
        assert np.allclose(b.results["displacement"], 2 * a.results["displacement"], rtol=1e-6,
                           atol=1e-9 * np.abs(a.results["displacement"]).max())

        other = p.copy()
        other.n_dof = 4000
        with pytest.raises(ValueError, match=r"another mesh than the project's now \(mesh dofs changed since\)"):
            A.analyse_design(other, folder)
        with pytest.raises(ValueError, match="this is an analysis, not a run"):
            A.analyse_design(p, os.path.join(d, "results", "design check"))

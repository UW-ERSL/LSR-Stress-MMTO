"""
test_template_physics.py
========================
The template physics (lsr_mmto/physics/template/) keeps working: it is registered as its line in
available_physics.MODULES would register it, its solver gives the right answer on a bar, and a short
U-bracket run goes through with it, with no core file changed. Copy this file for a physics of your own.

    python tests/test_template_physics.py        (or: python -m pytest tests)
"""
import contextlib
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lsr_mmto.physics import available_physics  # noqa: E402
from lsr_mmto.physics.template import template_module as TEMPLATE  # noqa: E402

UBRACKET = os.path.join(ROOT, "data", "models", "UBracket")
BAR = os.path.join(ROOT, "data", "models", "RectangularBeam")


@contextlib.contextmanager
def registered(module):
    """The core's lists with module added, as its line in available_physics.MODULES adds it when they are
    made, and taken out again afterwards."""
    from lsr_mmto.design import latent_penalties
    from lsr_mmto.formulation import quantities
    from lsr_mmto.problem import boundary_conditions, settings
    available_physics._checked(module)
    added_q = [quantities.Quantity(physics=[module.NAME], **q) for q in module.QUANTITIES]
    added_p = [latent_penalties.Penalty(physics=[module.NAME], **p) for p in module.PENALTIES]
    added_s = [settings.Setting(physics=module.NAME, **a) for a in module.SETTINGS]
    available_physics.MODULES.append(module)
    available_physics.PHYSICS[module.NAME] = module
    quantities.QUANTITIES.extend(added_q)
    latent_penalties.PENALTIES.extend(added_p)
    boundary_conditions.BC_KINDS.update(boundary_conditions.kinds_of(module))
    settings.SETTINGS.extend(added_s)
    settings._make_views()
    try:
        yield
    finally:
        available_physics.MODULES.remove(module)
        del available_physics.PHYSICS[module.NAME]
        for q in added_q:
            quantities.QUANTITIES.remove(q)
        for p in added_p:
            latent_penalties.PENALTIES.remove(p)
        for k in module.SUPPORT_AND_LOAD_KINDS:
            del boundary_conditions.BC_KINDS[k]
        for s in added_s:
            settings.SETTINGS.remove(s)
        settings._make_views()


def _project(name, stl, base_dir, n_dof):
    from lsr_mmto.problem.project import Project
    p = Project(name, stl=stl, n_dof=n_dof, single_layer=True, base_dir=base_dir)
    p.physics = "template"
    p.database = "6Materials_AF_3Attr"
    p.objective("DiffusionCompliance()")
    p.constraint("VolumeFraction()", "<=", 0.5)
    return p


def test_a_project_sees_only_its_own_physics_settings():
    """The Settings panel shows the settings every project has and those of the project's own physics, each
    group once; the build sets only that physics' settings that PyTO-AD keeps elsewhere."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from gui import settings_panel
    with registered(TEMPLATE):
        structural, template = (set(settings_panel.all_setting_keys(name)) for name in ("structural", "template"))
        assert ("common", "pnorm_exponent") in structural and ("common", "pnorm_exponent") not in template
        assert ("advanced", "template_coefficient_scale") in template
        assert ("advanced", "template_coefficient_scale") not in structural
        assert ("common", "maxIterations") in structural & template                 # every project has it
        for name in ("structural", "template", None):
            groups = [(tab, group) for tab, group, _ in settings_panel.setting_groups(name)]
            assert len(groups) == len(set(groups)), groups                        # each group once


def test_template_solver_gives_a_straight_line_along_a_bar():
    """c held at 0 at one end of a bar and 1 at the other, solid everywhere, one coefficient: c rises in a
    straight line from end to end, at every node. Trilinear elements give this exactly."""
    from lsr_mmto.problem import selections as S
    from lsr_mmto.pyto_ad_imports import linear_solvers
    with registered(TEMPLATE):
        p = _project("bar", "RectangularBeam.STL", BAR, 2000)
        p.add_selection("left", S.face("xmin"))
        p.add_selection("right", S.face("xmax"))
        p.bcs += [{"kind": "set_value", "on": "left", "value": 0.0},
                  {"kind": "set_value", "on": "right", "value": 1.0}]    # held values only: no load needed
        p.complete_settings()
        assert p.validate() == []
        mesh, mat_prop, bcs, _, _, _ = p.build(verbose=False)
        fe = TEMPLATE.make_solver(mesh, mat_prop, bcs, linear_solvers.Solvers.PARDISO, None, None)
        fe.set_coefficient(torch.full((mesh.num_elems,), 3.0, dtype=torch.float64))
        c = fe.solve(torch.ones(mesh.num_elems, dtype=torch.float64), None).detach().numpy()
        x = np.asarray(mesh.node_xyz, dtype=float)[:, 0]
        expected = (x - x.min()) / (x.max() - x.min())
        assert np.max(np.abs(c - expected)) < 1e-8


def test_template_physics_runs_with_no_core_file_changed():
    from lsr_mmto.problem import selections as S
    from lsr_mmto.problem.project import run
    from lsr_mmto.results import load_results
    with registered(TEMPLATE):
        p = _project("template", "UBracket.STL", UBRACKET, 3000)
        p.add_selection("bottom", S.face("ymin"))
        p.add_selection("top_right", S.box([0.75, 0.9, -1], [1.0, 1.0, 1]))
        p.bcs += [{"kind": "set_value", "on": "bottom", "value": 0.0},
                  {"kind": "source", "on": "top_right", "total": 1.0}]
        p.settings["common"]["maxIterations"] = 2
        p.complete_settings()
        assert p.validate() == [] and p.settings["advanced"]["template_coefficient_scale"] == 1.0

        q = _project("template", "UBracket.STL", UBRACKET, 3000)
        q.add_selection("bottom", S.face("ymin"))
        q.add_bc("fixed", "bottom", dofs="xyz")                         # a structural kind: not the template's
        assert any("not a kind of template support or load" in e for e in q.validate())

        mesh, _, bcs, _, _, _ = p.build(verbose=False)
        assert list(bcs) == ["c"] and len(bcs["c"].fixed_dofs) > 0
        assert np.isclose(bcs["c"].force.sum(), 1.0)                 # the source, shared by its nodes

        with tempfile.TemporaryDirectory() as d:
            run(p, save_dir=d, verbose=False)
            r = load_results.load(os.path.join(d, "run_data.npz"))
        assert list(r.results) == ["mean_value"] and r.results["mean_value"].shape == (r.n_elems,)
        assert r.result_info["mean_value"]["view"] == "Value"
        assert len(r.history["objective"]) >= 2                     # MMA evaluates the start too
        assert np.all(np.isfinite(r.history["objective"])) and r.history["objective"][0] > 0
        assert len(r.history["template_max_value"]) >= 2
    from lsr_mmto.formulation import quantities
    assert "DiffusionCompliance" not in {q.name for q in quantities.QUANTITIES}
    assert "template" not in available_physics.PHYSICS


def test_a_template_run_with_held_values_only():
    """The physics says what a project must have (NEEDS): the template needs a support and no load, so a
    run with two held values and no source goes through, and its save gives no RuntimeWarning (no load
    nodes, so no load centroid in the record); the structural physics still needs a load."""
    import warnings
    from lsr_mmto.problem import selections as S
    from lsr_mmto.problem.project import Project, run
    from lsr_mmto.results import load_results
    with registered(TEMPLATE):
        p = _project("held", "UBracket.STL", UBRACKET, 3000)
        p.add_selection("bottom", S.face("ymin"))
        p.add_selection("top_right", S.box([0.75, 0.9, -1], [1.0, 1.0, 1]))
        p.bcs += [{"kind": "set_value", "on": "bottom", "value": 0.0},
                  {"kind": "set_value", "on": "top_right", "value": 1.0}]
        p.objective("Mass()")                                       # f . c is zero with no source
        p.settings["common"]["maxIterations"] = 2
        assert p.validate() == []
        with tempfile.TemporaryDirectory() as d, warnings.catch_warnings(record=True) as heard:
            warnings.simplefilter("always")
            run(p, save_dir=d, verbose=False)
            r = load_results.load(os.path.join(d, "run_data.npz"))
        assert np.all(np.isfinite(r.history["objective"]))
        runtime = [f"{w.filename}:{w.lineno} {w.message}" for w in heard if issubclass(w.category, RuntimeWarning)]
        assert runtime == [], runtime
        summary = r.config["fea"]["loading_and_bc"]["c"]
        assert summary["load_node_count"] == 0 and "load_centroid_xyz" not in summary
    s = Project("structural", stl="UBracket.STL", n_dof=3000, single_layer=True, base_dir=UBRACKET)
    s.add_selection("bottom", S.face("ymin"))
    s.add_bc("fixed", "bottom", dofs="xyz")
    assert "no loads" in s.validate()


def test_the_saved_run_keeps_the_properties_the_physics_reads(monkeypatch):
    """A run saves each element's value of every property its physics reads (PROPERTIES), whatever they
    are: here the template's coefficient taken from Density, so nothing of the structural physics is
    needed to finish a run."""
    from lsr_mmto.optimize import save_results
    from lsr_mmto.physics.template import template_physics, template_quantities
    from lsr_mmto.problem import selections as S
    from lsr_mmto.problem.project import run
    saved = {}
    real = save_results.save_run_data

    def keep(**kw):
        saved.update(kw["fields"])
        return real(**kw)
    monkeypatch.setattr(save_results, "save_run_data", keep)
    monkeypatch.setattr(template_physics, "COEFFICIENT", "Density")
    monkeypatch.setattr(template_quantities, "COEFFICIENT", "Density")
    monkeypatch.setattr(TEMPLATE, "PROPERTIES", ["Density"])
    with registered(TEMPLATE):
        p = _project("template", "UBracket.STL", UBRACKET, 3000)
        p.add_selection("bottom", S.face("ymin"))
        p.add_selection("top_right", S.box([0.75, 0.9, -1], [1.0, 1.0, 1]))
        p.bcs += [{"kind": "set_value", "on": "bottom", "value": 0.0},
                  {"kind": "source", "on": "top_right", "total": 1.0}]
        p.settings["common"]["maxIterations"] = 2
        with tempfile.TemporaryDirectory() as d:
            run(p, save_dir=d, verbose=False)
    assert "property_Density" in saved and "property_Youngs_Modulus" not in saved
    assert not any(k.lower().startswith("youngs") for k in saved)


def test_a_result_the_physics_does_not_return_stops_the_save_with_a_message(monkeypatch):
    """final_summary returns the value of every RESULT_FIELDS entry; one left out is a clear error, not a
    result silently missing or taken from an earlier solve."""
    import pytest
    from lsr_mmto.physics.template import template_quantities
    from lsr_mmto.problem import selections as S
    from lsr_mmto.problem.project import run
    real = template_quantities.final_summary
    monkeypatch.setattr(TEMPLATE, "final_summary", lambda *a, **k: (real(*a, **k), {})[1])
    with registered(TEMPLATE):
        p = _project("template", "UBracket.STL", UBRACKET, 3000)
        p.add_selection("bottom", S.face("ymin"))
        p.add_selection("top_right", S.box([0.75, 0.9, -1], [1.0, 1.0, 1]))
        p.bcs += [{"kind": "set_value", "on": "bottom", "value": 0.0},
                  {"kind": "source", "on": "top_right", "total": 1.0}]
        p.settings["common"]["maxIterations"] = 2
        with tempfile.TemporaryDirectory() as d, pytest.raises(ValueError, match="returned no mean_value"):
            run(p, save_dir=d, verbose=False)


def test_a_large_template_problem_keeps_the_direct_solver(monkeypatch, capsys):
    """Above the direct solver's limit the loop uses DPCG only for a physics with a deflation space; the
    template has none, so it keeps PARDISO. The limit is lowered here so the test stays small."""
    import lsr_mmto.pyto_ad_imports as imports
    from lsr_mmto.problem import selections as S
    from lsr_mmto.problem.project import run
    monkeypatch.setattr(imports, "DIRECT_SOLVER_DOF_CUTOFF", 100)
    with registered(TEMPLATE):
        p = _project("template", "UBracket.STL", UBRACKET, 3000)
        p.add_selection("bottom", S.face("ymin"))
        p.add_selection("top_right", S.box([0.75, 0.9, -1], [1.0, 1.0, 1]))
        p.bcs += [{"kind": "set_value", "on": "bottom", "value": 0.0},
                  {"kind": "source", "on": "top_right", "total": 1.0}]
        p.settings["common"]["maxIterations"] = 2
        with tempfile.TemporaryDirectory() as d:
            run(p, save_dir=d, verbose=True)
    out = capsys.readouterr().out
    assert "Solver: PARDISO" in out and "Solver: DPCG" not in out


def test_a_large_structural_problem_still_uses_dpcg(monkeypatch, capsys):
    """The structural physics builds its deflation space, so above the limit it runs DPCG as before."""
    import lsr_mmto.pyto_ad_imports as imports
    from lsr_mmto.problem import selections as S
    from lsr_mmto.problem.project import Project, run
    monkeypatch.setattr(imports, "DIRECT_SOLVER_DOF_CUTOFF", 100)
    p = Project("structural", stl="UBracket.STL", n_dof=3000, single_layer=True, base_dir=UBRACKET)
    p.add_selection("bottom", S.face("ymin"))
    p.add_selection("top_right", S.box([0.75, 0.9, -1], [1.0, 1.0, 1]))
    p.add_bc("fixed", "bottom", dofs="xyz")
    p.add_bc("force", "top_right", total=[-1.0, 0.0, 0.0])
    p.database = "6Materials_AF_3Attr"
    p.objective("Compliance()")
    p.constraint("VolumeFraction()", "<=", 0.5)
    p.settings["common"]["maxIterations"] = 2
    with tempfile.TemporaryDirectory() as d:
        run(p, save_dir=d, verbose=True)
    assert "Solver: DPCG" in capsys.readouterr().out


def _test_penalty(weight, scale="first value"):
    """A penalty of the template for these tests: greyness of the densities plus the latent points' distance
    to 0.1, so its gradient has an x block and a z block."""
    return dict(name="TemplateTestPenalty", shown="Test penalty", help="A penalty for the tests only.",
                fn=lambda m: torch.mean(m.x_phys * (1 - m.x_phys)) + torch.mean((m.z - 0.1) ** 2),
                weight=weight, scale=scale, key="template_test_penalty")


def _held_and_sourced(name):
    from lsr_mmto.problem import selections as S
    p = _project(name, "UBracket.STL", UBRACKET, 3000)
    p.add_selection("bottom", S.face("ymin"))
    p.add_selection("top_right", S.box([0.75, 0.9, -1], [1.0, 1.0, 1]))
    p.bcs += [{"kind": "set_value", "on": "bottom", "value": 0.0},
              {"kind": "source", "on": "top_right", "total": 1.0}]
    return p


def test_a_physics_penalty_is_added_with_its_gradient(monkeypatch):
    """A penalty on the template's front page is added after the two every physics has, divided by its first
    value and times its weight, and the gradient of the objective with it agrees with central differences
    (the two every physics has are switched off, so the objective is mass plus this penalty)."""
    from lsr_mmto.optimize import optimization_loop as M
    from lsr_mmto.problem.project import _prepared, _run_arguments
    monkeypatch.setattr(TEMPLATE, "PENALTIES", [_test_penalty(lambda state, it: 5.0)])
    with registered(TEMPLATE), tempfile.TemporaryDirectory() as d:
        p = _held_and_sourced("penalty")
        p.objective("Mass()")
        p.settings["advanced"].update(use_penalization=False, use_z_smoothing=False)
        state = M.setup(**_run_arguments(_prepared(p), save_dir=d, verbose=False))
        assert [q.name for q in state.penalties] == ["LatentSmoothing", "DistancePenalty", "TemplateTestPenalty"]
        n = state.num_elems
        rng = np.random.default_rng(0)
        zeta = state.zeta0.flatten().copy()
        zeta[:n] = rng.uniform(0.2, 0.8, n)
        zeta[n:] = rng.uniform(state.lower.ravel()[n:], state.upper.ravel()[n:]) * 0.5
        f, g, _, _ = M.evaluate(state, zeta)              # the first: mass 1 (its first value), penalty 5
        assert np.isclose(f.item(), 6.0)
        assert state.history["template_test_penalty"] == [5.0]
        assert state.history["Pz_n"] == state.history["Pn"] == state.history["P"] == [0.0]

        everything = state.penalties
        state.penalties = everything[:2]                  # the same design without it: the mass's gradient alone
        g_mass = M.evaluate(state, zeta)[1]
        state.penalties = everything
        h = 1e-5
        for block in (rng.choice(n, 3, replace=False), n + rng.choice(len(zeta) - n, 3, replace=False)):
            for i in block:
                plus, minus = zeta.copy(), zeta.copy()
                plus[i] += h
                minus[i] -= h
                fd = (M.evaluate(state, plus)[0].item() - M.evaluate(state, minus)[0].item()) / (2 * h)
                assert abs(fd - g[i, 0]) <= 1e-5 * abs(g[i, 0]), (i, fd, g[i, 0])
            share = np.abs(g[block, 0] - g_mass[block, 0]).sum() / np.abs(g[block, 0]).sum()
            assert share > 0.1, share                     # the penalty's part of the gradient is not lost in it


def test_a_physics_penalty_keeps_its_history_and_its_schedule(monkeypatch):
    """A short template run with the test penalty on for its first iteration only: the saved run's history has
    its key, 0.5 at the first iteration (its first value, times 0.5) and 0 from then on; the history keys of
    the two every physics has are kept."""
    from lsr_mmto.problem.project import run
    from lsr_mmto.results import load_results
    monkeypatch.setattr(TEMPLATE, "PENALTIES", [_test_penalty(lambda state, it: 0.5 if state.iteration < 1 else 0)])
    with registered(TEMPLATE):
        p = _held_and_sourced("penalty")
        p.settings["common"]["maxIterations"] = 2
        with tempfile.TemporaryDirectory() as d:
            run(p, save_dir=d, verbose=False)
            r = load_results.load(os.path.join(d, "run_data.npz"))
    values = list(r.history["template_test_penalty"])
    assert len(values) >= 2 and values[0] == 0.5 and all(v == 0 for v in values[1:]), values
    assert all(len(r.history[k]) == len(values) for k in ("P", "Pn", "Pz_n"))


def test_a_quantity_of_another_physics_is_refused():
    """The project's check refuses a quantity of another physics, as the objective or as a constraint (the
    structural Compliance in a template project), and keeps the quantities every physics has and the
    template's own; a structural project may still name Compliance."""
    with registered(TEMPLATE):
        p = _held_and_sourced("other physics")
        p.complete_settings()
        assert p.validate() == []                           # DiffusionCompliance and VolumeFraction
        p.objective("Compliance()")
        p.constraint("Compliance()", "<=", 2.0)
        errors = [e for e in p.validate() if "quantity" in e]
        assert errors == ["objective: Compliance is not a template quantity",
                          "constraint 2: Compliance is not a template quantity"], errors
        assert all(e.step == "formulation" for e in p.validate() if "quantity" in e)
    from lsr_mmto.problem.project import Project
    s = Project.load(os.path.join(ROOT, "projects", "U-bracket.mmto"))
    s.objective("Compliance()")
    assert not any("is not a structural quantity" in e for e in s.validate())


def test_a_penalty_must_state_its_scale(monkeypatch):
    """A penalty entry without its scale, or with one that is not "raw" or "first value", is refused when the
    core's lists are made (at start-up), and nothing of it is added."""
    import pytest
    from lsr_mmto.design import latent_penalties
    entry = _test_penalty(lambda state, it: 0.5)
    del entry["scale"]
    monkeypatch.setattr(TEMPLATE, "PENALTIES", [entry])
    with pytest.raises(TypeError, match="scale"):
        with registered(TEMPLATE):
            pass
    monkeypatch.setattr(TEMPLATE, "PENALTIES", [_test_penalty(lambda state, it: 0.5, scale="normalised")])
    with pytest.raises(ValueError, match="scale must be one of 'raw', 'first value'"):
        with registered(TEMPLATE):
            pass
    assert [q.name for q in latent_penalties.PENALTIES] == ["LatentSmoothing", "DistancePenalty"]
    assert "template" not in available_physics.PHYSICS


if __name__ == "__main__":
    test_template_solver_gives_a_straight_line_along_a_bar()
    test_template_physics_runs_with_no_core_file_changed()
    print("ok (run with pytest for the tests that need monkeypatch)")


def test_the_mesh_is_sized_by_the_physics_unknowns_per_node():
    """Mesh dofs counts every unknown at every node: the template (one unknown per node) gets about three
    times the elements the structural physics (three per node) gets for the same number."""
    from lsr_mmto.problem.mesh import make_mesh
    with registered(TEMPLATE):
        p = _project("mesh size", "UBracket.STL", UBRACKET, 3000)
        assert p.unknowns_per_node() == 1
        one = make_mesh(p.stl_path(), p.n_dof, p.single_layer, p.unknowns_per_node())
        three = make_mesh(p.stl_path(), p.n_dof, p.single_layer, 3)
    assert 2.0 < one.num_elems / three.num_elems < 4.5, (one.num_elems, three.num_elems)

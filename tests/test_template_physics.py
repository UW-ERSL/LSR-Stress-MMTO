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
    from lsr_mmto.formulation import quantities
    from lsr_mmto.problem import boundary_conditions, settings
    available_physics._checked(module)
    added_q = [quantities.Quantity(physics=[module.NAME], **q) for q in module.QUANTITIES]
    added_s = [settings.Setting(physics=module.NAME, **a) for a in module.SETTINGS]
    available_physics.MODULES.append(module)
    available_physics.PHYSICS[module.NAME] = module
    quantities.QUANTITIES.extend(added_q)
    boundary_conditions.BC_KINDS.update({k: boundary_conditions.BCKind(**a)
                                         for k, a in module.SUPPORT_AND_LOAD_KINDS.items()})
    settings.SETTINGS.extend(added_s)
    settings._make_views()
    try:
        yield
    finally:
        available_physics.MODULES.remove(module)
        del available_physics.PHYSICS[module.NAME]
        for q in added_q:
            quantities.QUANTITIES.remove(q)
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
        q.add_support("bottom", dofs="xyz")                         # a structural kind: not the template's
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
    run with two held values and no source goes through; the structural physics still needs a load."""
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
        with tempfile.TemporaryDirectory() as d:
            run(p, save_dir=d, verbose=False)
            r = load_results.load(os.path.join(d, "run_data.npz"))
        assert np.all(np.isfinite(r.history["objective"]))
    s = Project("structural", stl="UBracket.STL", n_dof=3000, single_layer=True, base_dir=UBRACKET)
    s.add_selection("bottom", S.face("ymin"))
    s.add_support("bottom", dofs="xyz")
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
    p.add_support("bottom", dofs="xyz")
    p.add_force("top_right", total=[-1.0, 0.0, 0.0])
    p.database = "6Materials_AF_3Attr"
    p.objective("Compliance()")
    p.constraint("VolumeFraction()", "<=", 0.5)
    p.settings["common"]["maxIterations"] = 2
    with tempfile.TemporaryDirectory() as d:
        run(p, save_dir=d, verbose=True)
    assert "Solver: DPCG" in capsys.readouterr().out


if __name__ == "__main__":
    test_template_solver_gives_a_straight_line_along_a_bar()
    test_template_physics_runs_with_no_core_file_changed()
    print("ok (run with pytest for the tests that need monkeypatch)")

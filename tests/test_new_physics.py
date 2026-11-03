"""
test_new_physics.py
===================
A new physics needs no change to any core file: a small made-up one, "dummy", is given here as a
front page of its own (one field, w, with three unknowns per node; one quantity; two kinds of
support and load; one setting; one result field), registered as its line in MODULES would register
it, and a short U-bracket run is made with it. Its solve borrows the structural solver; what is
checked is that every entry of the front page reaches the place that uses it.

    python tests/test_new_physics.py        (or: python -m pytest tests)
"""
import contextlib
import os
import sys
import tempfile
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lsr_mmto.physics import available_physics  # noqa: E402
from lsr_mmto.physics.structural.structural_physics import MultiMaterialHexFEA  # noqa: E402
from lsr_mmto.pyto_ad_imports import MaterialModel  # noqa: E402

UBRACKET = os.path.join(ROOT, "data", "models", "UBracket")


# ----------------------------------------------------------------------------- the dummy front page
def _make_solver(mesh, mat_prop, bcs, linear_solver, dsolver, elem_body_force):
    return MultiMaterialHexFEA(mesh=mesh, mat_prop=mat_prop, bc=bcs["w"], solver=linear_solver, dsolver=dsolver,
                               rtol=1e-8, elem_body_force=elem_body_force)


def _analyse(m):
    E = m.props["Youngs_Modulus"]
    m.fe.set_material_fields(E, torch.full_like(E, float(m.settings.dummy_poissons_ratio)))
    m.w = m.fe.solve(m.x_phys, MaterialModel.SIMP)


def _clamp(bc, n, mesh, state, where):
    for d in (3 * n[:, None] + np.arange(3)[None, :]).ravel():
        state["values"][int(d)] = 0.0


def _push(bc, n, mesh, state, where):
    state["force"].reshape(-1, 3)[n] += np.asarray(bc["total"], dtype=float) / len(n)


def _final_summary(designs, solver, properties_of, settings, encoder, latent_dim, elem_volume,
                   design_domain_volume):
    _, x, z = designs[-1]
    with torch.no_grad():
        props = properties_of(torch.as_tensor(np.asarray(z), dtype=torch.get_default_dtype()))
        E = props["Youngs_Modulus"]
        solver.set_material_fields(E, torch.full_like(E, float(settings.dummy_poissons_ratio)))
        w = solver.solve(x, MaterialModel.SIMP).detach()
    solver.dummy_size = torch.linalg.norm(w[solver.edof].reshape(len(x), -1, 3), dim=2).mean(1).numpy()
    print(f"[DUMMY] largest element displacement: {solver.dummy_size.max():.4g}")


DUMMY = types.SimpleNamespace(
    __name__="tests.test_new_physics.DUMMY",
    NAME="dummy",
    DOC="a made-up physics for the test",
    FIELDS={"w": 3},
    PROPERTIES=["Youngs_Modulus"],
    make_solver=_make_solver,
    analyse=_analyse,
    SUPPORT_AND_LOAD_KINDS={
        "clamp": dict(doc="held still", role="support", field="w", check=lambda bc: [], apply=_clamp,
                      script=lambda bc, on: f"p.bcs.append({{'kind': 'clamp', 'on': {on}}})", fields=[],
                      shown="Clamp", help="Holds the selection still.", labels={}),
        "push": dict(doc="a total push", role="load", field="w", check=lambda bc: [], apply=_push,
                     script=lambda bc, on: f"p.bcs.append({{'kind': 'push', 'on': {on}, 'total': {bc['total']!r}}})",
                     fields=[("total", "vector", "force", "the total push")], shown="Push",
                     help="Pushes the selection.", labels={"total": "Total push"}),
    },
    supports_and_loads_error=lambda mesh, bcs: None,
    QUANTITIES=[dict(name="Give", objective=True, constraint=False, needs=[],
                     fn=lambda m, opt: torch.dot(m.fe.total_force, m.w), doc="F . w", shown="Give",
                     help="How much the part gives.", unit="{force}*{length}")],
    SETTINGS=[dict(section="advanced", key="dummy_poissons_ratio", default=0.3, label="Poisson's ratio",
                   help="One Poisson's ratio for every element.", group="Material model")],
    RESULT_FIELDS={"displacement_size": dict(view="Displacement", title="Displacement", measures="length",
                                              value=lambda solver: solver.dummy_size)},
    iteration_diagnostics=lambda history, iteration, solver, x_phys, properties, *a, **k: None,
    final_summary=_final_summary,
)


@contextlib.contextmanager
def registered(module):
    """The core's lists with module added, as its line in available_physics.MODULES adds it when they are
    made, and taken out again afterwards."""
    from lsr_mmto.formulation import quantities
    from lsr_mmto.problem import boundary_conditions, settings
    available_physics._checked(module)
    added_q = [quantities.Quantity(physics=[module.NAME], **q) for q in module.QUANTITIES]
    added_s = [settings.Setting(**a) for a in module.SETTINGS]
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


def dummy_project(base_dir):
    from lsr_mmto.problem import selections as S
    from lsr_mmto.problem.project import Project
    p = Project("dummy", stl="UBracket.STL", n_dof=3000, single_layer=True, base_dir=base_dir)
    p.physics = "dummy"
    p.add_selection("bottom", S.face("ymin"))
    p.add_selection("top_right", S.box([0.75, 0.9, -1], [1.0, 1.0, 1]))
    p.bcs += [{"kind": "clamp", "on": "bottom"}, {"kind": "push", "on": "top_right", "total": [-1.0, 0.0, 0.0]}]
    p.database = "6Materials_AF_3Attr"
    p.objective("Give()")
    p.constraint("VolumeFraction()", "<=", 0.5)
    p.settings["common"]["maxIterations"] = 2
    return p


def test_a_new_physics_runs_with_no_core_file_changed():
    from lsr_mmto.problem.project import run
    from lsr_mmto.results import load_results
    with registered(DUMMY):
        p = dummy_project(UBRACKET)
        p.complete_settings()
        assert p.validate() == [] and p.settings["advanced"]["dummy_poissons_ratio"] == 0.3

        q = dummy_project(UBRACKET)
        q.add_support("bottom", dofs="xyz")                         # a structural kind: not the dummy's
        assert any("not a kind of dummy support or load" in e for e in q.validate())

        mesh, _, bcs, _, _, _ = p.build(verbose=False)
        assert list(bcs) == ["w"] and len(bcs["w"].fixed_dofs) > 0
        assert np.isclose(bcs["w"].force.reshape(-1, 3).sum(0)[0], -1.0)

        with tempfile.TemporaryDirectory() as d:
            run(p, save_dir=d, verbose=False)
            r = load_results.load(os.path.join(d, "run_data.npz"))
        assert list(r.results) == ["displacement_size"] and r.results["displacement_size"].shape == (r.n_elems,)
        assert r.result_info["displacement_size"]["view"] == "Displacement"
        assert len(r.history["objective"]) >= 2                     # MMA evaluates the start too
    from lsr_mmto.formulation import quantities
    assert "Give" not in {q.name for q in quantities.QUANTITIES} and "dummy" not in available_physics.PHYSICS


if __name__ == "__main__":
    test_a_new_physics_runs_with_no_core_file_changed()
    print("ok")

"""
test_benchmark_projects.py
==========================
The four benchmarks as project files (projects/*.pyto) against their coded versions
(StructuralFEAExamples.py, ExamplesPureStructural.py, comparison/run_both.py).

At each benchmark's own mesh size: the same fixed dofs, imposed values, force vector
and keep-solid elements, bit for bit; the same optimization parameters, VAE settings
and run settings. Then 5 iterations of each, both ways, on a coarser mesh: the same
histories to 1e-9 (the solver's round-off varies slightly from run to run).

The full-size meshes take a few minutes, the airplane bracket most of it.

    python tests/test_benchmark_projects.py        (or: python -m pytest tests)
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "comparison"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from utils import project_paths  # noqa: E402

project_paths.bootstrap()

import run_both  # noqa: E402  # type: ignore
from ExamplesPureStructural import MMTOExamplesPureStructural, getMMTOProblemPureStructural  # noqa: E402
from PyTOImports import topopt_material_model  # noqa: E402  # type: ignore
from project import Project, run_arguments, VAE_PARAMS  # noqa: E402  # type: ignore

PROJECTS = os.path.join(ROOT, "projects")
BENCHMARKS = ("L-bracket", "U-bracket", "beam", "Airplane bracket")
COARSE = {"L-bracket": 3000, "U-bracket": 3000, "beam": 6000, "Airplane bracket": 15000}
_PER_RUN = ("to_problem", "save_dir", "verbose", "plot_progress", "plot_bcs", "plotter", "saveNet")


def coded(example, n_dof=None):
    """The benchmark as coded, with run_both's overrides applied as run_topopt applies them."""
    args = run_both.run_args(example)
    out = getMMTOProblemPureStructural(MMTOExamplesPureStructural[args["to_problem"]], nDOFDesired=n_dof)
    to_params = out[4]
    if args["relative_filter_radius"] is not None:
        to_params.RelativeFilterRadius = args["relative_filter_radius"]
    if args["material_filter_radius"] is not None:
        to_params.MaterialFilterRadius = args["material_filter_radius"]
    if args["pnorm_exponent"] is not None:
        topopt_material_model._PNORM_EXPONENT = float(args["pnorm_exponent"])
    return out, float(topopt_material_model._PNORM_EXPONENT), args


def project(example):
    p = Project.load(os.path.join(PROJECTS, f"{example}.pyto"))
    out = p.build(verbose=False)
    return p, out, float(topopt_material_model._PNORM_EXPONENT)


def _keep(e):
    return None if e is None else np.unique(np.asarray(e, dtype=int))


def compare_problem(example):
    (mesh0, _, bc0, _, tp0, vae0), p0, args = coded(example)
    p, (mesh, _, bc, _, tp, vae), pn = project(example)
    assert mesh.num_elems == mesh0.num_elems and mesh.num_nodes == mesh0.num_nodes
    order0, order = np.argsort(bc0.fixed_dofs), np.argsort(bc.fixed_dofs)
    assert np.array_equal(np.asarray(bc0.fixed_dofs)[order0], np.asarray(bc.fixed_dofs)[order]), "fixed dofs"
    assert np.array_equal(np.asarray(bc0.dirichlet_values)[order0], np.asarray(bc.dirichlet_values)[order]), \
        "imposed values"
    assert np.array_equal(bc0.force, bc.force), "force vector"
    k0, k = _keep(tp0.ElemsToKeep), _keep(tp.ElemsToKeep)
    assert (k0 is None and k is None) or np.array_equal(k0, k), "keep-solid elements"
    for f in ("Objective", "Constraints", "ExtrudeZ", "KeepFixedElems", "nDOFDesired",
              "RelativeFilterRadius", "MaterialFilterRadius"):
        assert getattr(tp0, f) == getattr(tp, f), (f, getattr(tp0, f), getattr(tp, f))
    assert os.path.normpath(tp0.MaterialsExcelFile) == os.path.normpath(tp.MaterialsExcelFile)
    assert p0 == pn, ("p-norm exponent", p0, pn)
    for f in VAE_PARAMS:
        assert getattr(vae0, f) == getattr(vae, f), ("VAE", f)
    # run settings: what the project gives run_topopt against what run_both gives it
    mine = run_arguments(p)
    for k_, v in args.items():
        if k_ in _PER_RUN or k_ in ("relative_filter_radius", "material_filter_radius", "pnorm_exponent"):
            continue
        v_mine = mine[k_]
        v_mine = getattr(v_mine, "name", v_mine)
        assert v_mine == v, ("run setting", k_, v, v_mine)
    assert mine["n_dof_desired"] == tp0.nDOFDesired
    assert mine["saveNet"] is None and os.path.normpath(args["saveNet"]) == \
        os.path.normpath(os.path.splitext(tp.MaterialsExcelFile)[0] + ".nt")


def compare_run(example, iters=5):
    from test_run_project import _run                                  # type: ignore
    p = Project.load(os.path.join(PROJECTS, f"{example}.pyto"))
    p.n_dof = COARSE[example]
    p.settings["common"]["maxIterations"] = iters
    args = run_arguments(p, verbose=False)
    a = _run(p, args)["history"]
    b = _run(MMTOExamplesPureStructural[run_both.SETUP[example]["problem"]], args)["history"]
    assert a.keys() == b.keys()
    for k in a:
        x, y = np.asarray(a[k], dtype=float), np.asarray(b[k], dtype=float)
        assert x.shape == y.shape and np.allclose(x, y, rtol=1e-9, atol=0, equal_nan=True), (example, k)


def _make(name, fn):
    def test():
        fn(name)
    test.__name__ = f"test_{fn.__name__}_{name.lower().replace(' ', '_').replace('-', '_')}"
    return test


for _b in BENCHMARKS:
    for _fn in (compare_problem, compare_run):
        _t = _make(_b, _fn)
        globals()[_t.__name__] = _t
del _b, _fn, _t


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

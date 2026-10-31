"""
test_benchmark_projects.py
==========================
The four benchmarks (projects/*.pyto) against what their coded versions gave before
they were retired: tests/data/benchmarks.npz holds the fixed dofs, imposed values,
forces and keep-solid elements, tests/data/benchmarks.json the mesh size, the
optimization parameters, the VAE settings and the run settings.

At each benchmark's own mesh size: the dofs and elements exactly, the forces to 1e-12.
The full-size meshes take about half a minute.

    python tests/test_benchmark_projects.py        (or: python -m pytest tests)
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils import project_paths  # noqa: E402

project_paths.bootstrap()

from PyTOImports import topopt_material_model  # noqa: E402  # type: ignore
from project import Project, run_arguments, VAE_PARAMS  # noqa: E402  # type: ignore

PROJECTS = os.path.join(ROOT, "projects")
DATA = os.path.join(ROOT, "tests", "data", "benchmarks")
BENCHMARKS = ("L-bracket", "U-bracket", "beam", "Airplane bracket")
_PER_RUN = ("to_problem", "save_dir", "verbose", "plot_progress", "plot_bcs", "plotter", "saveNet",
            "relative_filter_radius", "material_filter_radius", "pnorm_exponent", "n_dof_desired")


def _names(t):
    return [getattr(x, "name", x) for x in t] if isinstance(t, tuple) else t


def compare_problem(example):
    arrays = np.load(DATA + ".npz")
    ref = json.load(open(DATA + ".json"))[example]
    key = example.replace(" ", "_").replace("-", "_")

    p = Project.load(os.path.join(PROJECTS, f"{example}.pyto"))
    mesh, _, bc, _, tp, vae = p.build(verbose=False)
    assert (mesh.num_elems, mesh.num_nodes) == (ref["num_elems"], ref["num_nodes"]), "mesh size"

    order = np.argsort(bc.fixed_dofs)
    assert np.array_equal(np.asarray(bc.fixed_dofs)[order], arrays[key + "__fixed_dofs"]), "fixed dofs"
    assert np.array_equal(np.asarray(bc.dirichlet_values)[order], arrays[key + "__dirichlet_values"]), \
        "imposed values"
    F = np.asarray(bc.force, dtype=float)
    assert F.size == ref["n_force"]
    assert np.array_equal(np.flatnonzero(F), arrays[key + "__force_index"]), "loaded dofs"
    assert np.allclose(F[arrays[key + "__force_index"]], arrays[key + "__force_value"], rtol=1e-12, atol=0), \
        "force values"
    keep = np.zeros(0, dtype=int) if tp.ElemsToKeep is None else np.unique(np.asarray(tp.ElemsToKeep, dtype=int))
    assert np.array_equal(keep, arrays[key + "__keep"]), "keep-solid elements"

    assert _names(tp.Objective) == ref["objective"], "objective"
    assert [_names(c) for c in tp.Constraints] == ref["constraints"], "constraints"
    for f in ("ExtrudeZ", "KeepFixedElems", "nDOFDesired", "RelativeFilterRadius", "MaterialFilterRadius"):
        assert getattr(tp, f) == ref[f], (f, getattr(tp, f), ref[f])
    assert os.path.basename(tp.MaterialsExcelFile) == ref["database"]
    assert float(topopt_material_model._PNORM_EXPONENT) == ref["pnorm_exponent"], "p-norm exponent"
    for f in VAE_PARAMS:
        assert getattr(vae, f) == ref["vae"][f], ("VAE", f)

    args = run_arguments(p, verbose=False)
    for k, v in ref["run_settings"].items():
        if k in _PER_RUN:
            continue
        mine = getattr(args[k], "name", args[k])
        assert mine == v, ("run setting", k, v, mine)
    assert args["saveNet"] is None                       # the network shipped with the database


def _make(name):
    def test():
        compare_problem(name)
    test.__name__ = f"test_compare_problem_{name.lower().replace(' ', '_').replace('-', '_')}"
    return test


for _b in BENCHMARKS:
    _t = _make(_b)
    globals()[_t.__name__] = _t
del _b, _t


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

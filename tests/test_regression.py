"""
test_regression.py
==================
The numbers a run makes must not change by accident. The first iterations of every shipped project are
compared with stored values: the materials' latent points, and the objective and every constraint at each
iteration, to 1 part in a million. The L-bracket, the U-bracket and the rectangular beam run exactly as
their projects define them; the beam and the airplane bracket on a coarser mesh (30000 dofs) to keep the
test short, and the airplane bracket a second time on the iterative solver (DPCG with deflation), which
its own mesh would use, by lowering the direct solver's dof limit for the test.

The first iterations are repeatable; later ones drift by tiny amounts from one run to the next,
because the solver adds numbers in a different order on several threads. So only the first
iterations are compared.

A change that is meant to change results (a new formulation, a new PyTO-AD pin) will fail this test.
Then store the new values on purpose, and say why in the commit message:

    python tests/test_regression.py --update

About two minutes.
"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lsr_mmto.problem.project import Project, run  # noqa: E402

STORED = os.path.join(ROOT, "tests", "regression.json")
# name: (project, iterations, mesh dofs or None for the project's own, direct solver's dof limit or None)
CASES = {
    "L-bracket": ("L-bracket", 10, None, None),
    "U-bracket": ("U-bracket", 5, None, None),
    "rectangular beam": ("rectangular beam", 5, None, None),
    "beam, 30000 dofs": ("beam", 3, 30000, None),
    "Airplane bracket, 30000 dofs": ("Airplane bracket", 3, 30000, None),
    "Airplane bracket, 30000 dofs, iterative solver": ("Airplane bracket", 3, 30000, 20000),
}
RELATIVE = 1e-6


def first_iterations(project, iterations, n_dof=None, direct_limit=None):
    """The materials' latent points and the objective and constraints of each of the first iterations."""
    import lsr_mmto.pyto_ad_imports as pyto_ad
    p = Project.load(os.path.join(ROOT, "projects", f"{project}.mmto"))
    p.settings["common"]["maxIterations"] = iterations
    if n_dof is not None:
        p.n_dof = n_dof
    seen = {"objective": [], "constraints": []}

    def progress(state):
        seen["objective"].append(state["objective"])
        seen["constraints"].append(state["constraints"])
        seen.setdefault("z_real", np.asarray(state["z_real"], float).tolist())

    saved = pyto_ad.DIRECT_SOLVER_DOF_CUTOFF
    if direct_limit is not None:
        pyto_ad.DIRECT_SOLVER_DOF_CUTOFF = direct_limit
    try:
        with tempfile.TemporaryDirectory() as d:
            run(p, save_dir=d, verbose=False, progress=progress)
    finally:
        pyto_ad.DIRECT_SOLVER_DOF_CUTOFF = saved
    return seen


def _close(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and np.all(np.abs(a - b) <= RELATIVE * np.maximum(np.abs(b), 1e-12))


def test_first_iterations_unchanged():
    stored = json.load(open(STORED))
    for name, case in CASES.items():
        now, then = first_iterations(*case), stored[name]
        assert _close(now["z_real"], then["z_real"]), f"{name}: the materials' latent points changed"
        for i, (o, c, o0, c0) in enumerate(zip(now["objective"], now["constraints"],
                                               then["objective"], then["constraints"])):
            assert _close(o, o0), f"{name}, iteration {i}: objective {o!r}, stored {o0!r}"
            assert _close(c, c0), f"{name}, iteration {i}: constraints {c}, stored {c0}"
        assert len(now["objective"]) == len(then["objective"]), name


if __name__ == "__main__":
    if "--update" in sys.argv[1:]:
        values = {name: first_iterations(*case) for name, case in CASES.items()}
        with open(STORED, "w") as fh:
            json.dump(values, fh, indent=1)
        print(f"stored {STORED}")
    else:
        test_first_iterations_unchanged()
        print("ok")

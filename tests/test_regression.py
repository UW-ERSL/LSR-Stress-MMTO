"""
test_regression.py
==================
The numbers a run makes must not change by accident. The first iterations of the L-bracket and the
U-bracket, exactly as their projects define them, are compared with stored values: the materials'
latent points, and the objective and every constraint at each iteration, to 1 part in a million.

The first iterations are repeatable; later ones drift by tiny amounts from one run to the next,
because the solver adds numbers in a different order on several threads. So only the first
iterations are compared.

A change that is meant to change results (a new formulation, a new PyTO-AD pin) will fail this test.
Then store the new values on purpose, and say why in the commit message:

    python tests/test_regression.py --update

About half a minute.
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
ITERATIONS = {"L-bracket": 10, "U-bracket": 5}
RELATIVE = 1e-6


def first_iterations(name, iterations):
    """The materials' latent points and the objective and constraints of each of the first iterations."""
    p = Project.load(os.path.join(ROOT, "projects", f"{name}.mmto"))
    p.settings["common"]["maxIterations"] = iterations
    seen = {"objective": [], "constraints": []}

    def progress(state):
        seen["objective"].append(state["objective"])
        seen["constraints"].append(state["constraints"])
        seen.setdefault("z_real", np.asarray(state["z_real"], float).tolist())

    with tempfile.TemporaryDirectory() as d:
        run(p, save_dir=d, verbose=False, progress=progress)
    return seen


def _close(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and np.all(np.abs(a - b) <= RELATIVE * np.maximum(np.abs(b), 1e-12))


def test_first_iterations_unchanged():
    stored = json.load(open(STORED))
    for name, iterations in ITERATIONS.items():
        now, then = first_iterations(name, iterations), stored[name]
        assert _close(now["z_real"], then["z_real"]), f"{name}: the materials' latent points changed"
        for i, (o, c, o0, c0) in enumerate(zip(now["objective"], now["constraints"],
                                               then["objective"], then["constraints"])):
            assert _close(o, o0), f"{name}, iteration {i}: objective {o!r}, stored {o0!r}"
            assert _close(c, c0), f"{name}, iteration {i}: constraints {c}, stored {c0}"
        assert len(now["objective"]) == len(then["objective"]), name


if __name__ == "__main__":
    if "--update" in sys.argv[1:]:
        values = {name: first_iterations(name, n) for name, n in ITERATIONS.items()}
        with open(STORED, "w") as fh:
            json.dump(values, fh, indent=1)
        print(f"stored {STORED}")
    else:
        test_first_iterations_unchanged()
        print("ok")

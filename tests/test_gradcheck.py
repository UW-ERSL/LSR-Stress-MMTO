"""
test_gradcheck.py
=================
The gradient check itself: gradcheck.check(p) sets a project up as run(p) does and compares the
autodiff gradient with central finite differences. On a coarse U-bracket, for one density and one
latent variable, the two must agree closely for at least one step size. About half a minute.

    python tests/test_gradcheck.py        (or: python -m pytest tests)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lsr_mmto import Project  # noqa: E402
from lsr_mmto.physics import gradcheck  # noqa: E402


def test_autodiff_matches_finite_differences():
    p = Project.load(os.path.join(ROOT, "projects", "U-bracket.mmto"))
    p.n_dof, p.single_layer = 3000, True
    table = gradcheck.check(p, n_x=1, n_z=1)
    best = {}
    for row in table.rows:                                   # variable, h, dJ FD, dJ autodiff, rel. diff, ...
        best[row[0]] = min(best.get(row[0], 1.0), float(row[4]))
    assert len(best) == 2 and any(v.startswith("x[") for v in best) and any(v.startswith("z") for v in best)
    assert all(diff < 1e-4 for diff in best.values()), best


if __name__ == "__main__":
    test_autodiff_matches_finite_differences()
    print("ok")

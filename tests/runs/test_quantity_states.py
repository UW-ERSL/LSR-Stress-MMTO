"""
test_quantity_states.py
=======================
A quantity may keep a state from one iteration of a run to the next (between_iterations,
formulation/quantity_states.py); FailureFactorMax is the first that does (adaptive normalization, Le et al.
2010): c times the p-norm failure factor, c held fixed in the gradient and carried from one iteration to the
next as c_next = a * r + (1 - a) * c, r the highest failure factor over the p-norm one. Checked on a made-up
design (its value, its gradient, its state) and in a short run of a one-material bar, where the scale each
iteration records follows the formula from the ratio the iteration before recorded.

    python tests/runs/test_quantity_states.py        (or: python -m pytest tests)
"""
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests", "physics"))

from lsr_mmto.formulation.quantities import find_quantity  # noqa: E402
from lsr_mmto.formulation.quantity_states import run_state  # noqa: E402
from settings_for_a_test import default_run_settings  # noqa: E402

BAR = os.path.join(ROOT, "data", "models", "RectangularBeam")
P, DAMPING = 6.0, 0.5


def _design(svm, in_a_run=True):
    """A made-up analysed design: von Mises svm at each element, yield strength 2 everywhere; in a run it has the
    run's states (empty), else none."""
    svm = torch.as_tensor(svm, dtype=torch.float64)
    m = SimpleNamespace(svm=svm, props={"Yield_Strength": torch.full_like(svm, 2.0)}, N=svm.shape[0],
                        settings=default_run_settings(pnorm_exponent=P, stress_scale_damping=DAMPING))
    if in_a_run:
        m.quantity_states = {}
    return m


def _pnorm(f):
    return float((np.sum(np.asarray(f) ** P) / len(f)) ** (1.0 / P))


def test_outside_a_run_the_value_is_the_highest_failure_factor():
    """A design analysed on its own (no run): FailureFactorMax is the highest stress / yield strength, exactly."""
    m = _design([1.0, 3.0, 0.5, 2.0], in_a_run=False)
    assert abs(float(find_quantity("FailureFactorMax").fn(m, None)) - 1.5) < 1e-12


def test_in_a_run_the_scale_starts_at_1_and_is_fixed_in_the_gradient():
    """In a run, the first iteration's FailureFactorMax is the p-norm failure factor (scale 1); with a scale c, its
    value and its gradient are c times the p-norm one's, so no gradient goes through c."""
    q, pn = find_quantity("FailureFactorMax"), find_quantity("FailureFactorPNorm")
    m = _design([1.0, 3.0, 0.5, 2.0])
    assert abs(float(q.fn(m, None)) - float(pn.fn(m, None))) < 1e-12
    run_state(m, "FailureFactorMax", None)["scale"] = 1.7
    svm = m.svm.clone().requires_grad_(True)
    m.svm = svm
    (g,) = torch.autograd.grad(q.fn(m, None), svm)
    (g_pn,) = torch.autograd.grad(pn.fn(m, None), svm)
    assert torch.allclose(g, 1.7 * g_pn, rtol=1e-12, atol=0)


def test_the_scale_moves_only_between_iterations():
    """Asking for the value many times leaves the state as it is; between_iterations moves it once each time,
    c_next = a * r + (1 - a) * c from c = 1: two iterations worked by hand, and what each returns for the
    history (the scale it used and r)."""
    q = find_quantity("FailureFactorMax")
    m = _design([1.0, 3.0, 0.5, 2.0])
    for _ in range(5):
        q.fn(m, None)
    assert run_state(m, "FailureFactorMax", None) == {}
    r1 = 1.5 / _pnorm([0.5, 1.5, 0.25, 1.0])
    out = q.between_iterations(m, None, run_state(m, "FailureFactorMax", None), 0)
    c1 = DAMPING * r1 + (1 - DAMPING) * 1.0
    assert abs(out["failure_factor_scale"] - 1.0) < 1e-12 and abs(out["failure_factor_ratio"] - r1) < 1e-12
    assert abs(run_state(m, "FailureFactorMax", None)["scale"] - c1) < 1e-12
    m.svm = torch.tensor([2.0, 2.0, 2.0, 0.0], dtype=torch.float64)
    r2 = 1.0 / _pnorm([1.0, 1.0, 1.0, 0.0])
    out = q.between_iterations(m, None, run_state(m, "FailureFactorMax", None), 1)
    assert abs(out["failure_factor_scale"] - c1) < 1e-12
    assert abs(run_state(m, "FailureFactorMax", None)["scale"] - (DAMPING * r2 + (1 - DAMPING) * c1)) < 1e-12


def test_each_entry_and_each_run_keeps_its_own_state():
    """Two entries of the formulation (one quantity, other options) keep separate states, and a new run starts
    with none; a damping outside (0, 1] is refused, naming the setting."""
    m = _design([1.0, 2.0])
    run_state(m, "FailureFactorMax", None)["scale"] = 3.0
    assert run_state(m, "FailureFactorMax", {"tau": 0.5}) == {}
    assert run_state(_design([1.0, 2.0]), "FailureFactorMax", None) == {}
    m.settings.stress_scale_damping = 0.0
    try:
        find_quantity("FailureFactorMax").between_iterations(m, None, run_state(m, "FailureFactorMax", None), 0)
    except ValueError as e:
        assert "Stress scale damping" in str(e), e
    else:
        raise AssertionError("a damping of 0 was accepted")


def test_a_run_carries_the_scale_from_iteration_to_iteration():
    """A short run of a one-material bar with FailureFactorMax at most 1 and at least 0.01 (two entries, one
    state): the history has the scale and the ratio of every iteration, once, the scale starts at 1, and each
    next scale is a * r + (1 - a) * c of the one before."""
    from lsr_mmto.materials.databases import write_database
    from lsr_mmto.problem import selections as S
    from lsr_mmto.problem.project import Project, run
    with tempfile.TemporaryDirectory() as d:
        db = write_database(os.path.join(d, "steel.xlsx"), ["Youngs_Modulus", "Poissons_Ratio", "Density",
                                                            "Yield_Strength"], ["MPa", None, "t/mm^3", "MPa"],
                            ["Steel"], [None], ["#4c72b0"], [[200000.0, 0.3, 7.85e-9, 250.0]])
        p = Project("quantity states", stl="RectangularBeam.STL", n_dof=1500, single_layer=True, base_dir=BAR)
        p.units, p.physics, p.database = "mmKS", "structural", db
        p.material_source, p.pinned_material = "one_database_material", "Steel"
        p.add_selection("xmin", S.face("xmin"))
        p.add_selection("xmax", S.face("xmax"))
        p.add_bc("fixed", "xmin", dofs="xyz")
        p.add_bc("force", "xmax", total=[0, -1, 0])
        p.objective("Mass()")
        p.constraint("FailureFactorMax()", "<=", 1.0)
        p.constraint("FailureFactorMax()", ">=", 0.01)
        p.settings["common"].update(maxIterations=4, stress_scale_damping=DAMPING)
        out = run(p, save_dir=os.path.join(d, "run"), verbose=False)
        with np.load(os.path.join(out, "run_data.npz")) as h:
            c, r = h["h_failure_factor_scale"].astype(float), h["h_failure_factor_ratio"].astype(float)
            n = len(h["h_objective"])
    assert len(c) == len(r) == n and c[0] == 1.0, (len(c), n, c[0])
    assert np.allclose(c[1:], DAMPING * r[:-1] + (1 - DAMPING) * c[:-1], rtol=1e-5), (c, r)


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

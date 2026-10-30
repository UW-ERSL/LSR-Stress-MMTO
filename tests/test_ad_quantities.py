"""
test_ad_quantities.py
=====================
Autodiff gradients of the objectives and constraints against central finite
differences, on a coarse U-bracket (n_dof_desired = 3000, about 12,000 DOF;
6-material database with Density and Yield_Strength).

    python tests/test_ad_quantities.py        (or: python -m pytest tests)

One model is set up, as run_topopt does, and every quantity is checked on it at
the same random design. For each quantity, a few density and latent variables
with a sizeable gradient are perturbed with several step sizes h; the smallest
relative difference over h must be below TOL.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils import project_paths  # noqa: E402

project_paths.bootstrap()

from PyTOImports import TO_QOI  # noqa: E402  # type: ignore
from utils.gradcheck import _model_from_setup  # noqa: E402

TOL = 1e-5
STEPS = (1e-3, 1e-4, 1e-5)                # as utils/gradcheck.py; below 1e-5 round-off in the solve takes over
N_DOF = 3000

# (name, objective, constraint): the objective or the constraint is checked
QUANTITIES = [
    ("objective COMPLIANCE", (TO_QOI.COMPLIANCE, None), None),
    ("objective MASS", (TO_QOI.MASS, None), None),
    ("objective PNORM_STRESS, FAILURE_FACTOR", (TO_QOI.PNORM_STRESS, "FAILURE_FACTOR"), None),
    ("constraint COMPLIANCE", None, (TO_QOI.COMPLIANCE, None, 1.0)),
    ("constraint MASS", None, (TO_QOI.MASS, None, 1.0)),
    ("constraint STRESS_FAILURE_FACTOR", None, (TO_QOI.STRESS_FAILURE_FACTOR, None, 1.0)),
]

_SETUP = {}


def _setup():
    """The U-bracket model at a random design, built once."""
    if _SETUP:
        return _SETUP
    sys.path.insert(0, os.path.join(ROOT, "comparison"))
    import run_both
    from ExamplesPureStructural import MMTOExamplesPureStructural  # type: ignore
    from MainPureStructural import BetaScheduleScheme, Z0InitMethod  # type: ignore

    args = run_both.run_args("U-bracket")
    args.update(to_problem=MMTOExamplesPureStructural[args["to_problem"]],
                beta_schedule_scheme=BetaScheduleScheme[args["beta_schedule_scheme"]],
                z0_init_method=Z0InitMethod[args["z0_init_method"]],
                n_dof_desired=N_DOF, maxIterations=1)
    s = _model_from_setup(args)
    model = s["model"]
    n = model.N
    rng = np.random.default_rng(0)
    zeta = np.empty_like(s["lb"])
    zeta[:n] = rng.uniform(0.05, 1.0, n)
    zeta[n:] = rng.uniform(s["lb"][n:], s["ub"][n:])
    _SETUP.update(model=model, beta=s["beta"], eta=s["eta"], zeta=zeta)
    return _SETUP


def _value_and_gradient(model, zeta, beta, eta, objective, constraint):
    if objective is not None:
        model.to_params.Objective = objective
        model.to_params.Constraints = []
    else:
        model.to_params.Objective = (TO_QOI.MASS, None)
        model.to_params.Constraints = [constraint]

    def value(z):
        J, cons = model.evaluate(z, beta, eta)
        return J if objective is not None else cons[0]

    q = value(zeta)
    g = model.chain_to_x(model.grad_phys(q))
    return (lambda z: float(value(z).detach())), g


def check(name, objective, constraint, n_x=3, n_z=3):
    """Smallest relative difference over the step sizes, for each picked variable."""
    s = _setup()
    model, zeta, beta, eta = s["model"], s["zeta"], s["beta"], s["eta"]
    saved = (model.to_params.Objective, list(model.to_params.Constraints))
    try:
        value, g = _value_and_gradient(model, zeta, beta, eta, objective, constraint)
        n = model.N
        rng = np.random.default_rng(1)
        picks = list(rng.choice(np.argsort(-np.abs(g[:n]))[:200], n_x, replace=False))
        if np.any(g[n:] != 0):                       # MASS, COMPLIANCE, ... all depend on z
            picks += list(n + rng.choice(np.argsort(-np.abs(g[n:]))[:200], n_z, replace=False))
        errors = {}
        for i in picks:
            best = np.inf
            for h in STEPS:
                zp, zm = zeta.copy(), zeta.copy()
                zp[i] += h
                zm[i] -= h
                fd = (value(zp) - value(zm)) / (2 * h)
                best = min(best, abs(fd - g[i]) / max(abs(g[i]), 1e-300))
            label = f"x[{i}]" if i < n else f"z{(i - n) // n + 1}[{(i - n) % n}]"
            errors[label] = best
        return errors
    finally:
        model.to_params.Objective, model.to_params.Constraints = saved


def _assert_ok(name, objective, constraint):
    errors = check(name, objective, constraint)
    worst = max(errors.values())
    assert worst < TOL, f"{name}: worst relative difference {worst:.1e} >= {TOL:.0e}: {errors}"
    return errors


def test_objective_compliance():
    _assert_ok(*QUANTITIES[0])


def test_objective_mass():
    _assert_ok(*QUANTITIES[1])


def test_objective_failure_factor():
    _assert_ok(*QUANTITIES[2])


def test_constraint_compliance():
    _assert_ok(*QUANTITIES[3])


def test_constraint_mass():
    _assert_ok(*QUANTITIES[4])


def test_constraint_stress_failure_factor():
    _assert_ok(*QUANTITIES[5])


def test_missing_column_is_named():
    """COST on a database without a Cost column says so, instead of a bare KeyError."""
    s = _setup()
    model = s["model"]
    saved = (model.to_params.Objective, list(model.to_params.Constraints))
    model.to_params.Objective = (TO_QOI.MASS, None)
    model.to_params.Constraints = [(TO_QOI.COST, None, 1.0)]
    try:
        model.evaluate(s["zeta"], s["beta"], s["eta"])
    except KeyError as e:
        assert "Cost column" in str(e), str(e)
    else:
        raise AssertionError("COST without a Cost column did not raise")
    finally:
        model.to_params.Objective, model.to_params.Constraints = saved


if __name__ == "__main__":
    failed = 0
    for name, objective, constraint in QUANTITIES:
        errors = check(name, objective, constraint)
        worst = max(errors.values())
        status = "ok  " if worst < TOL else "FAIL"
        failed += worst >= TOL
        print(f"{status} {name:42s} worst relative difference {worst:.1e}")
    try:
        test_missing_column_is_named()
        print("ok   missing database column is named")
    except AssertionError as e:
        failed += 1
        print(f"FAIL missing database column: {e}")
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)

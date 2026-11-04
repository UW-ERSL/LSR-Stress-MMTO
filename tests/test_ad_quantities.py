"""
test_ad_quantities.py
=====================
Autodiff gradients of the objectives and constraints against central finite
differences, on a coarse U-bracket (n_dof = 3000, about 12,000 DOF;
6-material database with Density and Yield_Strength).

    python tests/test_ad_quantities.py        (or: python -m pytest tests)

One model is set up, as _run_topopt does, and every quantity is checked on it at
the same random design. For each quantity, a few density and latent variables
with a sizeable gradient are perturbed with several step sizes h; the smallest
relative difference over h must be below TOL.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lsr_mmto.optimize.gradient_check import _model_from_setup  # noqa: E402

try:                                        # about a minute: skipped by  pytest -m "not slow"
    import pytest
    pytestmark = pytest.mark.slow
except ImportError:                         # run directly, without pytest
    pass

TOL = 1e-5
STEPS = (1e-3, 1e-4, 1e-5)                # as optimize/gradient_check.py; below 1e-5 round-off in the solve takes over
N_DOF = 3000

# (name, objective, constraint): the objective or the constraint is checked
QUANTITIES = [
    ("objective Compliance", ("Compliance", None), None),
    ("objective Mass", ("Mass", None), None),
    ("objective FailureFactorPNorm", ("FailureFactorPNorm", None), None),
    ("constraint Compliance", None, ("Compliance", None, 1.0)),
    ("constraint Mass", None, ("Mass", None, 1.0)),
    ("constraint FailureFactorPNorm", None, ("FailureFactorPNorm", None, 1.0)),
    ("objective StressPNorm", ("StressPNorm", None), None),
    ("constraint VolumeFraction", None, ("VolumeFraction", None, 0.4)),
    ("constraint MaterialVolumeFraction, material 2", None,
     ("MaterialVolumeFraction", {"material_id": 2, "tau": 0.5}, 0.15)),
    ("constraint MassFraction, the lightest", None, ("MassFraction", {"material_id": "lightest"}, 0.4)),
]
# Cost is the one entry not checked here: no database with a Cost column is small enough.
# Mass fraction is also checked against a hand calculation (test_mass_fraction_by_hand).

_SETUP = {}


def _setup():
    """The U-bracket model at a random design, built once."""
    if _SETUP:
        return _SETUP
    from lsr_mmto.problem.project import Project, _run_arguments

    p = Project.load(os.path.join(ROOT, "projects", "U-bracket.mmto"))
    p.n_dof = N_DOF
    p.settings["common"]["maxIterations"] = 1
    args = _run_arguments(p, verbose=False)
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
        model.to_params.Objective = ("Mass", None)
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


def _make_test(name, objective, constraint):
    def test():
        _assert_ok(name, objective, constraint)
    test.__name__ = "test_" + "_".join(name.lower().replace(",", "").split())
    return test


for _q in QUANTITIES:                     # one test per quantity, so pytest reports each
    _t = _make_test(*_q)
    globals()[_t.__name__] = _t
del _q, _t


def test_missing_column_is_named():
    """Cost on a database without a Cost column says so, instead of a bare KeyError."""
    s = _setup()
    model = s["model"]
    saved = (model.to_params.Objective, list(model.to_params.Constraints))
    model.to_params.Objective = ("Mass", None)
    model.to_params.Constraints = [("Cost", None, 1.0)]
    try:
        model.evaluate(s["zeta"], s["beta"], s["eta"])
    except KeyError as e:
        assert "Cost column" in str(e), str(e)
    else:
        raise AssertionError("Cost without a Cost column did not raise")
    finally:
        model.to_params.Objective, model.to_params.Constraints = saved


def test_mass_fraction_by_hand():
    """Mass fraction = mass / (reference density * design domain volume), the reference density the
    database's. Four elements of volume 0.002 (domain 0.008), x_phys 1, 0.5, 0, 0.25 with decoded densities
    2000, 4000, 6000, 8000: mass (2000 + 2000 + 0 + 2000) * 0.002 = 12. Database densities 1500, 7800, 2700:
    the lightest solid part weighs 1500 * 0.008 = 12 (fraction 1), the heaviest 62.4 (12 / 62.4), the third
    21.6 (12 / 21.6). As a constraint it is fraction / limit - 1."""
    from types import SimpleNamespace
    import torch
    from lsr_mmto.formulation.formulation_text import to_params_form
    from lsr_mmto.formulation.quantities import find_quantity
    enc = SimpleNamespace(rawData=np.array([[1500.0, 1e9], [7800.0, 2e11], [2700.0, 7e10]]),
                          materialAttributes={"Density": {"idx": 0}, "Youngs_Modulus": {"idx": 1}})
    m = SimpleNamespace(x_phys=torch.tensor([1.0, 0.5, 0.0, 0.25]), elem_volume=0.002, enc=enc,
                        props={"Density": torch.tensor([2000.0, 4000.0, 6000.0, 8000.0])})
    fn = find_quantity("MassFraction").fn
    for reference, expected in (("lightest", 1.0), ("heaviest", 12 / 62.4), (2, 12 / 21.6), (0, 1.0)):
        _, opt, _ = to_params_form(f"MassFraction({reference})", "constraint", 0.4)
        assert abs(float(fn(m, opt)) - expected) < 1e-12, (reference, float(fn(m, opt)), expected)


def test_mass_fraction_on_the_model():
    """On the U-bracket model at a random design: Mass fraction of the heaviest is the Mass quantity divided
    by the heaviest density the database file gives (read here on its own) times the number of elements
    times the element volume; and the constraint's value is that over its limit, minus 1."""
    from lsr_mmto.materials.databases import database_table
    s = _setup()
    model = s["model"]
    saved = (model.to_params.Objective, list(model.to_params.Constraints))
    model.to_params.Objective = ("Mass", None)
    model.to_params.Constraints = [("MassFraction", {"material_id": "heaviest"}, 0.4)]
    try:
        J, cons = model.evaluate(s["zeta"], s["beta"], s["eta"])
    finally:
        model.to_params.Objective, model.to_params.Constraints = saved
    _, attributes, _, values = database_table(model.to_params.MaterialsExcelFile)
    heaviest = values[:, attributes.index("Density")].max()
    expected = float(J.detach()) / (heaviest * model.N * model.elem_volume)
    assert 0 < expected < 1 and abs((float(cons[0].detach()) + 1) * 0.4 - expected) < 1e-12 * expected


def test_registry():
    """Names are unique, each finds its entry, and each is an objective, a constraint or both."""
    from lsr_mmto.formulation.quantities import QUANTITIES as REGISTRY, find_quantity
    names = [q.name for q in REGISTRY]
    assert len(names) == len(set(names)), names
    for q in REGISTRY:
        assert find_quantity(q.name) is q, q.name
        assert q.objective or q.constraint, q.name


if __name__ == "__main__":
    failed = 0
    for name, objective, constraint in QUANTITIES:
        errors = check(name, objective, constraint)
        worst = max(errors.values())
        status = "ok  " if worst < TOL else "FAIL"
        failed += worst >= TOL
        print(f"{status} {name:42s} worst relative difference {worst:.1e}")
    test_registry()
    print("ok   registry")
    try:
        test_missing_column_is_named()
        print("ok   missing database column is named")
    except AssertionError as e:
        failed += 1
        print(f"FAIL missing database column: {e}")
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)

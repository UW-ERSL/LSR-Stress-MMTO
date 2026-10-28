"""
gradcheck.py
============
The autodiff gradient against central finite differences.

check(run_args)   sets the problem up as run_topopt(**run_args) does, then, at a
                  random design, compares the gradient of the objective J and of
                  the first constraint with central differences on a few design
                  variables, for several step sizes h

This checks the part autograd differentiates through the solver:
x, z -> x_phys, E, nu -> u -> J and the constraints. The penalties are left out.

A correct gradient agrees better as h shrinks, until round-off in the solve takes
over; a gap that stays the same for every h points to a wrong gradient.
"""
from __future__ import annotations

import numpy as np


class _Stop(Exception):
    pass


def _model_from_setup(run_args):
    """Run run_topopt's setup and hand back the autodiff model, before MMA starts."""
    import MainPureStructural as M
    found = {}

    def intercept(nVar, nCon, fn, x0, lb, ub, **kw):
        cells = dict(zip(fn.__code__.co_freevars, fn.__closure__))
        found["model"] = cells["ad_model"].cell_contents
        found["beta"] = cells["beta_proj"].cell_contents
        found["eta"] = cells["eta_proj"].cell_contents
        found["lb"] = np.asarray(lb, dtype=float).ravel()
        found["ub"] = np.asarray(ub, dtype=float).ravel()
        raise _Stop

    saved = M.runMMA
    M.runMMA = intercept
    try:
        M.run_topopt(**{**run_args, "verbose": False, "plot_progress": False})
    except _Stop:
        pass
    finally:
        M.runMMA = saved
    if "model" not in found:
        raise RuntimeError("run_topopt finished without reaching the optimizer")
    return found


def check(run_args, n_x=3, n_z=3, steps=(1e-3, 1e-4, 1e-5), seed=0):
    """Finite differences against autodiff at one random design."""
    s = _model_from_setup(run_args)
    model, beta, eta, lb, ub = s["model"], s["beta"], s["eta"], s["lb"], s["ub"]
    heaviside = run_args.get("use_heaviside_projection", True)
    n = model.N

    rng = np.random.default_rng(seed)
    zeta = np.empty_like(lb)
    zeta[:n] = rng.uniform(0.05, 1.0, n)
    zeta[n:] = rng.uniform(lb[n:], ub[n:])

    J, cons = model.evaluate(zeta, beta, eta, use_heaviside=heaviside)
    gJ = model.chain_to_x(model.grad_phys(J))
    gC = model.chain_to_x(model.grad_phys(cons[0])) if cons else None

    def values(z):
        Jz, cz = model.evaluate(z, beta, eta, use_heaviside=heaviside)
        return float(Jz), (float(cz[0]) if cz else np.nan)

    # variables with a sizeable gradient, so the comparison is not lost in noise
    pick_x = rng.choice(np.argsort(-np.abs(gJ[:n]))[:200], n_x, replace=False)
    pick_z = n + rng.choice(np.argsort(-np.abs(gJ[n:]))[:200], n_z, replace=False)

    def rel(a, b):
        return abs(a - b) / max(abs(b), 1e-300)

    rows = []
    for i in list(pick_x) + list(pick_z):
        name = f"x[{i}]" if i < n else f"z{(i - n) // n + 1}[{(i - n) % n}]"
        for h in steps:
            zp, zm = zeta.copy(), zeta.copy()
            zp[i] += h
            zm[i] -= h
            (Jp, cp), (Jm, cm) = values(zp), values(zm)
            fdJ, fdC = (Jp - Jm) / (2 * h), (cp - cm) / (2 * h)
            row = [name, f"{h:.0e}", f"{fdJ:.6e}", f"{gJ[i]:.6e}", f"{rel(fdJ, gJ[i]):.1e}"]
            if gC is not None:
                row += [f"{fdC:.6e}", f"{gC[i]:.6e}", f"{rel(fdC, gC[i]):.1e}"]
            rows.append(row)

    headers = ["variable", "h", "dJ, FD", "dJ, autodiff", "rel. diff"]
    if gC is not None:
        headers += ["dg1, FD", "dg1, autodiff", "rel. diff"]
    from .display import Table
    return Table(rows, headers=headers,
                 title="Gradient check: central differences against autodiff",
                 note=f"random design, beta = {beta:g}; J = {float(J):.6g}")

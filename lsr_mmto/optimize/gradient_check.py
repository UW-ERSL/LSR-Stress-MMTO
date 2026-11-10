"""
gradient_check.py
=================
The autodiff gradient against central finite differences.

check(p)          sets the project p up as run(p) does (optimization_loop.setup), then, at a random design,
                  compares the gradient of the objective J and of the first
                  constraint with central differences on a few design variables,
                  for several step sizes h

This checks the part autograd differentiates through the solver:
x, z -> x_phys, E, nu -> u -> J and the constraints. The penalties are left out.

A correct gradient agrees better as h shrinks, until round-off in the solve takes
over; a gap that stays the same for every h points to a wrong gradient.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from lsr_mmto.problem.project import Project
    from lsr_mmto.results.tables import Table


def _model_from_setup(run_args: dict[str, Any]) -> dict[str, Any]:
    """Set the project up as a run does (optimization_loop.setup) and hand back the autodiff model, the
    Heaviside beta and eta it starts with, and the bounds of every design variable, before any design
    is evaluated: {"model", "beta", "eta", "lb", "ub"}, the bounds flat (n,). run_args: _run_arguments'."""
    from lsr_mmto.optimize import optimization_loop as M
    state = M.setup(**{**run_args, "verbose": False, "plot_progress": False})
    if state is None:
        raise RuntimeError("the project names no material database, so there is nothing to check")
    return {"model": state.model, "beta": state.beta, "eta": state.s.eta_proj,
            "lb": np.asarray(state.lower, dtype=float).ravel(), "ub": np.asarray(state.upper, dtype=float).ravel()}


def check(project: Project, n_x: int = 3, n_z: int = 3, steps: tuple[float, ...] = (1e-3, 1e-4, 1e-5),
          seed: int = 0) -> Table:
    """Finite differences against autodiff at one random design of the project: n_x density and n_z latent
    variables, each picked at random among the 200 of its kind with the largest |objective gradient| with
    respect to the raw design variables, each at every step size; the first constraint's columns only when
    there is one. The project is checked and copied first, as a run does. No results are saved; as in a run,
    a network is trained and saved when the settings ask for a new one or there is none to read."""
    import tempfile
    from lsr_mmto.problem.project import _prepared, _run_arguments
    with tempfile.TemporaryDirectory() as scratch:              # the check makes no results
        run_args = _run_arguments(_prepared(project), save_dir=scratch, verbose=False)
        s = _model_from_setup(run_args)
    model, beta, eta, lb, ub = s["model"], s["beta"], s["eta"], s["lb"], s["ub"]
    heaviside, filtered = model.to_params.HeavisideProjection, model.to_params.APPLY_FILTER_TO_DENSITY
    n = model.N

    rng = np.random.default_rng(seed)
    zeta = np.empty_like(lb)
    zeta[:n] = rng.uniform(0.05, 1.0, n)
    zeta[n:] = rng.uniform(lb[n:], ub[n:])

    J, cons = model.evaluate(zeta, beta, eta, use_heaviside=heaviside, apply_density_filter=filtered)
    gJ = model.chain_to_x(model.grad_phys(J))
    gC = model.chain_to_x(model.grad_phys(cons[0])) if cons else None

    def values(z: np.ndarray) -> tuple[float, float]:
        """(J, first constraint or nan) at design z."""
        Jz, cz = model.evaluate(z, beta, eta, use_heaviside=heaviside, apply_density_filter=filtered)
        return float(Jz.detach()), (float(cz[0].detach()) if cz else np.nan)

    # variables with a sizeable gradient, so the comparison is not lost in noise
    pick_x = rng.choice(np.argsort(-np.abs(gJ[:n]))[:200], n_x, replace=False)
    pick_z = n + rng.choice(np.argsort(-np.abs(gJ[n:]))[:200], n_z, replace=False)

    def rel(a: float, b: float) -> float:
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
    from lsr_mmto.results.tables import Table
    return Table(rows, headers=headers,
                 title="Gradient check: central differences against autodiff",
                 note=f"random design, beta = {beta:g}; J = {float(J.detach()):.6g}")

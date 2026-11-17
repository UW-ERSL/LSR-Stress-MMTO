"""
iterative_solve_check.py
========================
Whether each iterative solve of a run reached its tolerance. PyTO-AD's DPCG solve (pyto/solve/deflation.py)
stops after its iteration limit with only a printed warning when it has not converged, and the run goes on with
an answer that is not one. CheckedDeflationSolver is that solver, which keeps for each solve
whether it converged (by DPCG's own test: it converged unless it used up its iterations), the relative residual
|f - K x| / |f| of its answer, and the step of the run the solve was for: None during the iterations, a design's
label during the final check (set by final_design_analysis.py) or an analysis. The run creates it
(optimization_loop.problem_solver) and keeps it, so a physics need do nothing; the run record keeps what did not
converge, and the Summary and Compare show it (results/tables.py), so a number from a solve that failed is never
taken for a result. Only DPCG is checked: a physics whose system does not fit the deflation space is solved by
plain CG inside PyTO-AD, which warns but is not recorded here.

    CheckedDeflationSolver       the deflation solver of a run; .solves: [(step, relative residual, converged)]
    short_label(label)           a design's label without its explanation in brackets ("Binarized topology")
"""
import numpy as np

from lsr_mmto.pyto_ad_imports import deflation


class CheckedDeflationSolver(deflation.DeflationSolver):
    """PyTO-AD's DeflationSolver, which keeps, for every solve, the step it was for (step), the relative residual
    of its answer and whether it converged (solves)."""

    step: "str | None"
    solves: list[tuple["str | None", float, bool]]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.step = None
        self.solves = []

    def deflatedPCG(self, K, f, W, M, rtol=1e-8, maxIters=2000, verbose=False, coarse=None):
        """The solve, then whether it converged and its relative residual |f - K x| / |f| kept in solves. It
        converged unless it used up maxIters (last_iterations, PyTO-AD's own test, the one behind its warning).
        Reads self.step. Sets self.solves."""
        x = super().deflatedPCG(K, f, W, M, rtol, maxIters, verbose, coarse)
        f = np.asarray(f, dtype=float).ravel()
        f_norm = float(np.linalg.norm(f))
        Kx = np.asarray(K @ np.asarray(x, dtype=float).ravel()).ravel()          # one column, whatever the shapes
        residual = float(np.linalg.norm(f - Kx)) / f_norm if f_norm else 0.0
        converged = self.last_iterations < maxIters or residual <= rtol     # at the limit, the last step may land
        self.solves.append((self.step, residual, bool(converged)))
        return x


def short_label(label: str) -> str:
    """A final-check design's label without the explanation in brackets: "Binarized topology"."""
    return label.split(" (")[0]

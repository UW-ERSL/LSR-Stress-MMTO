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
taken for a result. PyTO-AD's solve reads one deflation space, the solver's W, and solves a system that does not
fit it by plain CG, which is not checked; so the solver keeps one space for each size of system (add_space), and a
physics' solver of another size takes its own with space_for (thermoelastic: the displacement's and the
temperature's), so every iterative solve is a checked one.

    CheckedDeflationSolver       the deflation solver of a run; .solves: [(step, relative residual, converged)]
    one_value_per_node_space     a deflation space for a system of one unknown per node (temperature)
    short_label(label)           a design's label without its explanation in brackets ("Binarized topology")
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from lsr_mmto.pyto_ad_imports import deflation


class CheckedDeflationSolver(deflation.DeflationSolver):
    """PyTO-AD's DeflationSolver, which keeps, for every solve, the step it was for (step), the relative residual
    of its answer and whether it converged (solves)."""

    step: "str | None"
    solves: list[tuple["str | None", float, bool]]
    spaces: "dict[int, sparse matrix]"            # the deflation spaces besides W, by the size of their system

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.step = None
        self.solves = []
        self.spaces = {}

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

    def add_space(self, W) -> None:
        """Another deflation space, for systems of W.shape[0] unknowns. Sets self.spaces."""
        self.spaces[W.shape[0]] = W

    def space_for(self, n: int) -> "CheckedDeflationSolver | _WithSpace":
        """The deflation solver to give a solver of n unknowns: this one, or, when another of its spaces is for n
        unknowns, this one with that space as W, its solves kept in this one's solves."""
        W = getattr(self, "W", None)
        if n in self.spaces and (W is None or W.shape[0] != n):
            return _WithSpace(self, self.spaces[n])
        return self


class _WithSpace:
    """A CheckedDeflationSolver with another of its spaces as W: everything else, its solve and its record, is
    that solver's, read or set (a step set here is the checked solver's step), but for the coarse solve
    (_coarse_space)."""

    def __init__(self, checked: CheckedDeflationSolver, W) -> None:
        object.__setattr__(self, "checked", checked)
        object.__setattr__(self, "W", W)

    def __getattr__(self, name: str):
        if name == "checked":                     # not set yet (a copy being made): no endless lookup
            raise AttributeError(name)
        return getattr(self.checked, name)

    def __setattr__(self, name: str, value) -> None:
        setattr(self.checked, name, value)        # W and checked are set once, in __init__

    def __deepcopy__(self, memo):
        return self                               # one record per run: a copy of the solver keeps writing to it

    def _coarse_space(self, K, W):
        """PyTO-AD's coarse space ((KW)^T and a solver of E = W^T K W) without the 1e-10 it adds to E's
        diagonal. This form of DPCG keeps any error of the coarse solve and lets it grow until the solve
        diverges, faster the worse the system is scaled; a temperature's system is (Dirichlet rows of 1 beside
        entries of 5e-6 on a U-bracket), so a shift of 1e-12 already made it diverge (relative residual about
        1e3 after 2000 iterations) where the exact E took 71 to 102. E is positive definite when every group
        has nodes (one_value_per_node_space) and the field is held somewhere; when it is not (a body held
        nowhere), PyTO-AD's shifted coarse space is used, and the solve, if it fails, is recorded as not
        converged."""
        KW = (K @ W).tocsr()
        E = W.transpose() @ KW
        E = ((E + E.transpose()) / 2).tocsc()
        try:
            solve = spla.splu(E, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                              options={"SymmetricMode": True}).solve
        except RuntimeError:                      # E singular: "Factor is exactly singular"
            return self.checked._coarse_space(K, W)
        return KW.transpose().tocsr(), solve


def one_value_per_node_space(dsolver: "deflation.DeflationSolver", mesh) -> "sparse matrix":
    """The deflation space of a system of one unknown per node (a temperature): a constant on each group of
    nodes, one column per group with nodes in it (nodes x groups). Uses the groups dsolver has (structural's, for
    thermoelastic); makes them when it has none. Constant plus linear in each group failed to converge on a
    near-binary design (2000 iterations, residual 2e-3, at 108k nodes) where the constant alone took 78."""
    if getattr(dsolver, "ws_nodeGroupNumber", None) is None:
        dsolver.create_deflation_groups(mesh, dsolver.getRecommendedNumberOfGroups(nDOF=mesh.num_nodes))
    used, column = np.unique(np.asarray(dsolver.ws_nodeGroupNumber), return_inverse=True)
    n = mesh.num_nodes
    return sp.csr_matrix((np.ones(n), (np.arange(n), column.ravel())), shape=(n, len(used)))


def short_label(label: str) -> str:
    """A final-check design's label without the explanation in brackets: "Binarized topology"."""
    return label.split(" (")[0]

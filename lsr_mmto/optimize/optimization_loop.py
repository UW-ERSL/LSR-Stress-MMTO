"""
The optimization loop: density topology + VAE-decoded materials, for any physics on the list in
physics/available_physics.py.

    run(p) -> _run_topopt(...) = optimize(setup(...))

setup(...)          builds everything before the first design (optimize/run_setup.py): the problem, the
                    material network, the solver, the filters, the autodiff model, the start design, the
                    bounds and the schedules; returns the RunState
evaluate(state, zeta)
                    one design through ITERATION_STEPS, in order (each a function step(state, it)):
                    stop_if_asked, analyse_design, objective_and_constraints, add_penalties,
                    chain_to_design_variables, record, carry_quantity_states, continuation; returns the
                    objective (scaled by its first value, with the penalties), its gradient, the
                    constraints and their gradients, with respect to the optimizer's variables
optimize(state)     the constraint method runs from the start design, asking evaluate() about each design;
                    then the run record (optimize/save_results.py) and the final analysis
                    (optimize/final_design_analysis.py)

THE DESIGN THROUGH ONE ITERATION:
    zeta = [x ; z]                         design vector (the optimizer's variables)
      x  -> x_filt = (H_density @ x)/Hs    density filter
      x_filt -> x_phys = Heaviside(x_filt) optional projection (density only)
      x_phys = 1 at the elements to keep solid
      z  -> decode (VAE) -> per-element material properties (the physics' PROPERTIES)
    the physics' analyse (its front page): the solve(x_phys) and what its quantities read
      -> objective/constraints
    all of the above is one torch tree; gradients by autograd, through PyTO-AD's solve
    the penalties (PENALTIES, design/latent_penalties.py: latent smoothing, the distance to the real
      materials, and the physics' own) added to the objective, each with its weight this iteration
    gradient carried back through Heaviside + density filter by autograd -> w.r.t. x

Latents z are raw design variables (no forward filter).

A CONSTRAINT METHOD decides what is done with the constraints. CONSTRAINT_METHODS holds one, MMA, which
hands them to its subproblem as constraints (g <= 0). How to add another: docs/how_to_add_features.md,
"A constraint method".
"""
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
import time

from lsr_mmto.pyto_ad_imports import mat_lib, MaterialModel, runMMA
from lsr_mmto.formulation.quantities import find_quantity
from lsr_mmto.formulation.quantity_states import run_state, state_key
from lsr_mmto.optimize.history import constraints_over_limit, log_design_state, log_objective_and_constraints
from lsr_mmto.optimize.final_design_analysis import run_postprocessing
from lsr_mmto.optimize.run_setup import RunState, setup

if TYPE_CHECKING:
    import pyvista

    from lsr_mmto.problem.project import Project
    from lsr_mmto.problem.settings import RunSettings


class _StopRequested(Exception):
    """Raised inside evaluate() when should_stop() says so; optimize() catches it."""


MMA_KKT_TOL = 1e-6


class Iteration:
    """The numbers of one design, as the steps of evaluate() compute them, in order. Only zeta is set at the
    start; each step's docstring says which of the fields below it sets. n design variables, m constraints,
    N elements."""

    zeta: np.ndarray                    # the design as the method gives it, then flat (n,)
    z: np.ndarray                       # latents, element-major (N, latent_dim)
    J_t: torch.Tensor                   # the objective in the model's tree
    cons_t: list[torch.Tensor]          # the constraints in the model's tree
    x_phys: np.ndarray                  # (N,)
    J_phys: float                       # the objective
    Jn: float                           # the objective over its first value
    obj: float                          # the objective the method gets: Jn plus the penalties
    grad_obj: np.ndarray                # (n,) w.r.t. (x_phys, z), then (n, 1) w.r.t. the method's variables
    cons: np.ndarray                    # (m, 1)
    grad_cons: np.ndarray               # (m, n), w.r.t. (x_phys, z), then w.r.t. the method's variables
    penalties: dict[str, float]         # {history key: value} of the penalties

    def __init__(self, zeta: np.ndarray) -> None:
        self.zeta = zeta


def _run_topopt(*, s: "RunSettings", project: "Project", save_dir: str, verbose: bool, plot_progress: bool,
                plot_bcs: bool, plotter: "pyvista.Plotter | None", progress: Callable[[dict[str, Any]], None] | None,
                should_stop: Callable[[], bool] | None, on_network_error: Callable[[str], bool] | None) -> None:
    """One run of a project: setup(), then optimize(). Project.run calls it with the project's settings."""
    state = setup(s=s, project=project, save_dir=save_dir, verbose=verbose, plot_progress=plot_progress,
                  plot_bcs=plot_bcs, plotter=plotter, progress=progress, should_stop=should_stop,
                  on_network_error=on_network_error)
    if state is None:
        return
    return optimize(state)


# ---------------------------------------------------------------------- the steps of one iteration
# evaluate(state, zeta) runs these in order for each design the constraint method asks about. Each
# step reads and writes the run state and the iteration's numbers; the order is the order that gives
# today's numbers, so a new step goes where its inputs are ready and before the steps that read it.

def stop_if_asked(state: RunState, it: Iteration) -> None:
    """The window's Stop: the run ends before this design is evaluated (raises _StopRequested when
    should_stop() says so)."""
    if state.should_stop is not None and state.should_stop():
        raise _StopRequested


def analyse_design(state: RunState, it: Iteration) -> None:
    """The design through the autodiff model: density filter and Heaviside x -> x_phys, decode z ->
    material properties, the physics' solve and quantities, in one torch tree; then the design's log
    and the physics' own per-iteration diagnostics, appended to the history.

    Reads it.zeta, state.beta, state.iteration. Sets it.zeta (flattened), it.z, it.J_t, it.cons_t,
    it.x_phys; the model's tree (MMTOModelAD.evaluate)."""
    s, to_params, model = state.s, state.to_params, state.model
    it.zeta = np.asarray(it.zeta).flatten()
    it.z = it.zeta[state.num_elems:].reshape(state.latent_dim, state.num_elems).T   # element-major (N, latentDim)
    it.J_t, it.cons_t = model.evaluate(it.zeta, state.beta, s.eta_proj,
                                       use_heaviside=to_params.HeavisideProjection,
                                       apply_density_filter=to_params.APPLY_FILTER_TO_DENSITY)
    # x_phys and the decoded properties for logging and diagnostics: the same numbers, detached
    it.x_phys = model.x_phys.detach().numpy().copy()
    material_properties = {k: v.detach() for k, v in model.props.items()}
    zTorch = model.z.detach()
    log_design_state(state.history, state.iteration, it.x_phys, it.z, material_properties,
                     zTorch, state.encoder, state.num_elems, state.beta, s.eta_proj, verbose=state.verbose)
    state.physics.iteration_diagnostics(state.history, state.iteration, model, it.x_phys,
                                        material_properties, state.elem_volume, state.design_domain_volume,
                                        plot_progress=state.plot_progress, plotter=state.plotter,
                                        verbose=state.verbose)


def objective_and_constraints(state: RunState, it: Iteration) -> None:
    """The objective and the constraints with their gradients with respect to (x_phys, z) by autograd;
    the objective divided by the size of its first value (never below 1e-12, as a penalty's scale), so a
    negative objective keeps its sign and is still minimized.

    Reads it.J_t, it.cons_t, state.objective_scale. Sets it.J_phys (the objective), it.Jn and it.obj (both
    the scaled objective), it.grad_obj (n,) scaled the same, it.cons (m, 1), it.grad_cons (m, n);
    state.objective_scale on the first design."""
    model = state.model
    it.obj = float(it.J_t.detach())
    it.grad_obj = model.grad_phys(it.J_t)
    it.cons = np.array([[float(c.detach())] for c in it.cons_t])
    it.grad_cons = np.array([model.grad_phys(c) for c in it.cons_t])

    it.J_phys = float(it.obj)
    if state.objective_scale is None:
        state.objective_scale = max(abs(it.J_phys), 1e-12)
    it.Jn = it.J_phys / float(state.objective_scale)
    it.grad_obj = it.grad_obj / float(state.objective_scale)
    it.obj = it.Jn


def add_penalties(state: RunState, it: Iteration) -> None:
    """Each penalty of the run (PENALTIES), in order, with its weight this iteration: weight * value /
    scale added to the objective and the same times the value's gradient to the objective's gradient;
    its history keys get what it added and its raw value (0 while its weight is 0).

    Reads it.obj, state.penalties, state.penalty_scales. Sets it.penalties ({history key:
    value}), it.obj, it.grad_obj (in place); state.penalty_scales (a penalty's first value). A penalty's
    weight(state, it) may change the run state too (design/latent_penalties.py)."""
    model = state.model
    it.penalties = {}
    for p in state.penalties:
        w = p.weight(state, it)
        raw = added = 0.0
        if w:
            value_t = p.fn(model)
            raw = float(value_t.detach())
            if p.scale == "first value":
                if p.name not in state.penalty_scales:
                    state.penalty_scales[p.name] = max(float(raw), 1e-12)   # its first value, kept
                scale = state.penalty_scales[p.name]
                added = float(w) * (raw / scale)
                it.obj = it.obj + added
                it.grad_obj += float(w) * model.grad_phys(value_t) / scale
            else:
                added = float(w) * raw
                it.obj = it.obj + added
                it.grad_obj += float(w) * model.grad_phys(value_t)
        if p.raw_key:
            it.penalties[p.raw_key] = raw
        it.penalties[p.key] = added


def chain_to_design_variables(state: RunState, it: Iteration) -> None:
    """The gradients carried back through the Heaviside projection and the density filter (autograd),
    to the optimizer's variables, in the shapes the constraint method gets.

    Reads it.grad_obj, it.grad_cons, it.cons. Sets it.grad_obj (n, 1), it.cons (m, 1), it.grad_cons (m, n)."""
    model, n = state.model, state.num_design_var
    it.grad_obj = model.chain_to_x(it.grad_obj)
    for i in range(it.grad_cons.shape[0]):
        it.grad_cons[i, :] = model.chain_to_x(it.grad_cons[i, :])
    it.grad_obj = np.array([it.grad_obj]).reshape((n, 1))
    it.cons = np.array(it.cons).reshape((-1, 1))
    it.grad_cons = np.array(it.grad_cons).reshape((len(it.cons), n))


def record(state: RunState, it: Iteration) -> None:
    """The iteration's line in the history and the log, and what the window's progress hears.

    Reads it.obj, it.J_phys, it.Jn, it.penalties, it.cons, it.zeta, it.x_phys, it.z, state.iteration.
    Sets state.last_zeta (a copy of the design). The window hears each element's material as the run's material
    source finds it."""
    log_objective_and_constraints(state.history, it.obj, it.J_phys, it.Jn, it.penalties, it.cons,
                                  state.to_params, verbose=state.verbose, material_labels=state.material_labels,
                                  summary_rows=state.physics.SUMMARY_ROWS)
    state.last_zeta = it.zeta.copy()
    if state.progress is not None:
        material = state.encoder.getClosestRealMaterialIndex(torch.as_tensor(it.z, dtype=state.z_real.dtype))
        state.progress({"iteration": int(state.iteration), "objective": float(it.J_phys),
                        "constraints": [float(c) for c in np.ravel(it.cons)], "x_phys": np.array(it.x_phys),
                        "z": np.array(it.z), "z_real": state.z_real.detach().cpu().numpy(),
                        "material": np.asarray(material.cpu())})


def carry_quantity_states(state: RunState, it: Iteration) -> None:
    """Each quantity of the formulation that keeps a state between iterations updates it from this design
    (its between_iterations, quantities.py), and what it returns is appended to the history; once for each
    state, so two entries that share one (state_key) move it once.

    Reads the model after this design's analysis, state.iteration. Sets the model's quantity_states (run_state),
    state.history (the keys the quantities return, one number each)."""
    model = state.model
    entries = [tuple(state.to_params.Objective)] + [tuple(c) for c in state.to_params.Constraints]
    done = set()
    for name, opt in (spec[:2] for spec in entries):
        q = find_quantity(name)
        if q.between_iterations is None or state_key(name, opt) in done:
            continue
        done.add(state_key(name, opt))
        recorded = q.between_iterations(model, opt, run_state(model, name, opt), state.iteration)
        for key, value in (recorded or {}).items():
            state.history.setdefault(key, []).append(float(value))


def continuation(state: RunState, it: Iteration) -> None:
    """The next iteration's count, SIMP penalties and Heaviside beta, each as its schedule says
    (optimize/schedules.py).

    Reads state.iteration, state.beta, state.simp_penalties. Sets state.iteration (one more), state.simp_penalties
    (handed to the physics' set_simp_penalties when one grew), state.beta (when the Heaviside projection is on
    and it grew)."""
    s = state.s
    state.iteration += 1
    grown = {name: schedule.next(state.simp_penalties[name], state.iteration)
             for name, schedule in state.simp_schedules.items()}
    if any(did for _, did in grown.values()):
        state.simp_penalties = {name: value for name, (value, _) in grown.items()}
        state.physics.set_simp_penalties(state.solver, state.simp_penalties)
    if state.to_params.HeavisideProjection:
        beta_new, beta_updated = state.beta_schedule.next(state.beta, state.iteration)
        if beta_updated:
            state.beta = beta_new
            if state.verbose:
                print(f"[PROJ] Updated beta -> {state.beta:.4g} (eta={s.eta_proj}, at most {s.beta_max:g})")


ITERATION_STEPS = [
    stop_if_asked,
    analyse_design,
    objective_and_constraints,
    add_penalties,
    chain_to_design_variables,
    record,
    carry_quantity_states,
    continuation,
]


def evaluate(state: RunState, zeta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One design through ITERATION_STEPS: (objective, its gradient, constraints, their gradients), the
    objective scaled and with the penalties, the gradients with respect to the optimizer's variables;
    shapes (1, 1), (n, 1), (m, 1), (m, n). zeta: the n design variables, any shape."""
    it = Iteration(zeta)
    for step in ITERATION_STEPS:
        step(state, it)
    return np.array([[float(it.obj)]]), it.grad_obj, it.cons, it.grad_cons


# ---------------------------------------------------------------------- the optimizer
def optimize(state: RunState) -> None:
    """The constraint method runs from the start design; then the run record and the final analysis
    (saved to state's save_dir). A Stop from the window ends the method and keeps the last design evaluated;
    another error of the method becomes a RuntimeError naming the constraints over their limit when some
    are, and is raised as it is otherwise.

    Reads state.iteration, state.last_zeta, state.beta, state.during_run. Sets state.during_run
    (stopped_at_iter, when stopped)."""
    s, to_params, history = state.s, state.to_params, state.history
    method = CONSTRAINT_METHODS[CONSTRAINT_METHOD]
    tStart = time.time()
    try:
        zetaOptimal, method_settings = method["run"](state, lambda zeta: evaluate(state, zeta),
                                                     state.zeta0, state.lower, state.upper)
    except _StopRequested:                     # stopped from outside: the last design evaluated
        zetaOptimal, method_settings = state.last_zeta, method["record"](state)
        state.during_run["stopped_at_iter"] = int(state.iteration)
        print(f"Stopped at iteration {state.iteration}; saving the design reached so far.")
    except Exception as e:                     # the optimizer failed: say which constraints were far off, if any
        over = constraints_over_limit(history, to_params, state.material_labels)
        if not over:
            raise
        raise RuntimeError(
            f"The optimizer could not continue at iteration {max(state.iteration - 1, 0)}. These constraints were "
            f"over their limit: " + "; ".join(over) + ". A constraint many times over its limit at the start "
            "usually cannot be met from the starting design: check the limit and its unit. "
            f"(The optimizer's own error: {type(e).__name__}: {e})") from e
    tEnd = time.time()
    print(f"Total optimization time: {tEnd - tStart:.2f} seconds")
    from lsr_mmto.optimize.save_results import run_record
    run_config = run_record(state, method_settings, runtime_sec=tEnd - tStart,
                            fixed_in_code={"mma_kkt_tol": MMA_KKT_TOL})

    project, model, matEncoder = state.project, state.model, state.encoder
    run_postprocessing(
        history=history,
        zetaOptimal=zetaOptimal,
        fe_solver=state.solver,
        checked=state.checked,
        matEncoder=matEncoder,
        mat_lib=mat_lib,
        MaterialModel=MaterialModel,
        physical_density=lambda x: model.physical_density(x, state.beta, s.eta_proj,
                                                          to_params.HeavisideProjection,
                                                          to_params.APPLY_FILTER_TO_DENSITY),
        zRealPoints=state.z_real,
        material_colors=state.material_colors,
        material_labels=state.material_labels,
        num_elems=state.num_elems,
        latentDim=state.latent_dim,
        elem_volume=state.elem_volume,
        design_domain_volume=state.design_domain_volume,
        binarize_topology=s.binarize_topology,
        snap_to_real_material=s.snap_to_real_material,
        save_dir=state.save_dir,
        run_config=run_config,
        example_name=getattr(project, "name", str(project)),
        physics=state.physics,
        properties_of=matEncoder.decode,
        settings=s,
    )


def mma_record(state: RunState) -> dict[str, str | float]:
    """What the run record keeps of the MMA run."""
    s = state.s
    return {"optimizer": "MMA (mmaWrapper)", "kktTol": MMA_KKT_TOL, "move_limit": float(s.mma_move_limit),
            "fTolerance": s.rel_conv_tol, "gTolerance": s.rel_conv_tol}


def mma(state: RunState,
        evaluate_design: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        zeta0: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, dict[str, str | float]]:
    """MMA with the constraints as its constraints (g <= 0): PyTO-AD's runMMA, at most to_params.MaxIterations
    iterations and the timeLimit setting; returns the design it ends at, flat (n,), and mma_record(state)."""
    s = state.s
    optResults = runMMA(
        state.num_design_var,
        len(state.to_params.Constraints),
        evaluate_design,
        zeta0.reshape(-1, 1),
        lower,
        upper,
        maxIterations=state.to_params.MaxIterations,
        timeLimitSecs=s.timeLimit,
        move_limit=float(s.mma_move_limit),
        kktTol=MMA_KKT_TOL,
        fTolerance=s.rel_conv_tol,
        gTolerance=s.rel_conv_tol,
        verbose=False,
    )
    return np.asarray(optResults[0]).flatten(), mma_record(state)


# ---------------------------------------------------------------------- the constraint methods
# A constraint method is {"run": run, "record": record}:
#   run(state, evaluate, zeta0, lower, upper) -> (zeta_optimal, record)
#       evaluate(zeta) gives, for a design, (objective, its gradient, constraints, their gradients),
#       the objective scaled and with the penalties, every gradient with respect to the optimizer's
#       variables (shapes (1, 1), (n, 1), (m, 1), (m, n)); constraints are met when <= 0.
#       zeta0 is the start design, lower and upper the bounds of every variable, all (n, 1).
#       run returns the design it ends at (n,) and what the run record keeps of how it ran.
#   record(state) -> what the run record keeps, also when the run is stopped from the window.
# Its settings are entries of SETTINGS (problem/settings.py), read as state.s.<name>.
CONSTRAINT_METHODS = {
    "MMA": {"run": mma, "record": mma_record},
}
CONSTRAINT_METHOD = "MMA"          # the one there is; a second method comes with the setting that chooses it

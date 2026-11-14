"""
The optimization loop: density topology + VAE-decoded materials, for any physics on the list in
physics/available_physics.py.

    run(p) -> _run_topopt(...) = optimize(setup(...))

setup(...)          builds everything before the first design: the problem, the material network, the
                    solver, the filters, the autodiff model (optimize/design_model.py), the start design
                    and the bounds; returns the RunState
evaluate(state, zeta)
                    one design through ITERATION_STEPS, in order (each a function step(state, it)):
                    stop_if_asked, analyse_design, objective_and_constraints, keep_solid,
                    latent_sensitivity_filter, add_penalties, chain_to_design_variables, record,
                    continuation; returns the objective (scaled by its first value, with the penalties),
                    its gradient, the constraints and their gradients, with respect to the optimizer's
                    variables
optimize(state)     the constraint method runs from the start design, asking evaluate() about each design;
                    then the run record and the final analysis (optimize/final_design_analysis.py)

THE DESIGN THROUGH ONE ITERATION:
    zeta = [x ; z]                         design vector (the optimizer's variables)
      x  -> x_filt = (H_density @ x)/Hs    density filter
      x_filt -> x_phys = Heaviside(x_filt) optional projection (density only)
      z  -> decode (VAE) -> per-element material properties (the physics' PROPERTIES)
    the physics' analyse (its front page): the solve(x_phys) and what its quantities read
      -> objective/constraints
    all of the above is one torch tree; gradients by autograd, through PyTO-AD's solve
    the penalties (PENALTIES, design/latent_penalties.py: latent smoothing, the distance to the real
      materials, and the physics' own) added to the objective, each with its weight this iteration
    gradient carried back through Heaviside + density filter by autograd -> w.r.t. x

Latents z are raw design variables (no forward filter); their *gradients* are optionally smoothed by
smooth_latent_gradient_inplace, gated by APPLY_LATENT_SENS_FILTER (switched off).

A CONSTRAINT METHOD decides what is done with the constraints. CONSTRAINT_METHODS holds one, MMA, which
hands them to its subproblem as constraints (g <= 0). How to add another: docs/how_to_add_features.md,
"A constraint method".
"""
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
import time
import os
import shutil
import sys

from lsr_mmto.materials.databases import in_shipped_folder, material_display
from lsr_mmto.pyto_ad_imports import (
    deflation,
    linear_solvers,
    hex_structural_fea,
    hex_element_stiffness,
    createFilters,
    mat_lib,
    MaterialModel,
    runMMA,
    initialize_SIMP_STRUCTURAL_PENALTY,
    initialize_SIMP_THERMAL_PENALTY,
    increment_SIMP_STRUCTURAL_PENALTY,
    increment_SIMP_THERMAL_PENALTY,
)
from lsr_mmto.materials.encoder import MaterialEncoder, NetworkRunCancelled, network_warning
from lsr_mmto.physics.available_physics import physics_module
from lsr_mmto.design.latent_penalties import (PENALTIES, smooth_latent_constraint_gradient_inplace,
                                              smooth_latent_gradient_inplace)
from lsr_mmto.design.latent_to_properties import decode
from lsr_mmto.optimize.design_model import MMTOModelAD
from lsr_mmto.problem.settings import BetaScheduleScheme, Z0InitMethod
from lsr_mmto.optimize.filters import create_density_and_material_filters
from lsr_mmto.formulation.quantities import find_quantity
from lsr_mmto.optimize.history import (
    compute_domain_volume,
    constraints_over_limit,
    init_run_history,
    log_design_state,
    log_objective_and_constraints,
)
from lsr_mmto.optimize.final_design_analysis import run_postprocessing

if TYPE_CHECKING:
    from types import ModuleType

    import pyvista
    from scipy.sparse import spmatrix

    from pyto.core.bc import BC
    from pyto.core.hex_mesher import HexMesher
    from pyto.core.mat_lib import Material
    from lsr_mmto.design.latent_penalties import Penalty
    from lsr_mmto.materials.databases import VAEParams
    from lsr_mmto.optimize.mmto_to_params import MMTOParams
    from lsr_mmto.problem.project import Project
    from lsr_mmto.problem.settings import RunSettings
def get_beta_schedule(scheme: BetaScheduleScheme) -> dict[str, float]:
    """Return the Heaviside beta-continuation parameters (PAPER scheme).

    Keys: beta_init/beta_max (range), beta_factor (growth per update),
    beta_update_every (iters between updates), beta_start_iter (first update).
    """
    if scheme == BetaScheduleScheme.PAPER:
        return {
            "beta_init": 1.0,
            "beta_max": 32,
            "beta_factor": 1.5,
            "beta_update_every": 5,
            "beta_start_iter": 50,
        }
    if scheme == BetaScheduleScheme.PAPER_BM1:
        # Banh L-bracket benchmark: beta_max 16, per Table 2.
        return {
            "beta_init": 1.0,
            "beta_max": 16,
            "beta_factor": 1.5,
            "beta_update_every": 5,
            "beta_start_iter": 50,
        }

    raise ValueError(f"Unknown beta schedule scheme: {scheme}")
    
def update_beta(beta_proj: float, mmaIterations: int, beta_schedule: dict[str, float]) -> tuple[float, bool]:
    """Decide whether to increase the Heaviside sharpness beta this iteration.

    Multiplies beta by beta_factor every beta_update_every iterations after
    beta_start_iter, capped at beta_max.

    Returns (beta_new, updated_flag).
    """
    if mmaIterations <= beta_schedule["beta_start_iter"]:
        return beta_proj, False

    if (mmaIterations % beta_schedule["beta_update_every"]) != 0:
        return beta_proj, False

    if beta_proj >= beta_schedule["beta_max"]:
        return beta_proj, False

    return min(beta_proj * beta_schedule["beta_factor"], beta_schedule["beta_max"]), True

class _StopRequested(Exception):
    """Raised inside evaluate() when should_stop() says so; optimize() catches it."""


APPLY_LATENT_SENS_FILTER = False      # (H_material@g)/Hs smoothing of the latent gradient blocks: switched off
MMA_MOVE_LIMIT, MMA_KKT_TOL = 0.05, 1e-6


class RunState:
    """Everything a run keeps from its setup to its end: what setup() built, and what changes from one
    iteration to the next (the iteration count, the Heaviside beta, the penalties' scales and weights).
    The steps of an iteration read and change it; nothing else is shared between them.
    Its fields are the keywords setup() builds it with, declared below; the declarations give types only.
    A step's "Reads" line names the fields of the iteration and those of the run state that change during the
    run; what setup built and does not change (the settings, to_params, the model) is not listed."""

    # what setup was given
    s: "RunSettings"
    to_problem: "Project"
    save_dir: str
    verbose: bool
    plot_progress: bool
    plotter: "pyvista.Plotter | None"
    progress: Callable[[dict[str, Any]], None] | None
    should_stop: Callable[[], bool] | None
    args_at_start: dict[str, Any]                 # the run's arguments as given, for the run record
    provenance: dict[str, Any]                    # save_results.provenance()
    save_net: str                                 # the network file, resolved
    # what it built
    physics: "ModuleType"
    mesh: "HexMesher"
    bcs: "dict[str, BC]"
    to_params: "MMTOParams"
    vae_params: "VAEParams"
    solver: Any                                   # the physics' solver, whatever its make_solver returns
    encoder: MaterialEncoder
    network_errors: dict[str, float]              # {attribute: % of its range}
    model: MMTOModelAD
    history: dict[str, list[float | np.ndarray]]
    material_colors: list[str]
    material_labels: list[str]
    z_real: torch.Tensor                          # the database's materials in latent space, (materials, latent_dim)
    num_elems: int
    latent_dim: int
    num_design_var: int                           # n = num_elems * (1 + latent_dim)
    elem_volume: float
    design_domain_volume: float
    H_material: "spmatrix"
    Hs_material: np.ndarray
    beta_schedule: dict[str, float]
    initial_density: float
    zeta0: np.ndarray                             # start design, (n, 1)
    lower: np.ndarray                             # bounds, (n, 1)
    upper: np.ndarray
    # what changes from one iteration to the next
    iteration: int
    beta: float
    objective_scale: float | None
    penalties: "list[Penalty]"
    penalty_scales: dict[str, float]
    gamma: float
    objective_history: list[float]
    gamma_ramp_started: bool
    use_z_smoothing: bool
    use_distance_penalty: bool
    during_run: dict[str, int]
    last_zeta: np.ndarray                         # the last design evaluated, flat (n,)

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


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


def _run_topopt(*, s: "RunSettings", to_problem: "Project", save_dir: str, verbose: bool, plot_progress: bool,
                plot_bcs: bool, plotter: "pyvista.Plotter | None", progress: Callable[[dict[str, Any]], None] | None,
                should_stop: Callable[[], bool] | None, on_network_error: Callable[[str], bool] | None) -> None:
    """One run of a project: setup(), then optimize(). Project.run calls it with the project's settings."""
    state = setup(s=s, to_problem=to_problem, save_dir=save_dir, verbose=verbose, plot_progress=plot_progress,
                  plot_bcs=plot_bcs, plotter=plotter, progress=progress, should_stop=should_stop,
                  on_network_error=on_network_error)
    if state is None:
        return
    return optimize(state)


def problem_solver(physics: "ModuleType", mesh: "HexMesher", mat_prop: "Material", bcs: "dict[str, BC]",
                   elem_body_force: np.ndarray | None, to_params: "MMTOParams") -> Any:
    """The physics' solver of the problem (its make_solver), with the linear solver chosen by size.

    Direct (PARDISO) for small problems; iterative (DPCG + deflation) beyond the DOF cutoff, for a
    physics that builds a deflation space (its deflation_space); a physics without one keeps the
    direct solver at every size. Mirrors topopt_mma.py so the MMTO driver uses the same tested
    iterative path as single-material. The adjoint solve in PyTO-AD's differentiable solve reuses
    fe_solver.solver and fe_solver.dsolver, so it follows this choice automatically. elem_body_force: one
    per element, or None (Project.build always gives None). Prints the solver chosen."""
    from lsr_mmto.pyto_ad_imports import DIRECT_SOLVER_DOF_CUTOFF    # dof limit for the direct solver; above it, DPCG

    linear_solver = linear_solvers.Solvers.PARDISO
    dsolver = deflation.DeflationSolver()
    if to_params.nDOFDesired > DIRECT_SOLVER_DOF_CUTOFF and physics.deflation_space is not None:
        linear_solver = linear_solvers.Solvers.DPCG
        physics.deflation_space(dsolver, mesh)
    print("Solver:", linear_solver.name)
    return physics.make_solver(mesh, mat_prop, bcs, linear_solver, dsolver, elem_body_force)


def setup(*, s: "RunSettings", to_problem: "Project", save_dir: str, verbose: bool, plot_progress: bool,
          plot_bcs: bool, plotter: "pyvista.Plotter | None", progress: Callable[[dict[str, Any]], None] | None,
          should_stop: Callable[[], bool] | None, on_network_error: Callable[[str], bool] | None) -> RunState | None:
    """
    Everything before the first design is evaluated: the problem built, the material network read or
    trained, the solver, the filters, the autodiff model, the start design and the bounds. Returns the
    RunState (None when the project names no material database).

    DENSITY FILTER + HEAVISIDE PROJECTION (density ONLY):
      x (MMA var) --density filter--> x_filt = (H@x)/Hs --projection--> x_phys = H_{beta,eta}(x_filt)

    Latents:
      z are raw MMA variables; no direct filtering of z is performed.

    The penalties added to the objective: PENALTIES (design/latent_penalties.py), those of the run's
    physics, in order; e.g. latent smoothing (density-weighted):
      Pz = lambda * sum_ell sum_e x_e * ( z_e^(ell) - ((H z^(ell))/Hs)_e )^2
    """
    _args_at_start = dict(s.as_dict(), to_problem=to_problem, save_dir=save_dir, verbose=verbose,
                          plot_progress=plot_progress, plot_bcs=plot_bcs, plotter=plotter)   # what the run records
    saveNet = s.saveNet                                     # resolved below: None is the .nt beside the database
    from lsr_mmto.optimize import save_results as _save
    _provenance = _save.provenance()   # the code as it is when the run starts

    history = init_run_history()
    if not hasattr(to_problem, "build"):
        raise TypeError(f"to_problem must be a Project, e.g. Project.load('projects/L-bracket.mmto'), "
                        f"not {to_problem!r}; the coded examples are now the project files in projects/")
    physics = physics_module(to_problem.physics)            # the front page of the project's physics
    penalties = [p for p in PENALTIES if p.applies_to(physics.NAME)]
    for key in (k for p in penalties for k in p.keys()):    # each penalty's history keys, new ones
        if key in history:
            raise ValueError(f"penalty history key {key!r} is used twice")
        history[key] = []
    problem = to_problem
    mesh, mat_prop, bcs, elem_body_force, to_params, vae_params = problem.build()
    print(f"[SWEEP] nDOFDesired={to_params.nDOFDesired}  "
          f"RelativeFilterRadius={to_params.RelativeFilterRadius}  "
          f"MaterialFilterRadius={to_params.MaterialFilterRadius}  "
          f"P={s.pnorm_exponent}")

    if to_params.MaterialsExcelFile is None:
        print("Please provide a valid MaterialsExcelFile in to_params.")
        return None

    matEncoder = MaterialEncoder(vae_params)
    matEncoder.readExcel(to_params.MaterialsExcelFile)
    material_colors, material_labels = material_display(to_params.MaterialsExcelFile)

    if saveNet is None:
        base, _ = os.path.splitext(to_params.MaterialsExcelFile)
        saveNet = base + ".nt"

    vae_file_exists = os.path.exists(saveNet) and os.path.getsize(saveNet) > 0

    if s.use_pretrained_vae and vae_file_exists:
        print(f"Loading pre-trained autoencoder from file: {saveNet}")
        matEncoder.loadAutoencoderFromFile(saveNet)
    else:
        if in_shipped_folder(saveNet):
            raise ValueError(f"training a new VAE would overwrite the shipped {os.path.basename(saveNet)}; to train "
                             f"a different network, copy the database to a folder of your own and use the copy")
        if vae_file_exists:                                 # one step of undo: the network being replaced
            previous = os.path.splitext(saveNet)[0] + ".previous.nt"
            shutil.copy2(saveNet, previous)
            print(f"Keeping the network being replaced as {previous}")
        print(f"Training autoencoder and saving to: {saveNet}")
        time_start = time.time()
        matEncoder.trainAutoencoder(
            vae_params.numEpochs,
            vae_params.klFactor,
            saveNet,
            vae_params.learningRate,
            vae_params.maxAttributeErrorPercent,
            should_stop=should_stop,            # a stop while training: TrainingStopped, no network saved
        )
        time_end = time.time()
        print(f"Autoencoder training time: {time_end - time_start:.2f} seconds")

    network_errors = matEncoder.encodingErrorsPercent()     # read or just trained, the network is kept either way
    warning = network_warning(network_errors, s.network_error_warning)
    if warning is not None:
        print(f"[WARNING] {warning[0].upper() + warning[1:]}.")
        if on_network_error is not None and not on_network_error(warning):
            raise NetworkRunCancelled(warning)

    with torch.no_grad():
        matEncoder.training_latents = matEncoder.vaeNet.encoder(
            matEncoder.scaledMaterialData
        ).cpu()
    matEncoder.printEncodingErrors()

    zRealPoints = matEncoder.training_latents
    zRealPoints_np = zRealPoints.detach().cpu().numpy().astype(float)

    solver = problem_solver(physics, mesh, mat_prop, bcs, elem_body_force, to_params)

    num_dof = solver.bc.num_dofs
    print(f"Number of DOF: {num_dof}")

    num_elems = mesh.num_elems
    latentDim = matEncoder.vae_params.latentDim
    num_design_var = num_elems + num_elems * latentDim
    print(f"Using latent dimension: {latentDim}")
    elem_volume = float(np.prod(solver.mesh.elem_size))
    design_domain_volume = compute_domain_volume(solver)
    print("Creating filter...")
    H_density, Hs_density, H_material, Hs_material = create_density_and_material_filters(
        solver, to_params, createFilters
    )
    ad_model = MMTOModelAD(
        solver, physics, matEncoder, H_density, Hs_density, H_material, Hs_material,
        zRealPoints_np, to_params, s, elem_volume,
    )

    # ---- visualize loads & boundary conditions before optimization ----
    if plot_bcs:
            print("[plot_bcs] opening interactive loads/BCs window...")
            solver.plot_mesh(
                title="Loads & BCs (red = applied force, black = fixed/symmetry DOFs)",
                plot_bc=True,
                rel_arrow_scale=0.1,
                offsetArrow=True,
                camera_position="iso",
                save_path=None,          # interactive window, no file
            )

    beta_schedule = get_beta_schedule(s.beta_schedule_scheme)

    if s.use_continuation:
        initialize_SIMP_STRUCTURAL_PENALTY(1.5)
        initialize_SIMP_THERMAL_PENALTY(1)
    else:
        initialize_SIMP_STRUCTURAL_PENALTY(3)
        initialize_SIMP_THERMAL_PENALTY(1)

    initialDensity = 0.5                       # or the bound of the first constraint that sets it
    for name, _, bound in to_params.Constraints:
        if find_quantity(name).sets_start_density:
            initialDensity = bound
            break

    d0 = initialDensity * np.ones(num_elems, dtype=float)
    z0 = np.zeros(latentDim * num_elems, dtype=float)

    if s.z0_init_method == Z0InitMethod.LIGHTEST:
        zLightest = matEncoder.getLightestMaterial()
        for i in range(latentDim):
            z0[i * num_elems : (i + 1) * num_elems] = zLightest[i]

    elif s.z0_init_method == Z0InitMethod.HEAVIEST:
        zHeaviest = matEncoder.getHeaviestMaterial()
        for i in range(latentDim):
            z0[i * num_elems : (i + 1) * num_elems] = zHeaviest[i]

    elif s.z0_init_method == Z0InitMethod.ORIGIN:
        z0[:] = 0.0

    elif s.z0_init_method == Z0InitMethod.UNIFORM:
        for i in range(latentDim):
            z0[i * num_elems : (i + 1) * num_elems] = np.random.uniform(
                -0.5, 0.5, size=num_elems
            )
    else:
        raise ValueError(f"Unknown s.z0_init_method: {s.z0_init_method}")

    zeta0 = np.concatenate((d0, z0), axis=0).reshape(-1, 1)

    lowerBound = np.zeros(num_design_var, dtype=float).reshape(-1, 1)
    upperBound = np.ones(num_design_var, dtype=float).reshape(-1, 1)

    lowerBound[num_elems:num_design_var] = np.min(zRealPoints.detach().cpu().numpy())
    upperBound[num_elems:num_design_var] = np.max(zRealPoints.detach().cpu().numpy())

    return RunState(
        # what setup was given
        s=s, to_problem=to_problem, save_dir=save_dir, verbose=verbose, plot_progress=plot_progress,
        plotter=plotter, progress=progress, should_stop=should_stop,
        args_at_start=_args_at_start, provenance=_provenance, save_net=saveNet,
        # what it built
        physics=physics, mesh=mesh, bcs=bcs, to_params=to_params, vae_params=vae_params, solver=solver,
        encoder=matEncoder, network_errors=network_errors, model=ad_model, history=history,
        material_colors=material_colors, material_labels=material_labels,
        z_real=zRealPoints, num_elems=num_elems, latent_dim=latentDim, num_design_var=num_design_var,
        elem_volume=elem_volume, design_domain_volume=design_domain_volume,
        H_material=H_material, Hs_material=Hs_material,
        beta_schedule=beta_schedule, initial_density=initialDensity,
        zeta0=zeta0, lower=lowerBound, upper=upperBound,
        # what changes from one iteration to the next
        iteration=0,                        # designs evaluated so far
        beta=float(beta_schedule["beta_init"]),
        objective_scale=None,               # the first objective; later ones are divided by it
        penalties=penalties,                # the penalties of the run's physics, in the order they are added
        penalty_scales={},                  # the first value of each penalty scaled by it, by name
        gamma=float(s.gamma_init),          # the distance penalty's weight
        objective_history=[],               # scaled objectives, for the start of the weight's ramp
        gamma_ramp_started=False,
        use_z_smoothing=s.use_z_smoothing, use_distance_penalty=s.use_penalization,   # switched off on schedule
        during_run={},                      # iteration at which a setting changed during the run
        last_zeta=zeta0.flatten().copy(),
    )


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
    it.z = it.zeta[state.num_elems:].reshape(state.latent_dim, -1).T   # numpy, element-major (N, latentDim)
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
    the objective divided by its first value.

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
        state.objective_scale = it.J_phys
    it.Jn = it.J_phys / float(state.objective_scale)
    it.grad_obj = it.grad_obj / float(state.objective_scale)
    it.obj = it.Jn


def keep_solid(state: RunState, it: Iteration) -> None:
    """Elements to keep solid get the objective's most favourable slope.

    Sets it.grad_obj, in place, at to_params.ElemsToKeep (nothing when None)."""
    if state.to_params.ElemsToKeep is not None:
        it.grad_obj[state.to_params.ElemsToKeep] = np.min(it.grad_obj)


def latent_sensitivity_filter(state: RunState, it: Iteration) -> None:
    """Smoothing of the latent gradient blocks (switched off: APPLY_LATENT_SENS_FILTER).

    Reads it.grad_cons (smoothed in place when switched on). Sets it.grad_obj (a copy, smoothed when
    switched on)."""
    it.grad_obj = it.grad_obj.copy()
    if APPLY_LATENT_SENS_FILTER:
        smooth_latent_gradient_inplace(it.grad_obj, state.H_material, state.Hs_material, state.num_elems,
                                       state.latent_dim)
        smooth_latent_constraint_gradient_inplace(it.grad_cons, state.H_material, state.Hs_material,
                                                  state.num_elems, state.latent_dim)


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
    Sets state.last_zeta (a copy of the design)."""
    log_objective_and_constraints(state.history, it.obj, it.J_phys, it.Jn, it.penalties, it.cons,
                                  state.to_params, verbose=state.verbose, material_labels=state.material_labels)
    state.last_zeta = it.zeta.copy()
    if state.progress is not None:
        state.progress({"iteration": int(state.iteration), "objective": float(it.J_phys),
                        "constraints": [float(c) for c in np.ravel(it.cons)], "x_phys": np.array(it.x_phys),
                        "z": np.array(it.z), "z_real": state.z_real.detach().cpu().numpy()})


def continuation(state: RunState, it: Iteration) -> None:
    """The next iteration's count, SIMP penalty and Heaviside beta.

    Reads state.iteration, state.beta. Sets state.iteration (one more), state.beta (when the Heaviside
    projection is on and the schedule raises it). The SIMP penalties are PyTO-AD's module-level values,
    raised by 0.25, up to PyTO-AD's maximum, whenever the new count is a multiple of 10 and continuation
    is on."""
    s = state.s
    state.iteration += 1
    if s.use_continuation and (state.iteration % 10 == 0):
        increment_SIMP_THERMAL_PENALTY(0.25)
        increment_SIMP_STRUCTURAL_PENALTY(0.25)
    if state.to_params.HeavisideProjection:
        beta_new, beta_updated = update_beta(
            beta_proj=state.beta,
            mmaIterations=state.iteration,
            beta_schedule=state.beta_schedule,
        )
        if beta_updated:
            state.beta = beta_new
            if state.verbose:
                print(f"[PROJ] Updated beta -> {state.beta:.4g} (eta={s.eta_proj}, scheme={s.beta_schedule_scheme.value})")


ITERATION_STEPS = [
    stop_if_asked,
    analyse_design,
    objective_and_constraints,
    keep_solid,
    latent_sensitivity_filter,
    add_penalties,
    chain_to_design_variables,
    record,
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
    # ---- assemble the full parameter manifest (no plotting; saved as YAML+pkl) ----
    from lsr_mmto.optimize import save_results as _save
    _run_args = dict(state.args_at_start, saveNet=state.save_net)   # as given, with the VAE file resolved
    _iters = None
    try:
        _iters = max((len(v) for v in history.values() if hasattr(v, "__len__")),
                     default=None)
    except Exception:
        pass
    to_problem = state.to_problem
    run_config = _save.build_run_config(
        example_name=getattr(to_problem, "name", str(to_problem)),
        run_args=_run_args, to_params=to_params, vae_params=state.vae_params,
        beta_schedule=state.beta_schedule, mesh=state.mesh, bcs=state.bcs, fields=state.physics.FIELDS,
        mat_encoder=state.encoder, z_real_points=state.z_real,
        material_colors=state.material_colors, material_labels=state.material_labels,
        mma_settings=method_settings,
        runtime_sec=(tEnd - tStart), iterations=_iters,
        comment=getattr(to_params, "Comment", ""))
    run_config["during_run"] = dict(state.during_run)
    run_config["network_errors_percent"] = state.network_errors   # what the run's materials were off by, per property
    run_config["network_reconstruction"] = state.encoder.reconstruction()   # and per material: database vs network
    run_config["provenance"] = state.provenance
    run_config["units"] = to_problem.unit_record()
    from lsr_mmto.pyto_ad_imports import topopt_material_model
    run_config["fixed_in_code"] = {
        "mma_move_limit": MMA_MOVE_LIMIT, "mma_kkt_tol": MMA_KKT_TOL,
        "apply_latent_sensitivity_filter": APPLY_LATENT_SENS_FILTER,
        "initial_density": float(state.initial_density),
        "simp_structural_penalty_at_end": float(topopt_material_model._SIMP_STRUCTURAL_PENALTY),
        "simp_thermal_penalty_at_end": float(topopt_material_model._SIMP_THERMAL_PENALTY),
        "stress_relaxation_exponent": float(topopt_material_model._SIMP_STRESS_RELAXATION),
        "void_stiffness_relative": float(topopt_material_model._EVOID_RELATIVE),
    }

    model, matEncoder = state.model, state.encoder
    run_postprocessing(
        history=history,
        zetaOptimal=zetaOptimal,
        fe_solver=state.solver,
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
        example_name=getattr(to_problem, "name", str(to_problem)),
        physics=state.physics,
        properties_of=lambda z: decode(matEncoder, z),
        settings=s,
    )


def mma_record(state: RunState) -> dict[str, str | float]:
    """What the run record keeps of the MMA run."""
    s = state.s
    return {"optimizer": "MMA (mmaWrapper)", "kktTol": MMA_KKT_TOL, "move_limit": MMA_MOVE_LIMIT,
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
        move_limit=MMA_MOVE_LIMIT,
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

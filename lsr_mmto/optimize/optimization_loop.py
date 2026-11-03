"""
The optimization loop: density topology + VAE-decoded materials, MMA optimizer, for any physics
on the list in physics/available_physics.py.

PIPELINE:
    zeta = [x ; z]                         design vector (MMA variables)
      x  -> x_filt = (H_density @ x)/Hs    density filter
      x_filt -> x_phys = Heaviside(x_filt) optional projection (density only)
      z  -> decode (VAE) -> per-element material properties (the physics' PROPERTIES)
    the physics' analyse (its front page): the solve(x_phys) and what its quantities read
      -> objective/constraints
    all of the above is one torch tree (optimize/design_model.py, MMTOModelAD);
    gradients by autograd, through PyTO-AD's solve
    optional penalties (latent z-smoothing, distance-to-real-material) added here
    gradient carried back through Heaviside + density filter by autograd -> w.r.t. x
    MMA takes one step.

Latents z are raw MMA variables (no forward filter); their *gradients* are
optionally smoothed by smooth_latent_gradient_inplace, gated by
APPLY_LATENT_SENS_FILTER (note: plain (H@g)/Hs smoothing, not Sigmund's
density-weighted form).
"""
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
from lsr_mmto.design.latent_penalties import (distance_penalty, latent_smoothing,
                                              smooth_latent_constraint_gradient_inplace,
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
def get_beta_schedule(scheme: BetaScheduleScheme):
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
    
def update_beta(beta_proj, mmaIterations, beta_schedule):
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
    The steps of an iteration read and change it; nothing else is shared between them."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class Iteration:
    """The numbers of one design, as the steps of evaluate() compute them, in order."""

    def __init__(self, zeta):
        self.zeta = zeta


def _run_topopt(*, s, to_problem, save_dir, verbose, plot_progress, plot_bcs, plotter, progress, should_stop,
                on_network_error):
    """One run of a project: setup(), then optimize(). Project.run calls it with the project's settings."""
    state = setup(s=s, to_problem=to_problem, save_dir=save_dir, verbose=verbose, plot_progress=plot_progress,
                  plot_bcs=plot_bcs, plotter=plotter, progress=progress, should_stop=should_stop,
                  on_network_error=on_network_error)
    if state is None:
        return
    return optimize(state)


def setup(*, s, to_problem, save_dir, verbose, plot_progress, plot_bcs, plotter, progress, should_stop,
          on_network_error):
    """
    Everything before the first design is evaluated: the problem built, the material network read or
    trained, the solver, the filters, the autodiff model, the start design and the bounds. Returns the
    RunState (None when the project names no material database).

    DENSITY FILTER + HEAVISIDE PROJECTION (density ONLY):
      x (MMA var) --density filter--> x_filt = (H@x)/Hs --projection--> x_phys = H_{beta,eta}(x_filt)

    Latents:
      z are raw MMA variables; no direct filtering of z is performed.

    Optional latent smoothing penalty (density-weighted):
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

    # --- Solver selection: direct (PARDISO) for small problems; iterative
    #     (DPCG + deflation) beyond the DOF cutoff, for a physics that builds a deflation space
    #     (its deflation_space); a physics without one keeps the direct solver at every size.
    #     Mirrors topopt_mma.py so the MMTO driver uses the same tested iterative path as
    #     single-material. The adjoint solve in PyTO-AD's differentiable solve reuses
    #     fe_solver.solver and fe_solver.dsolver, so it follows this choice automatically.
    from lsr_mmto.pyto_ad_imports import DIRECT_SOLVER_DOF_CUTOFF    # dof limit for the direct solver; above it, DPCG

    linear_solver = linear_solvers.Solvers.PARDISO
    dsolver = deflation.DeflationSolver()
    if to_params.nDOFDesired > DIRECT_SOLVER_DOF_CUTOFF and physics.deflation_space is not None:
        linear_solver = linear_solvers.Solvers.DPCG
        physics.deflation_space(dsolver, mesh)
    print("Solver:", linear_solver.name)

    solver = physics.make_solver(mesh, mat_prop, bcs, linear_solver, dsolver, elem_body_force)

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
        distance_scale=None,                # the distance penalty's own first value, its scale
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

def stop_if_asked(state, it):
    """The window's Stop: the run ends before this design is evaluated."""
    if state.should_stop is not None and state.should_stop():
        raise _StopRequested


def penalties_switched_off_on_schedule(state, it):
    """The latent smoothing and the distance penalty, each switched off from its set iteration on."""
    s = state.s
    if (
        state.use_z_smoothing
        and s.z_smoothing_disable_after_iter is not None
        and state.iteration >= s.z_smoothing_disable_after_iter
    ):
        state.use_z_smoothing = False
        state.during_run["z_smoothing_off_at_iter"] = int(state.iteration)
        if state.verbose:
            print(f"[INFO] z-smoothing penalty disabled at iteration {state.iteration}")
    # Disable distance-based penalty after specified iteration
    if (
        state.use_distance_penalty
        and not s.use_dynamic_gamma_scheduling
        and s.distance_penalty_disable_after_iter is not None
        and state.iteration >= s.distance_penalty_disable_after_iter
    ):
        state.use_distance_penalty = False
        state.during_run["distance_penalty_off_at_iter"] = int(state.iteration)
        if state.verbose:
            print(f"[INFO] Distance-based penalty disabled at iteration {state.iteration}")


def analyse_design(state, it):
    """The design through the autodiff model: density filter and Heaviside x -> x_phys, decode z ->
    material properties, the physics' solve and quantities, in one torch tree; then the design's log
    and the physics' own per-iteration diagnostics."""
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
    state.physics.iteration_diagnostics(state.history, state.iteration, state.solver, it.x_phys,
                                        material_properties, state.elem_volume, state.design_domain_volume,
                                        plot_progress=state.plot_progress, plotter=state.plotter,
                                        verbose=state.verbose)


def objective_and_constraints(state, it):
    """The objective and the constraints with their gradients with respect to (x_phys, z) by autograd;
    the objective divided by its first value."""
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


def keep_solid(state, it):
    """Elements to keep solid get the objective's most favourable slope."""
    if state.to_params.ElemsToKeep is not None:
        it.grad_obj[state.to_params.ElemsToKeep] = np.min(it.grad_obj)


def latent_sensitivity_filter(state, it):
    """Smoothing of the latent gradient blocks (switched off: APPLY_LATENT_SENS_FILTER)."""
    it.grad_obj = it.grad_obj.copy()
    if APPLY_LATENT_SENS_FILTER:
        smooth_latent_gradient_inplace(it.grad_obj, state.H_material, state.Hs_material, state.num_elems,
                                       state.latent_dim)
        smooth_latent_constraint_gradient_inplace(it.grad_cons, state.H_material, state.Hs_material,
                                                  state.num_elems, state.latent_dim)


def latent_smoothing_penalty(state, it):
    """The latent smoothing penalty added to the objective, while it is switched on."""
    s, model = state.s, state.model
    if state.use_z_smoothing:
        # RAW scaling, applied consistently every iteration. The z-smoothing
        # penalty is ZERO at a spatially-uniform start, so it has no valid
        # own-initial-value to normalize by (dividing by it would blow up).
        # z_smoothing_weight multiplies the raw roughness directly.
        Pz_t = latent_smoothing(model.x_phys, model.z, model.Hm, model.Hsm)
        it.Pzn = float(s.z_smoothing_weight) * float(Pz_t.detach())
        it.obj = it.obj + it.Pzn
        it.grad_obj += float(s.z_smoothing_weight) * model.grad_phys(Pz_t)
    else:
        it.Pzn = 0.0


def distance_penalty_step(state, it):
    """The distance-to-real-material penalty added to the objective, and its weight's growth."""
    s, model = state.s, state.model
    it.P = 0.0
    it.Pn = 0.0
    if state.use_distance_penalty and (s.use_dynamic_gamma_scheduling or state.iteration >= int(s.penalty_start_iter)):
        # RAW penalty (gamma = 1); weighting + normalization applied here
        Dp_t = distance_penalty(model.x_phys, model.z, model.z_real)
        penalty_raw = float(Dp_t.detach())

        if state.distance_scale is None:
            state.distance_scale = max(float(penalty_raw), 1e-12)   # freeze initial scale (once)

        # gamma is now a pure ratio: penalty's initial share of the objective
        it.P = float(penalty_raw)
        it.Pn = float(state.gamma) * (it.P / state.distance_scale)
        it.obj = it.obj + it.Pn
        # x enters the realizability penalty as a frozen weight (detached in the AD
        # model), so its x block is zero by construction, as before.
        it.grad_obj += float(state.gamma) * model.grad_phys(Dp_t) / state.distance_scale

        if s.use_dynamic_gamma_scheduling:
            state.objective_history.append(float(it.Jn))
            if not state.gamma_ramp_started:
                # the ramp starts once J changed by less than the plateau tolerance over the last window
                # iterations (not before the earliest start), or in time to reach gamma_max end_margin
                # iterations before the last
                _K, _rel_tol = int(s.gamma_plateau_window), float(s.gamma_plateau_tolerance)
                _min_iter = int(s.gamma_plateau_earliest_iter)
                _n_ramp = int(np.ceil(np.log(max(s.gamma_max / s.gamma_init, 1 + 1e-9))
                                      / np.log(max(s.gamma_factor, 1 + 1e-9))))
                _deadline = int(state.to_params.MaxIterations) - _n_ramp - int(s.gamma_ramp_end_margin)
                _J_hist = state.objective_history
                _settled = (state.iteration >= _min_iter and len(_J_hist) > _K and
                            abs(_J_hist[-1] - _J_hist[-1 - _K])
                            <= _rel_tol * max(abs(_J_hist[-1 - _K]), 1e-12))
                if _settled or state.iteration >= _deadline:
                    state.gamma_ramp_started = True
                    state.during_run["gamma_ramp_started_at_iter"] = int(state.iteration)
                    if state.verbose:
                        print(f"[DYN-GAMMA] ramp started at iter {state.iteration} "
                              f"({'plateau' if _settled else 'deadline'})")
            if state.gamma_ramp_started:
                state.gamma = min(state.gamma * s.gamma_factor, s.gamma_max)
        else:
            state.gamma = min(state.gamma * s.gamma_factor, s.gamma_max)


def chain_to_design_variables(state, it):
    """The gradients carried back through the Heaviside projection and the density filter (autograd),
    to the optimizer's variables, in the shapes the constraint method gets."""
    model, n = state.model, state.num_design_var
    it.grad_obj = model.chain_to_x(it.grad_obj)
    for i in range(it.grad_cons.shape[0]):
        it.grad_cons[i, :] = model.chain_to_x(it.grad_cons[i, :])
    it.grad_obj = np.array([it.grad_obj]).reshape((n, 1))
    it.cons = np.array(it.cons).reshape((-1, 1))
    it.grad_cons = np.array(it.grad_cons).reshape((len(it.cons), n))


def record(state, it):
    """The iteration's line in the history and the log, and what the window's progress hears."""
    log_objective_and_constraints(state.history, it.obj, it.J_phys, it.Jn, it.P, it.Pn, it.Pzn, it.cons,
                                  state.to_params, verbose=state.verbose, material_labels=state.material_labels)
    state.last_zeta = it.zeta.copy()
    if state.progress is not None:
        state.progress({"iteration": int(state.iteration), "objective": float(it.J_phys),
                        "constraints": [float(c) for c in np.ravel(it.cons)], "x_phys": np.array(it.x_phys),
                        "z": np.array(it.z), "z_real": state.z_real.detach().cpu().numpy()})


def continuation(state, it):
    """The next iteration's count, SIMP penalty and Heaviside beta."""
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
    penalties_switched_off_on_schedule,
    analyse_design,
    objective_and_constraints,
    keep_solid,
    latent_sensitivity_filter,
    latent_smoothing_penalty,
    distance_penalty_step,
    chain_to_design_variables,
    record,
    continuation,
]


def evaluate(state, zeta):
    """One design through ITERATION_STEPS: (objective, its gradient, constraints, their gradients), the
    objective scaled and with the penalties, the gradients with respect to the optimizer's variables."""
    it = Iteration(zeta)
    for step in ITERATION_STEPS:
        step(state, it)
    return np.array([[float(it.obj)]]), it.grad_obj, it.cons, it.grad_cons


# ---------------------------------------------------------------------- the optimizer
def optimize(state):
    """The constraint method runs from the start design; then the run record and the final analysis."""
    s, to_params, history = state.s, state.to_params, state.history
    tStart = time.time()
    try:
        zetaOptimal, method_settings = mma(state, lambda zeta: evaluate(state, zeta),
                                           state.zeta0, state.lower, state.upper)
    except _StopRequested:                     # stopped from outside: the last design evaluated
        zetaOptimal, method_settings = state.last_zeta, mma_record(state)
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


def mma_record(state):
    """What the run record keeps of the MMA run."""
    s = state.s
    return {"optimizer": "MMA (mmaWrapper)", "kktTol": MMA_KKT_TOL, "move_limit": MMA_MOVE_LIMIT,
            "fTolerance": s.rel_conv_tol, "gTolerance": s.rel_conv_tol}


def mma(state, evaluate_design, zeta0, lower, upper):
    """MMA with the constraints as its constraints (g <= 0)."""
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

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
from lsr_mmto.optimize.history import (
    compute_domain_volume,
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
    """Raised inside the MMA callback when should_stop() says so; _run_topopt catches it."""


def _run_topopt(*, s, to_problem, save_dir, verbose, plot_progress, plot_bcs, plotter, progress, should_stop,
                on_network_error):
    """
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
    use_z_smoothing, use_penalization = s.use_z_smoothing, s.use_penalization      # switched off during the run
    from lsr_mmto.optimize import save_results as _save
    _provenance = _save.provenance()   # the code as it is when the run starts

    history = init_run_history()
    if not hasattr(to_problem, "build"):
        raise TypeError(f"to_problem must be a Project, e.g. Project.load('projects/L-bracket.mmto'), "
                        f"not {to_problem!r}; the coded examples are now the project files in projects/")
    physics = physics_module(to_problem.physics)            # the front page of the project's physics
    problem = to_problem
    mesh_structural, mat_prop_struct, bcs, elem_body_force, to_params, vae_params = problem.build()
    print(f"[SWEEP] nDOFDesired={to_params.nDOFDesired}  "
          f"RelativeFilterRadius={to_params.RelativeFilterRadius}  "
          f"MaterialFilterRadius={to_params.MaterialFilterRadius}  "
          f"P={s.pnorm_exponent}")

    if to_params.MaterialsExcelFile is None:
        print("Please provide a valid MaterialsExcelFile in to_params.")
        return

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

    solver = linear_solvers.Solvers.PARDISO
    dsolver = deflation.DeflationSolver()
    if to_params.nDOFDesired > DIRECT_SOLVER_DOF_CUTOFF and physics.deflation_space is not None:
        solver = linear_solvers.Solvers.DPCG
        physics.deflation_space(dsolver, mesh_structural)
    print("Solver:", solver.name)

    fe_solver_structural = physics.make_solver(mesh_structural, mat_prop_struct, bcs, solver, dsolver,
                                               elem_body_force)

    num_dof = fe_solver_structural.bc.num_dofs
    print(f"Number of DOF: {num_dof}")

    num_elems = mesh_structural.num_elems
    latentDim = matEncoder.vae_params.latentDim
    num_design_var = num_elems + num_elems * latentDim
    print(f"Using latent dimension: {latentDim}")
    elem_volume = float(np.prod(fe_solver_structural.mesh.elem_size))
    design_domain_volume = compute_domain_volume(fe_solver_structural)
    print("Creating filter...")
    H_density, Hs_density, H_material, Hs_material = create_density_and_material_filters(
        fe_solver_structural, to_params, createFilters
    )
    ad_model = MMTOModelAD(
        fe_solver_structural, physics, matEncoder, H_density, Hs_density, H_material, Hs_material,
        zRealPoints_np, to_params, s, elem_volume,
    )


    # ---- visualize loads & boundary conditions before optimization ----
    if plot_bcs:
            print("[plot_bcs] opening interactive loads/BCs window...")
            fe_solver_structural.plot_mesh(
                title="Loads & BCs (red = applied force, black = fixed/symmetry DOFs)",
                plot_bc=True,
                rel_arrow_scale=0.1,
                offsetArrow=True,
                camera_position="iso",
                save_path=None,          # interactive window, no file
            )
    APPLY_LATENT_SENS_FILTER = False  # (H_material@g)/Hs smoothing of latent
                                      # gradient blocks; set False.
    MMA_MOVE_LIMIT, MMA_KKT_TOL = 0.05, 1e-6

    mmaIterations = 0
    obj0 = None
    Sd0 = None    # frozen reference scale for distance penalty (own initial value)
    _J_hist = []               # objective history for dynamic-gamma plateau trigger
    _dyn_ramp_started = False   # latches True once the dynamic ramp begins
    _during_run = {}            # iteration at which a setting changed during the run
    gamma = float(s.gamma_init)
    beta_schedule = get_beta_schedule(s.beta_schedule_scheme)
    beta_proj = float(beta_schedule["beta_init"])

    if s.use_continuation:
        initialize_SIMP_STRUCTURAL_PENALTY(1.5)
        initialize_SIMP_THERMAL_PENALTY(1)
    else:
        initialize_SIMP_STRUCTURAL_PENALTY(3)
        initialize_SIMP_THERMAL_PENALTY(1)

    def MMTO_optimization_function(zeta):
        """MMA callback: returns (obj, grad_obj, cons, grad_cons) for design zeta.

        Steps:
          1. split zeta into density x and latents z
          2. AD model: density filter and Heaviside x -> x_phys, decode z -> material properties,
             and solve, in one torch tree
          3. x_phys and the decoded properties, detached, for logging and diagnostics
          4. postprocess the solve; record stress-violation diagnostics into history
          5. obj/cons values from the AD model; their gradients w.r.t. (x_phys, z) by autograd
          6. normalize obj by obj0; add optional z-smoothing and distance penalties (autograd)
          7. latent-block sens filter, then Heaviside + density filter (autograd)
          8. log, update beta continuation / SIMP continuation
        """
        nonlocal mmaIterations, obj0, Sd0, gamma, beta_proj, use_z_smoothing, use_penalization, _dyn_ramp_started
        nonlocal _last_zeta
        if should_stop is not None and should_stop():
            raise _StopRequested
        if (
            use_z_smoothing
            and s.z_smoothing_disable_after_iter is not None
            and mmaIterations >= s.z_smoothing_disable_after_iter
        ):
            use_z_smoothing = False
            _during_run["z_smoothing_off_at_iter"] = int(mmaIterations)
            if verbose:
                print(f"[INFO] z-smoothing penalty disabled at iteration {mmaIterations}")
        # Disable distance-based penalty after specified iteration
        if (
            use_penalization
            and not s.use_dynamic_gamma_scheduling
            and s.distance_penalty_disable_after_iter is not None
            and mmaIterations >= s.distance_penalty_disable_after_iter
        ):
            use_penalization = False
            _during_run["distance_penalty_off_at_iter"] = int(mmaIterations)
            if verbose:
                print(f"[INFO] Distance-based penalty disabled at iteration {mmaIterations}")
        zeta = np.asarray(zeta).flatten()

        # ===================== SPLIT DESIGN VECTOR =====================
        z = zeta[num_elems:].reshape(latentDim, -1).T   # numpy, element-major (N, latentDim)

        # ============ DECODE + MAIN FEA: one torch tree (AD model) ============
        # zeta -> x_phys, decoded properties -> PyTO-AD solve -> stress -> J, constraints
        J_t, cons_t = ad_model.evaluate(zeta, beta_proj, s.eta_proj,
                                        use_heaviside=to_params.HeavisideProjection,
                                        apply_density_filter=to_params.APPLY_FILTER_TO_DENSITY)

        # x_phys and the decoded properties for logging and diagnostics: the same numbers, detached
        x_phys = ad_model.x_phys.detach().numpy().copy()
        material_properties = {k: v.detach() for k, v in ad_model.props.items()}
        zTorch = ad_model.z.detach()
        log_design_state(history, mmaIterations, x_phys, z, material_properties,
                         zTorch, matEncoder, num_elems, beta_proj, s.eta_proj, verbose=verbose)

        physics.iteration_diagnostics(history, mmaIterations, fe_solver_structural, x_phys, material_properties,
                                      elem_volume, design_domain_volume, plot_progress=plot_progress,
                                      plotter=plotter, verbose=verbose)

        # ============ OBJECTIVE / CONSTRAINTS / SENSITIVITIES =========
        # gradients w.r.t. (x_phys, z), the layout the manual code returned
        obj = float(J_t.detach())
        grad_obj = ad_model.grad_phys(J_t)
        cons = np.array([[float(c.detach())] for c in cons_t])
        grad_cons = np.array([ad_model.grad_phys(c) for c in cons_t])

        # ================= PENALTIES + NORMALIZATION ==================
        J_phys = float(obj)
        if obj0 is None:
            obj0 = J_phys

        Jn = J_phys / float(obj0)
        grad_obj = grad_obj / float(obj0)
        obj = Jn

        if to_params.ElemsToKeep is not None:
            grad_obj[to_params.ElemsToKeep] = np.min(grad_obj)

        grad_obj_phys = grad_obj.copy()
        if APPLY_LATENT_SENS_FILTER:
                    smooth_latent_gradient_inplace(grad_obj_phys, H_material, Hs_material, num_elems, latentDim)
                    smooth_latent_constraint_gradient_inplace(grad_cons, H_material, Hs_material, num_elems, latentDim)

        if use_z_smoothing:
            # RAW scaling, applied consistently every iteration. The z-smoothing
            # penalty is ZERO at a spatially-uniform start, so it has no valid
            # own-initial-value to normalize by (dividing by it would blow up).
            # z_smoothing_weight multiplies the raw roughness directly.
            Pz_t = latent_smoothing(ad_model.x_phys, ad_model.z, ad_model.Hm, ad_model.Hsm)
            Pzn = float(s.z_smoothing_weight) * float(Pz_t.detach())
            obj = obj + Pzn
            grad_obj_phys += float(s.z_smoothing_weight) * ad_model.grad_phys(Pz_t)
        else:
            Pzn = 0.0

        P = 0.0
        Pn = 0.0

        if use_penalization and (s.use_dynamic_gamma_scheduling or mmaIterations >= int(s.penalty_start_iter)):
            # RAW penalty (gamma = 1); weighting + normalization applied here
            Dp_t = distance_penalty(ad_model.x_phys, ad_model.z, ad_model.z_real)
            penalty_raw = float(Dp_t.detach())

            if Sd0 is None:
                Sd0 = max(float(penalty_raw), 1e-12)   # freeze initial scale (once)

            # gamma is now a pure ratio: penalty's initial share of the objective
            P = float(penalty_raw)
            Pn = float(gamma) * (P / Sd0)
            obj = obj + Pn
            # x enters the realizability penalty as a frozen weight (detached in the AD
            # model), so its x block is zero by construction, as before.
            grad_obj_phys += float(gamma) * ad_model.grad_phys(Dp_t) / Sd0

            if s.use_dynamic_gamma_scheduling:
                _J_hist.append(float(Jn))
                if not _dyn_ramp_started:
                    # the ramp starts once J changed by less than the plateau tolerance over the last window
                    # iterations (not before the earliest start), or in time to reach gamma_max end_margin
                    # iterations before the last
                    _K, _rel_tol = int(s.gamma_plateau_window), float(s.gamma_plateau_tolerance)
                    _min_iter = int(s.gamma_plateau_earliest_iter)
                    _n_ramp = int(np.ceil(np.log(max(s.gamma_max / s.gamma_init, 1 + 1e-9))
                                          / np.log(max(s.gamma_factor, 1 + 1e-9))))
                    _deadline = int(to_params.MaxIterations) - _n_ramp - int(s.gamma_ramp_end_margin)
                    _settled = (mmaIterations >= _min_iter and len(_J_hist) > _K and
                                abs(_J_hist[-1] - _J_hist[-1 - _K])
                                <= _rel_tol * max(abs(_J_hist[-1 - _K]), 1e-12))
                    if _settled or mmaIterations >= _deadline:
                        _dyn_ramp_started = True
                        _during_run["gamma_ramp_started_at_iter"] = int(mmaIterations)
                        if verbose:
                            print(f"[DYN-GAMMA] ramp started at iter {mmaIterations} "
                                  f"({'plateau' if _settled else 'deadline'})")
                if _dyn_ramp_started:
                    gamma = min(gamma * s.gamma_factor, s.gamma_max)
            else:
                gamma = min(gamma * s.gamma_factor, s.gamma_max)

        grad_obj = grad_obj_phys

        # ============== CHAIN RULE -> DESIGN VARIABLES ================
        grad_obj = ad_model.chain_to_x(grad_obj)
        for i in range(grad_cons.shape[0]):
            grad_cons[i, :] = ad_model.chain_to_x(grad_cons[i, :])

        grad_obj = np.array([grad_obj]).reshape((num_design_var, 1))
        cons = np.array(cons).reshape((-1, 1))
        grad_cons = np.array(grad_cons).reshape((len(cons), num_design_var))

        log_objective_and_constraints(history, obj, J_phys, Jn, P, Pn, Pzn, cons,
                                      to_params, verbose=verbose, material_labels=material_labels)
        _last_zeta = zeta.copy()
        if progress is not None:
            progress({"iteration": int(mmaIterations), "objective": float(J_phys),
                      "constraints": [float(c) for c in np.ravel(cons)], "x_phys": np.array(x_phys),
                      "z": np.array(z), "z_real": zRealPoints.detach().cpu().numpy()})

        # ================ CONTINUATION + BETA UPDATE ==================
        mmaIterations += 1

        if s.use_continuation and (mmaIterations % 10 == 0):
            increment_SIMP_THERMAL_PENALTY(0.25)
            increment_SIMP_STRUCTURAL_PENALTY(0.25)
        if to_params.HeavisideProjection:
            beta_new, beta_updated = update_beta(
                beta_proj=beta_proj,
                mmaIterations=mmaIterations,
                beta_schedule=beta_schedule,
            )
            if beta_updated:
                beta_proj = beta_new
                if verbose:
                    print(f"[PROJ] Updated beta -> {beta_proj:.4g} (eta={s.eta_proj}, scheme={s.beta_schedule_scheme.value})")
        return np.array([[float(obj)]]), grad_obj, cons, grad_cons

    initialDensity = 0.5
    for name, _, bound in to_params.Constraints:
        if name == "VolumeFraction":
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

    # ============================ RUN MMA ============================
    nVariables = num_design_var
    nConstraints = len(to_params.Constraints)

    tStart = time.time()
    _last_zeta = zeta0.flatten().copy()
    try:
        optResults = runMMA(
            nVariables,
            nConstraints,
            MMTO_optimization_function,
            zeta0.reshape(-1, 1),
            lowerBound,
            upperBound,
            maxIterations=to_params.MaxIterations,
            timeLimitSecs=s.timeLimit,
            move_limit=MMA_MOVE_LIMIT,
            kktTol=MMA_KKT_TOL,
            fTolerance=s.rel_conv_tol,
            gTolerance=s.rel_conv_tol,
            verbose=False,
        )
        zetaOptimal = np.asarray(optResults[0]).flatten()
    except _StopRequested:                     # stopped from outside: the last design evaluated
        zetaOptimal = _last_zeta
        _during_run["stopped_at_iter"] = int(mmaIterations)
        print(f"Stopped at iteration {mmaIterations}; saving the design reached so far.")
    tEnd = time.time()
    print(f"Total optimization time: {tEnd - tStart:.2f} seconds")
    # ---- assemble the full parameter manifest (no plotting; saved as YAML+pkl) ----
    _run_args = dict(_args_at_start, saveNet=saveNet)   # as given, with the VAE file resolved
    _iters = None
    try:
        _iters = max((len(v) for v in history.values() if hasattr(v, "__len__")),
                     default=None)
    except Exception:
        pass
    run_config = _save.build_run_config(
        example_name=getattr(to_problem, "name", str(to_problem)),
        run_args=_run_args, to_params=to_params, vae_params=vae_params,
        beta_schedule=beta_schedule, mesh=mesh_structural, bcs=bcs, fields=physics.FIELDS,
        mat_encoder=matEncoder, z_real_points=zRealPoints,
        material_colors=material_colors, material_labels=material_labels,
        mma_settings={"optimizer": "MMA (mmaWrapper)", "kktTol": MMA_KKT_TOL,
                      "move_limit": MMA_MOVE_LIMIT,
                      "fTolerance": s.rel_conv_tol, "gTolerance": s.rel_conv_tol},
        runtime_sec=(tEnd - tStart), iterations=_iters,
        comment=getattr(to_params, "Comment", ""))
    run_config["during_run"] = dict(_during_run)
    run_config["network_errors_percent"] = network_errors   # what the run's materials were off by, per property
    run_config["network_reconstruction"] = matEncoder.reconstruction()   # and per material: database vs network
    run_config["provenance"] = _provenance
    run_config["units"] = to_problem.unit_record()
    from lsr_mmto.pyto_ad_imports import topopt_material_model
    run_config["fixed_in_code"] = {
        "mma_move_limit": MMA_MOVE_LIMIT, "mma_kkt_tol": MMA_KKT_TOL,
        "apply_latent_sensitivity_filter": APPLY_LATENT_SENS_FILTER,
        "initial_density": float(initialDensity),
        "simp_structural_penalty_at_end": float(topopt_material_model._SIMP_STRUCTURAL_PENALTY),
        "stress_relaxation_exponent": float(topopt_material_model._SIMP_STRESS_RELAXATION),
        "void_stiffness_relative": float(topopt_material_model._EVOID_RELATIVE),
    }

    run_postprocessing(
        history=history,
        zetaOptimal=zetaOptimal,
        fe_solver=fe_solver_structural,
        matEncoder=matEncoder,
        mat_lib=mat_lib,
        MaterialModel=MaterialModel,
        physical_density=lambda x: ad_model.physical_density(x, beta_proj, s.eta_proj,
                                                             to_params.HeavisideProjection,
                                                             to_params.APPLY_FILTER_TO_DENSITY),
        zRealPoints=zRealPoints,
        material_colors=material_colors,
        material_labels=material_labels,
        num_elems=num_elems,
        latentDim=latentDim,
        elem_volume=elem_volume,
        design_domain_volume=design_domain_volume,
        binarize_topology=s.binarize_topology,
        snap_to_real_material=s.snap_to_real_material,
        save_dir=save_dir,
        run_config=run_config,
        example_name=getattr(to_problem, "name", str(to_problem)),
        physics=physics,
        properties_of=lambda z: decode(matEncoder, z),
        settings=s,
    )

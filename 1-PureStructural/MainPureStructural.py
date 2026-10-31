"""
Structural MMTO driver: density topology + VAE-decoded materials, MMA optimizer.

PIPELINE:
    zeta = [x ; z]                         design vector (MMA variables)
      x  -> x_filt = (H_density @ x)/Hs    density filter
      x_filt -> x_phys = Heaviside(x_filt) optional projection (density only)
      z  -> decode (VAE) -> E, Y, mass_density  per-element material properties
    FEA solve(x_phys) -> von Mises stress (relaxed) -> objective/constraints
    all of the above is one torch tree (AutodiffPureStructural.MMTOModelAD);
    gradients by autograd, through PyTO-AD's solve
    optional penalties (latent z-smoothing, distance-to-real-material) added here
    gradient carried back through Heaviside + density filter by autograd -> w.r.t. x
    MMA takes one step.

Latents z are raw MMA variables (no forward filter); their *gradients* are
optionally smoothed by _smooth_latent_gradient_inplace, gated by
APPLY_LATENT_SENS_FILTER (note: plain (H@g)/Hs smoothing, not Sigmund's
density-weighted form).
"""
import numpy as np
import torch
import time
import os
import sys

from enum import Enum

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../0-Common")))

from materialColors import material_colors 
from PyTOImports import (
    deflation,
    linear_solvers,
    hex_structural_fea,
    hex_element_stiffness,
    createFilters,
    mat_lib,
    MaterialModel,
    TO_QOI,
    runMMA,
    initialize_SIMP_STRUCTURAL_PENALTY,
    initialize_SIMP_THERMAL_PENALTY,
    increment_SIMP_STRUCTURAL_PENALTY,
    increment_SIMP_THERMAL_PENALTY,
)
from ExamplesPureStructural import (
    MMTOExamplesPureStructural,
    getMMTOProblemPureStructural,
    material_colors,
)
from materialEncoder import MaterialEncoder 
import AutodiffPureStructural as AD
from mmto_utility_functions import (
    create_density_and_material_filters,
    compute_domain_volume,
    init_run_history,
    start_run,
    log_design_state,
    record_fea_diagnostics,
    log_objective_and_constraints,
    run_postprocessing,
)
class Z0InitMethod(Enum):
    LIGHTEST = "lightest"
    HEAVIEST = "heaviest"
    ORIGIN = "origin"
    UNIFORM = "uniform"
class BetaScheduleScheme(Enum):
    PAPER = "paper"
    PAPER_BM1 = "paper_bm1"
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

def run_topopt(
    to_problem,
    # ---- per-run sweep overrides (None => use the example's values) ----
    n_dof_desired=None,
    relative_filter_radius=None,
    material_filter_radius=None,
    pnorm_exponent=None,
    timeLimit=10 * 60 * 60,
    saveNet=None,
    plot_progress=True,
    verbose=True,
    save_dir=None,
    # -------------------------------
    # VAE parameters
    # -------------------------------
    use_pretrained_vae=False,
    # -------------------------------
    # Misc. parameters
    # -------------------------------
    use_penalization=False,
    use_continuation=True,
    use_constant_poissons_ratio: bool = True,
    constant_poissons_ratio: float = 0.3,
    snap_to_real_material=True,
    # -------------------------------
    # Optimization parameters
    # -------------------------------    
    rel_conv_tol=1e-7,
    maxIterations=200,
    binarize_topology=True,
    z0_init_method=Z0InitMethod.ORIGIN,
    # -------------------------------
    # Plotting/diagnostic parameters
    # -------------------------------
    plot_bcs=False,            # <-- ADD: when True, show loads/BCs once before optimization
    # -------------------------------
    # Distance penalty parameters
    # -------------------------------
    gamma_init=1e-3,
    gamma_max=0.05,
    gamma_factor=1.05,
    penalty_start_iter=10,
    distance_penalty_disable_after_iter=100,
    use_dynamic_gamma_scheduling=False,
    plotter=None,
    # -------------------------------
    # Heaviside projection parameters
    # -------------------------------
    use_heaviside_projection=True,
    eta_proj=0.5,
    beta_schedule_scheme=BetaScheduleScheme.PAPER,
    # -------------------------------
    # z - H z / Hs smoothing penalty
    # -------------------------------
    use_z_smoothing=False,
    z_smoothing_weight=1e-3,
    z_smoothing_disable_after_iter=50,
):
    """
    DENSITY FILTER + HEAVISIDE PROJECTION (density ONLY):
      x (MMA var) --density filter--> x_filt = (H@x)/Hs --projection--> x_phys = H_{beta,eta}(x_filt)

    Latents:
      z are raw MMA variables; no direct filtering of z is performed.

    Optional latent smoothing penalty (density-weighted):
      Pz = lambda * sum_ell sum_e x_e * ( z_e^(ell) - ((H z^(ell))/Hs)_e )^2
    """
    _args_at_start = dict(locals())   # the arguments as given; the run switches some off later

    history = init_run_history()
    mesh_structural, mat_prop_struct, bc_struct, elem_body_force, to_params, vae_params = (
        getMMTOProblemPureStructural(to_problem, nDOFDesired=n_dof_desired)
    )

    # ---- per-run sweep overrides (None => keep the example's value) ---------
     # Must be set AFTER the example
    # (which assigns these) and BEFORE the filters are built at line ~285.
    # NOTE: MaterialFilterRadius=None makes r_z SILENTLY FOLLOW r_rho
    if relative_filter_radius is not None:
        to_params.RelativeFilterRadius = relative_filter_radius
    if material_filter_radius is not None:
        to_params.MaterialFilterRadius = material_filter_radius
    if pnorm_exponent is not None:
        from PyTOImports import topopt_material_model  # type: ignore
        topopt_material_model._PNORM_EXPONENT = float(pnorm_exponent)
    print(f"[SWEEP] nDOFDesired={to_params.nDOFDesired}  "
          f"RelativeFilterRadius={to_params.RelativeFilterRadius}  "
          f"MaterialFilterRadius={to_params.MaterialFilterRadius}  "
          f"P={pnorm_exponent if pnorm_exponent is not None else '(example default)'}")
    # ------------------------------------------------------------------------

    if to_params.MaterialsExcelFile is None:
        print("Please provide a valid MaterialsExcelFile in to_params.")
        return

    matEncoder = MaterialEncoder(vae_params)
    matEncoder.readExcel(to_params.MaterialsExcelFile)

    if saveNet is None:
        base, _ = os.path.splitext(to_params.MaterialsExcelFile)
        saveNet = base + ".nt"

    vae_file_exists = os.path.exists(saveNet) and os.path.getsize(saveNet) > 0

    if use_pretrained_vae and vae_file_exists:
        print(f"Loading pre-trained autoencoder from file: {saveNet}")
        matEncoder.loadAutoencoderFromFile(saveNet)
    else:
        print(f"Training autoencoder and saving to: {saveNet}")
        time_start = time.time()
        matEncoder.trainAutoencoder(
            vae_params.numEpochs,
            vae_params.klFactor,
            saveNet,
            vae_params.learningRate,
            vae_params.maxAttributeErrorPercent,
        )
        time_end = time.time()
        print(f"Autoencoder training time: {time_end - time_start:.2f} seconds")

    with torch.no_grad():
        matEncoder.training_latents = matEncoder.vaeNet.encoder(
            matEncoder.scaledMaterialData
        ).cpu()
    matEncoder.printEncodingErrors()

    zRealPoints = matEncoder.training_latents
    zRealPoints_np = zRealPoints.detach().cpu().numpy().astype(float)


    # --- Solver selection: direct (PARDISO) for small problems; iterative
    #     (DPCG + deflation) beyond the DOF cutoff. Mirrors topopt_mma.py so the
    #     MMTO driver uses the same tested iterative path as single-material.
    #     The adjoint solve in PyTO-AD's differentiable solve reuses fe_solver.solver
    #     and fe_solver.dsolver, so it follows this choice automatically.
    try:
        from PyTOImports import DIRECT_SOLVER_DOF_CUTOFF
    except Exception:
        DIRECT_SOLVER_DOF_CUTOFF = 120000  # dof limit for direct solver; above -> DPCG

    solver = linear_solvers.Solvers.PARDISO
    dsolver = deflation.DeflationSolver()
    if (to_params.nDOFDesired > DIRECT_SOLVER_DOF_CUTOFF):
        solver = linear_solvers.Solvers.DPCG
        nGroups = dsolver.getRecommendedNumberOfGroups(nDOF=3 * mesh_structural.num_nodes)
        dsolver.create_deflation_groups(mesh_structural, nGroups)
        dsolver.create_deflation_matrix(mesh_structural)
    print("Solver:", solver.name)

    fe_solver_structural = AD.MultiMaterialHexFEA(
        mesh=mesh_structural,
        mat_prop=mat_prop_struct,
        bc=bc_struct,
        solver=solver,
        dsolver=dsolver,
        rtol=1e-8,
        elem_body_force=elem_body_force,
    )

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
    ad_model = AD.MMTOModelAD(
        fe_solver_structural, matEncoder, H_density, Hs_density, H_material, Hs_material,
        zRealPoints_np, to_params, use_constant_poissons_ratio, constant_poissons_ratio, elem_volume,
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
    APPLY_DENSITY_FILTER = True
    APPLY_LATENT_SENS_FILTER = False  # (H_material@g)/Hs smoothing of latent
                                      # gradient blocks; set False.
    MMA_MOVE_LIMIT, MMA_KKT_TOL = 0.05, 1e-6
    # dynamic gamma: the ramp starts once J changed by less than rel_tol over the last
    # window iterations (not before min_iter), or end_margin iterations before it must
    DYN_GAMMA = {"window": 10, "rel_tol": 1e-3, "min_iter": 20, "end_margin": 30}

    mmaIterations = 0
    obj0 = None
    Sd0 = None    # frozen reference scale for distance penalty (own initial value)
    _J_hist = []               # objective history for dynamic-gamma plateau trigger
    _dyn_ramp_started = False   # latches True once the dynamic ramp begins
    _during_run = {}            # iteration at which a setting changed during the run
    gamma = float(gamma_init)
    beta_schedule = get_beta_schedule(beta_schedule_scheme)
    beta_proj = float(beta_schedule["beta_init"])

    if use_continuation:
        initialize_SIMP_STRUCTURAL_PENALTY(1.5)
        initialize_SIMP_THERMAL_PENALTY(1)
    else:
        initialize_SIMP_STRUCTURAL_PENALTY(3)
        initialize_SIMP_THERMAL_PENALTY(1)

    def heaviside_projection(x_filt: np.ndarray, beta: float, eta: float):
        """Smoothed Heaviside projection x_phys(x_filt) and derivative dx_phys/dx_filt.

        x_phys = (tanh(b*eta) + tanh(b*(x_filt-eta))) / (tanh(b*eta) + tanh(b*(1-eta))).
        Larger beta -> sharper (more 0/1) projection.
        """
        x_filt = np.asarray(x_filt, dtype=float).reshape(-1)
        denom = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        x = (np.tanh(beta * eta) + np.tanh(beta * (x_filt - eta))) / denom
        dxphys_dxfilt = (beta * (1.0 - np.tanh(beta * (x_filt - eta)) ** 2)) / denom
        return x, dxphys_dxfilt

    def _smooth_latent_gradient_inplace(vec: np.ndarray):
        """Sensitivity-filter (in place) each latent block of an objective gradient.

        Applies (H_material @ g)/Hs_material to each per-latent-dim slice. This is a
        Sigmund-style regularizer on the latent gradient (z itself is not filtered).
        """
        for ell in range(latentDim):
            sl = slice(
                num_elems + ell * num_elems,
                num_elems + (ell + 1) * num_elems,
            )
            vec[sl] = (H_material @ vec[sl]) / Hs_material

    def _smooth_latent_constraint_gradient_inplace(mat: np.ndarray):
        """Same latent-block sensitivity filter as above, for the constraint Jacobian."""
        nCons = mat.shape[0]
        for i in range(nCons):
            for ell in range(latentDim):
                sl = slice(
                    num_elems + ell * num_elems,
                    num_elems + (ell + 1) * num_elems,
                )
                mat[i, sl] = (H_material @ mat[i, sl]) / Hs_material

    def MMTO_optimization_function(zeta):
        """MMA callback: returns (obj, grad_obj, cons, grad_cons) for design zeta.

        Steps:
          1. split zeta into density x and latents z
          2. density filter x -> x_filt, Heaviside x_filt -> x_phys (numpy, for logging)
          3. AD model: decode z -> material properties and solve, in one torch tree
          4. postprocess the solve; record stress-violation diagnostics into history
          5. obj/cons values from the AD model; their gradients w.r.t. (x_phys, z) by autograd
          6. normalize obj by obj0; add optional z-smoothing and distance penalties (autograd)
          7. latent-block sens filter, then Heaviside + density filter (autograd)
          8. log, update beta continuation / SIMP continuation
        """
        nonlocal mmaIterations, obj0, Sd0, gamma, beta_proj, use_z_smoothing, use_penalization, _dyn_ramp_started
        if (
            use_z_smoothing
            and z_smoothing_disable_after_iter is not None
            and mmaIterations >= z_smoothing_disable_after_iter
        ):
            use_z_smoothing = False
            _during_run["z_smoothing_off_at_iter"] = int(mmaIterations)
            if verbose:
                print(f"[INFO] z-smoothing penalty disabled at iteration {mmaIterations}")
        # Disable distance-based penalty after specified iteration
        if (
            use_penalization
            and not use_dynamic_gamma_scheduling
            and distance_penalty_disable_after_iter is not None
            and mmaIterations >= distance_penalty_disable_after_iter
        ):
            use_penalization = False
            _during_run["distance_penalty_off_at_iter"] = int(mmaIterations)
            if verbose:
                print(f"[INFO] Distance-based penalty disabled at iteration {mmaIterations}")
        zeta = np.asarray(zeta).flatten()

        # ===================== SPLIT DESIGN VECTOR =====================
        x = zeta[:num_elems].copy()
        z = zeta[num_elems:].reshape(latentDim, -1).T   # numpy, element-major (N, latentDim)

        # ============== DENSITY FILTER + HEAVISIDE PROJECTION ==========
        x_filt = (H_density @ x) / Hs_density if APPLY_DENSITY_FILTER else x.copy()

        if use_heaviside_projection:
            x_phys, _ = heaviside_projection(x_filt, beta_proj, eta_proj)
        else:
            x_phys = x_filt

        # ============ DECODE + MAIN FEA: one torch tree (AD model) ============
        # zeta -> x_phys, decoded properties -> PyTO-AD solve -> stress -> J, constraints
        to_params.current_major_iter = int(mmaIterations)
        J_t, cons_t = ad_model.evaluate(zeta, beta_proj, eta_proj,
                                        use_heaviside=use_heaviside_projection,
                                        apply_density_filter=APPLY_DENSITY_FILTER)

        # the decoded properties for logging and diagnostics: the same numbers, detached
        material_properties = {k: v.detach() for k, v in ad_model.props.items()}
        zTorch = ad_model.z.detach()
        log_design_state(history, mmaIterations, x_phys, z, material_properties,
                         zTorch, matEncoder, num_elems, beta_proj, eta_proj, verbose=verbose)

        fe_solver_structural.mesh.setPseudoDensity(torch.as_tensor(x_phys))   # PyTO-AD's mesh takes torch
        fe_solver_structural.postprocess()

        record_fea_diagnostics(history, mmaIterations, x_phys, z, material_properties,
                               fe_solver_structural, elem_volume, design_domain_volume,
                               plot_progress=plot_progress, plotter=plotter, verbose=verbose)

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
                    _smooth_latent_gradient_inplace(grad_obj_phys)
                    _smooth_latent_constraint_gradient_inplace(grad_cons)

        if use_z_smoothing:
            # RAW scaling, applied consistently every iteration. The z-smoothing
            # penalty is ZERO at a spatially-uniform start, so it has no valid
            # own-initial-value to normalize by (dividing by it would blow up).
            # z_smoothing_weight multiplies the raw roughness directly.
            Pz_t = AD.latent_smoothing(ad_model.x_phys, ad_model.z, ad_model.Hm, ad_model.Hsm)
            Pzn = float(z_smoothing_weight) * float(Pz_t.detach())
            obj = obj + Pzn
            grad_obj_phys += float(z_smoothing_weight) * ad_model.grad_phys(Pz_t)
        else:
            Pzn = 0.0

        P = 0.0
        Pn = 0.0

        if use_penalization and (use_dynamic_gamma_scheduling or mmaIterations >= int(penalty_start_iter)):
            # RAW penalty (gamma = 1); weighting + normalization applied here
            Dp_t = AD.distance_penalty(ad_model.x_phys, ad_model.z, ad_model.z_real)
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

            if use_dynamic_gamma_scheduling:
                _J_hist.append(float(Jn))
                if not _dyn_ramp_started:
                    _K, _rel_tol, _min_iter = DYN_GAMMA["window"], DYN_GAMMA["rel_tol"], DYN_GAMMA["min_iter"]
                    _n_ramp = int(np.ceil(np.log(max(gamma_max / gamma_init, 1 + 1e-9))
                                          / np.log(max(gamma_factor, 1 + 1e-9))))
                    _deadline = int(maxIterations) - _n_ramp - DYN_GAMMA["end_margin"]
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
                    gamma = min(gamma * gamma_factor, gamma_max)
            else:
                gamma = min(gamma * gamma_factor, gamma_max)

        grad_obj = grad_obj_phys

        # ============== CHAIN RULE -> DESIGN VARIABLES ================
        grad_obj = ad_model.chain_to_x(grad_obj)
        for i in range(grad_cons.shape[0]):
            grad_cons[i, :] = ad_model.chain_to_x(grad_cons[i, :])

        grad_obj = np.array([grad_obj]).reshape((num_design_var, 1))
        cons = np.array(cons).reshape((-1, 1))
        grad_cons = np.array(grad_cons).reshape((len(cons), num_design_var))

        log_objective_and_constraints(history, obj, J_phys, Jn, P, Pn, Pzn, cons,
                                      to_params, verbose=verbose)

        # ================ CONTINUATION + BETA UPDATE ==================
        mmaIterations += 1

        if use_continuation and (mmaIterations % 10 == 0):
            increment_SIMP_THERMAL_PENALTY(0.25)
            increment_SIMP_STRUCTURAL_PENALTY(0.25)
        if use_heaviside_projection:
            beta_new, beta_updated = update_beta(
                beta_proj=beta_proj,
                mmaIterations=mmaIterations,
                beta_schedule=beta_schedule,
            )
            if beta_updated:
                beta_proj = beta_new
                if verbose:
                    print(f"[PROJ] Updated beta -> {beta_proj:.4g} (eta={eta_proj}, scheme={beta_schedule_scheme.value})")
        return np.array([[float(obj)]]), grad_obj, cons, grad_cons

    initialDensity = 0.5
    for constraint in to_params.Constraints:
        if constraint[0] == TO_QOI.VOLUME_FRACTION and not (
            isinstance(constraint[1], dict) and ("material_id" in constraint[1])
        ):
            initialDensity = constraint[2]
            break

    d0 = initialDensity * np.ones(num_elems, dtype=float)
    z0 = np.zeros(latentDim * num_elems, dtype=float)

    if z0_init_method == Z0InitMethod.LIGHTEST:
        zLightest = matEncoder.getLightestMaterial()
        for i in range(latentDim):
            z0[i * num_elems : (i + 1) * num_elems] = zLightest[i]

    elif z0_init_method == Z0InitMethod.HEAVIEST:
        zHeaviest = matEncoder.getHeaviestMaterial()
        for i in range(latentDim):
            z0[i * num_elems : (i + 1) * num_elems] = zHeaviest[i]

    elif z0_init_method == Z0InitMethod.ORIGIN:
        z0[:] = 0.0

    elif z0_init_method == Z0InitMethod.UNIFORM:
        for i in range(latentDim):
            z0[i * num_elems : (i + 1) * num_elems] = np.random.uniform(
                -0.5, 0.5, size=num_elems
            )
    else:
        raise ValueError(f"Unknown z0_init_method: {z0_init_method}")

    zeta0 = np.concatenate((d0, z0), axis=0).reshape(-1, 1)

    lowerBound = np.zeros(num_design_var, dtype=float).reshape(-1, 1)
    upperBound = np.ones(num_design_var, dtype=float).reshape(-1, 1)

    lowerBound[num_elems:num_design_var] = np.min(zRealPoints.detach().cpu().numpy())
    upperBound[num_elems:num_design_var] = np.max(zRealPoints.detach().cpu().numpy())

    # ============================ RUN MMA ============================
    nVariables = num_design_var
    nConstraints = len(to_params.Constraints)

    tStart = time.time()
    optResults = runMMA(
        nVariables,
        nConstraints,
        MMTO_optimization_function,
        zeta0.reshape(-1, 1),
        lowerBound,
        upperBound,
        maxIterations=maxIterations,
        timeLimitSecs=timeLimit,
        move_limit=MMA_MOVE_LIMIT,
        kktTol=MMA_KKT_TOL,
        fTolerance=rel_conv_tol,
        gTolerance=rel_conv_tol,
        verbose=False,
    )
    zetaOptimal = np.asarray(optResults[0]).flatten()
    tEnd = time.time()
    print(f"Total optimization time: {tEnd - tStart:.2f} seconds")
    # ---- assemble the full parameter manifest (no plotting; saved as YAML+pkl) ----
    import run_io as _run_io
    _run_args = dict(_args_at_start, saveNet=saveNet)   # as given, with the VAE file resolved
    _iters = None
    try:
        _iters = max((len(v) for v in history.values() if hasattr(v, "__len__")),
                     default=None)
    except Exception:
        pass
    run_config = _run_io.build_run_config(
        example_name=getattr(to_problem, "name", str(to_problem)),
        run_args=_run_args, to_params=to_params, vae_params=vae_params,
        beta_schedule=beta_schedule, mesh=mesh_structural, bc=bc_struct,
        mat_encoder=matEncoder, z_real_points=zRealPoints,
        material_colors=material_colors,
        mma_settings={"optimizer": "MMA (mmaWrapper)", "kktTol": MMA_KKT_TOL,
                      "move_limit": MMA_MOVE_LIMIT,
                      "fTolerance": rel_conv_tol, "gTolerance": rel_conv_tol},
        runtime_sec=(tEnd - tStart), iterations=_iters,
        comment=getattr(to_params, "Comment", ""))
    run_config["during_run"] = dict(_during_run)
    from PyTOImports import topopt_material_model  # type: ignore
    run_config["fixed_in_code"] = {
        "mma_move_limit": MMA_MOVE_LIMIT, "mma_kkt_tol": MMA_KKT_TOL,
        "dynamic_gamma": dict(DYN_GAMMA),
        "apply_density_filter": APPLY_DENSITY_FILTER,
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
        heaviside_projection=heaviside_projection,
        H_density=H_density,
        Hs_density=Hs_density,
        zRealPoints=zRealPoints,
        material_colors=material_colors,
        num_elems=num_elems,
        latentDim=latentDim,
        beta_proj=beta_proj,
        eta_proj=eta_proj,
        elem_volume=elem_volume,
        design_domain_volume=design_domain_volume,
        apply_density_filter=APPLY_DENSITY_FILTER,
        use_heaviside_projection=use_heaviside_projection,
        binarize_topology=binarize_topology,
        snap_to_real_material=snap_to_real_material,
        save_dir=save_dir,
        run_config=run_config,
        example_name=getattr(to_problem, "name", str(to_problem)),
    )

if __name__ == "__main__":
    # L-bracket benchmark, the r_rho = r_z = 3.5 cell.
    # Expect Phi_p 2.5377, grey 2.47%, VF 0.1472 / 0.1472 / 0.0993.
    save_dir = start_run(name="LBracket_BM1_rel3p5_rz3p5")
    to_problem = MMTOExamplesPureStructural.LBracketTopLoad_Stress_BM1
    run_topopt(
        to_problem=to_problem,
        save_dir=save_dir,

        # grid and filters
        relative_filter_radius=3.5,
        material_filter_radius=3.5,          # pin it: None makes r_z follow r_rho
        maxIterations=300,
        rel_conv_tol=1e-7,
        timeLimit=36000,

        # VAE
        use_pretrained_vae=True,
        saveNet="./1-PureStructural/MaterialDataPureStructural/BM1_3Mats_2Attr.nt",
        z0_init_method=Z0InitMethod.ORIGIN,
        snap_to_real_material=False,

        # material model
        use_constant_poissons_ratio=False,
        constant_poissons_ratio=0.3,
        use_continuation=False,

        # distance penalty
        use_penalization=True,
        gamma_init=1e-3,
        gamma_factor=1.1,
        gamma_max=10,
        penalty_start_iter=0,
        use_dynamic_gamma_scheduling=True,
        distance_penalty_disable_after_iter=250,   # inert while dynamic scheduling is on

        # latent smoothing
        use_z_smoothing=True,
        z_smoothing_weight=1e-4,
        z_smoothing_disable_after_iter=250,

        # Heaviside projection
        use_heaviside_projection=True,
        eta_proj=0.5,
        beta_schedule_scheme=BetaScheduleScheme.PAPER_BM1,   # beta_max 16

        # output
        binarize_topology=True,
        plot_progress=False,
        plot_bcs=False,
        verbose=True,
    )
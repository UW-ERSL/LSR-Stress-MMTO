"""
The optimization loop: density topology + VAE-decoded materials, for any physics on the list in
physics/available_physics.py.

    run(p) -> _run_topopt(...) = optimize(setup(...))

setup(...)          builds everything before the first design: the problem, the material network, the
                    solver, the filters, the autodiff model (optimize/design_model.py), the start design
                    and the bounds; returns the RunState
evaluate(state, zeta)
                    one design through ITERATION_STEPS, in order (each a function step(state, it)):
                    stop_if_asked, analyse_design, objective_and_constraints,
                    latent_sensitivity_filter, add_penalties, chain_to_design_variables, record,
                    carry_quantity_states, continuation; returns the objective (scaled by its first value,
                    with the penalties), its gradient, the constraints and their gradients, with respect to
                    the optimizer's variables
optimize(state)     the constraint method runs from the start design, asking evaluate() about each design;
                    then the run record and the final analysis (optimize/final_design_analysis.py)

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
    linear_solvers,
    hex_element_stiffness,
    createFilters,
    mat_lib,
    MaterialModel,
    runMMA,
)
from lsr_mmto.materials.encoder import MaterialEncoder, NetworkRunCancelled, network_warning
from lsr_mmto.physics.available_physics import physics_module
from lsr_mmto.design.latent_penalties import (PENALTIES, smooth_latent_constraint_gradient_inplace,
                                              smooth_latent_gradient_inplace)
from lsr_mmto.optimize.design_model import MMTOModelAD
from lsr_mmto.problem.settings import Z0InitMethod
from lsr_mmto.optimize.filters import create_density_and_material_filters
from lsr_mmto.formulation.quantities import find_quantity
from lsr_mmto.formulation.quantity_states import run_state, state_key
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
    from lsr_mmto.materials.material_source import MaterialSource
    from lsr_mmto.optimize.iterative_solve_check import CheckedDeflationSolver
    from lsr_mmto.optimize.mmto_to_params import MMTOParams
    from lsr_mmto.problem.project import Project
    from lsr_mmto.problem.settings import RunSettings
BETA_KEYS = ("beta_init", "beta_max", "beta_factor", "beta_update_every", "beta_start_iter")


def get_beta_schedule(s: "RunSettings") -> dict[str, float]:
    """The Heaviside sharpness schedule, as the run's settings give it: {key: value} of BETA_KEYS. beta starts at
    beta_init and, after iteration beta_start_iter, grows beta_factor times every beta_update_every iterations,
    up to beta_max (update_beta). ValueError, in the settings' own words, for numbers it cannot run with."""
    b = {k: getattr(s, k) for k in BETA_KEYS}
    wrong = [text for bad, text in (
        (b["beta_init"] <= 0, "Starting sharpness must be above 0"),
        (b["beta_max"] < b["beta_init"], "Largest sharpness must be at least Starting sharpness"),
        (b["beta_factor"] < 1, "Sharpness growth must be at least 1"),
        (b["beta_update_every"] < 1, "Grow every must be at least 1"),
        (b["beta_start_iter"] < 0, "Start growing after iteration must be at least 0")) if bad]
    if wrong:
        raise ValueError("; ".join(wrong))
    return b


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
    project: "Project"
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
    checked: "CheckedDeflationSolver"             # its deflation solver: whether each iterative solve converged
    encoder: "MaterialSource"                     # the run's material source (materials/material_source.py)
    network_errors: dict[str, float]              # {attribute: % of its range}
    database_record: dict[str, Any]               # network_stamp.database_stamp: the database as the run read it
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
    simp_penalties: dict[str, float]              # each SIMP exponent of the physics (SIMP_PENALTIES), as now
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


SIMP_STEP, SIMP_EVERY = 0.25, 10              # SIMP continuation: raised by this, every this many iterations


def simp_penalties_at_start(physics: "ModuleType", continuation: bool) -> dict[str, float]:
    """{name: value}: each SIMP exponent of the physics (its SIMP_PENALTIES) at the start of a run, the start
    with SIMP continuation when it is on."""
    return {name: float(with_continuation if continuation else start)
            for name, (start, with_continuation, _) in physics.SIMP_PENALTIES.items()}


def raised_simp_penalties(physics: "ModuleType", values: dict[str, float]) -> dict[str, float]:
    """One step of SIMP continuation: each exponent raised by SIMP_STEP, up to its largest (SIMP_PENALTIES)."""
    return {name: min(values[name] + SIMP_STEP, float(largest))
            for name, (_, _, largest) in physics.SIMP_PENALTIES.items()}


def problem_solver(physics: "ModuleType", mesh: "HexMesher", mat_prop: "Material", bcs: "dict[str, BC]",
                   elem_body_force: np.ndarray | None, to_params: "MMTOParams",
                   dsolver: "CheckedDeflationSolver | None" = None) -> Any:
    """The physics' solver of the problem (its make_solver), with the linear solver chosen by size.

    Direct (PARDISO) for small problems; iterative (DPCG + deflation) beyond the DOF cutoff, for a
    physics that builds a deflation space (its deflation_space); a physics without one keeps the
    direct solver at every size. Mirrors topopt_mma.py so the MMTO driver uses the same tested
    iterative path as single-material. The adjoint solve in PyTO-AD's differentiable solve reuses
    fe_solver.solver and fe_solver.dsolver, so it follows this choice automatically. elem_body_force: one
    per element, or None (Project.build always gives None). dsolver: the deflation solver, which keeps whether
    each solve converged (iterative_solve_check.py); the caller makes it and keeps it, to read those solves
    whatever the physics' solver keeps (a new one when not given). Above the limit, raises ValueError when the
    deflation space leaves a field without a space of its size (iterative_solve_check.py). Prints the solver
    chosen."""
    from lsr_mmto.optimize.iterative_solve_check import CheckedDeflationSolver
    from lsr_mmto.pyto_ad_imports import DIRECT_SOLVER_DOF_CUTOFF    # dof limit for the direct solver; above it, DPCG

    linear_solver = linear_solvers.Solvers.PARDISO
    if dsolver is None:
        dsolver = CheckedDeflationSolver()
    if to_params.nDOFDesired > DIRECT_SOLVER_DOF_CUTOFF and physics.deflation_space is not None:
        linear_solver = linear_solvers.Solvers.DPCG
        physics.deflation_space(dsolver, mesh)
        for field, per_node in physics.FIELDS.items():     # a field no space fits would be solved unchecked
            n = per_node * mesh.num_nodes
            W = getattr(dsolver.space_for(n), "W", None)
            if W is None or W.shape[0] != n:
                raise ValueError(f"physics {physics.NAME!r}: its deflation_space gives no space for the field "
                                 f"{field} ({per_node} unknown{'s' if per_node > 1 else ''} per node), so its solve would not be checked")
    print("Solver:", linear_solver.name)
    return physics.make_solver(mesh, mat_prop, bcs, linear_solver, dsolver, elem_body_force)


def material_source(project: "Project", s: "RunSettings", vae_params: "VAEParams", xlsx: str,
                    saveNet: str | None, should_stop: Callable[[], bool] | None,
                    on_network_error: Callable[[str], bool] | None) \
        -> "tuple[MaterialSource, torch.Tensor, dict[str, float], str]":
    """The run's material source (materials/material_source.py), the one place a run's materials come from, as
    the project says (project.material_source, project.pinned_material), and what the run reads with it: the
    database's materials as latent points (materials, latent dimension), the network's largest error per
    property in % ({} with no network), and saveNet resolved (None is the .nt beside the database).

    One database material: DatabaseMaterials, the pinned material (the database's only one when none is pinned)
    for every element, at its database values; no network is read or trained. Otherwise the material network of the database xlsx: read when there is one and
    s.use_pretrained_vae, else trained and saved; a network further off than s.network_error_warning asks
    on_network_error whether to go on (NetworkRunCancelled when it says no). With a pinned material, the
    network pinned to it (PinnedMaterial); else the network itself."""
    from lsr_mmto.materials.material_source import DatabaseMaterials, PinnedMaterial
    if project.material_source == "one_database_material":
        from lsr_mmto.materials.databases import database_table
        names = [str(n) for n in database_table(xlsx)[0]]
        materials = DatabaseMaterials(xlsx, 0 if project.pinned_material is None else
                                      names.index(str(project.pinned_material)))
        print(f"Materials: {names[int(materials.rows)]}, at its database values in every element; no network")
        return materials, torch.zeros((len(materials.materialNames), 0)), {}, saveNet
    matEncoder = MaterialEncoder(vae_params)
    matEncoder.readExcel(xlsx)

    if saveNet is None:
        base, _ = os.path.splitext(xlsx)
        saveNet = base + ".nt"

    vae_file_exists = os.path.exists(saveNet) and os.path.getsize(saveNet) > 0

    if s.use_pretrained_vae and vae_file_exists:
        print(f"Loading pre-trained autoencoder from file: {saveNet}")
        from lsr_mmto.materials.network_stamp import read_network
        trained_on = read_network(saveNet)[1]                   # the project's check compared it; again here, with
        if trained_on is None or trained_on.get("main") != matEncoder.database_record["main"]:   # what was read
            raise ValueError(f"{os.path.basename(xlsx)} is not the database {os.path.basename(saveNet)} was trained "
                             f"on: it changed since, or the network has no record of it; check the project again")
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

    if project.pinned_material is not None:
        row = [str(n) for n in matEncoder.materialNames].index(str(project.pinned_material))
        print(f"Materials: {project.pinned_material} in every element, as the network gives it")
        return (PinnedMaterial(matEncoder, row), torch.zeros((len(matEncoder.materialNames), 0)), network_errors,
                saveNet)
    return matEncoder, matEncoder.training_latents, network_errors, saveNet


def setup(*, s: "RunSettings", project: "Project", save_dir: str, verbose: bool, plot_progress: bool,
          plot_bcs: bool, plotter: "pyvista.Plotter | None", progress: Callable[[dict[str, Any]], None] | None,
          should_stop: Callable[[], bool] | None, on_network_error: Callable[[str], bool] | None) -> RunState | None:
    """
    Everything before the first design is evaluated: the problem built, the material network read or
    trained, the solver, the filters, the autodiff model, the start design and the bounds. Returns the
    RunState (None when the project names no material database).

    DENSITY FILTER + HEAVISIDE PROJECTION (density ONLY):
      x (MMA var) --density filter--> x_filt = (H@x)/Hs --projection--> x_phys = H_{beta,eta}(x_filt),
      then x_phys = 1 at the elements to keep solid

    Latents:
      z are raw MMA variables; no direct filtering of z is performed.

    The penalties added to the objective: PENALTIES (design/latent_penalties.py), those of the run's
    physics, in order; e.g. latent smoothing (density-weighted):
      Pz = lambda * sum_ell sum_e x_e * ( z_e^(ell) - ((H z^(ell))/Hs)_e )^2
    """
    # the record keeps the project under to_problem, the key the runs already saved have
    _args_at_start = dict(s.as_dict(), to_problem=project, save_dir=save_dir, verbose=verbose,
                          plot_progress=plot_progress, plot_bcs=plot_bcs, plotter=plotter)   # what the run records
    saveNet = s.saveNet                                     # resolved below: None is the .nt beside the database
    from lsr_mmto.optimize import save_results as _save
    _provenance = _save.provenance()   # the code as it is when the run starts

    history = init_run_history()
    if not hasattr(project, "build"):
        raise TypeError(f"project must be a Project, e.g. Project.load('projects/L-bracket.mmto'), "
                        f"not {project!r}; the coded examples are now the project files in projects/")
    physics = physics_module(project.physics)            # the front page of the project's physics
    mesh, mat_prop, bcs, elem_body_force, to_params, vae_params = project.build()
    print(f"[SWEEP] nDOFDesired={to_params.nDOFDesired}  "
          f"RelativeFilterRadius={to_params.RelativeFilterRadius}  "
          f"MaterialFilterRadius={to_params.MaterialFilterRadius}")

    if to_params.MaterialsExcelFile is None:
        print("Please provide a valid MaterialsExcelFile in to_params.")
        return None

    material_colors, material_labels = material_display(to_params.MaterialsExcelFile)
    matEncoder, zRealPoints, network_errors, saveNet = material_source(
        project, s, vae_params, to_params.MaterialsExcelFile, saveNet, should_stop, on_network_error)
    from lsr_mmto.materials.network_stamp import database_stamp
    database_record = getattr(getattr(matEncoder, "network", matEncoder), "database_record", None) or \
        database_stamp(to_params.MaterialsExcelFile)        # every cell, as the network read them, for the record
    zRealPoints_np = zRealPoints.detach().cpu().numpy().astype(float)

    from lsr_mmto.optimize.iterative_solve_check import CheckedDeflationSolver
    checked = CheckedDeflationSolver()                # keeps whether each iterative solve converged
    solver = problem_solver(physics, mesh, mat_prop, bcs, elem_body_force, to_params, checked)

    num_dof = solver.bc.num_dofs
    print(f"Number of DOF: {num_dof}")

    num_elems = mesh.num_elems
    latentDim = matEncoder.latent_dim
    num_design_var = num_elems + num_elems * latentDim
    print(f"Using latent dimension: {latentDim}")
    # the penalties of the run's physics; with no latent points, those that read them are left out
    penalties = [p for p in PENALTIES if p.applies_to(physics.NAME) and (latentDim or not p.reads_latent_points)]
    left_out = [p.shown for p in PENALTIES if p.applies_to(physics.NAME) and p not in penalties]
    if left_out:
        print(f"[INFO] No latent points: left out {', '.join(left_out)}")
    for key in (k for p in penalties for k in p.keys()):    # each penalty's history keys, new ones
        if key in history:
            raise ValueError(f"penalty history key {key!r} is used twice")
        history[key] = []
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

    beta_schedule = get_beta_schedule(s)

    simp_penalties = simp_penalties_at_start(physics, s.use_continuation)
    physics.set_simp_penalties(solver, simp_penalties)

    initialDensity = 0.5                       # or the bound of the first constraint that sets it
    for c in to_params.Constraints:
        if find_quantity(c[0]).sets_start_density:
            initialDensity = c[2]
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

    lowerBound[num_elems:num_design_var] = np.min(zRealPoints_np, initial=np.inf)    # with no latent points
    upperBound[num_elems:num_design_var] = np.max(zRealPoints_np, initial=-np.inf)   # the slices are empty

    return RunState(
        # what setup was given
        s=s, project=project, save_dir=save_dir, verbose=verbose, plot_progress=plot_progress,
        plotter=plotter, progress=progress, should_stop=should_stop,
        args_at_start=_args_at_start, provenance=_provenance, save_net=saveNet,
        # what it built
        physics=physics, mesh=mesh, bcs=bcs, to_params=to_params, vae_params=vae_params, solver=solver, checked=checked,
        encoder=matEncoder, network_errors=network_errors, database_record=database_record, model=ad_model, history=history,
        material_colors=material_colors, material_labels=material_labels,
        z_real=zRealPoints, num_elems=num_elems, latent_dim=latentDim, num_design_var=num_design_var,
        elem_volume=elem_volume, design_domain_volume=design_domain_volume,
        H_material=H_material, Hs_material=Hs_material,
        beta_schedule=beta_schedule, initial_density=initialDensity,
        zeta0=zeta0, lower=lowerBound, upper=upperBound,
        # what changes from one iteration to the next
        iteration=0,                        # designs evaluated so far
        beta=float(beta_schedule["beta_init"]),
        simp_penalties=simp_penalties,      # raised by SIMP continuation
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
    """The next iteration's count, SIMP penalties and Heaviside beta.

    Reads state.iteration, state.beta, state.simp_penalties. Sets state.iteration (one more), state.beta (when
    the Heaviside projection is on and the schedule raises it), state.simp_penalties (raised_simp_penalties,
    then handed to the physics' set_simp_penalties) whenever the new count is a multiple of 10 and continuation
    is on."""
    s = state.s
    state.iteration += 1
    if s.use_continuation and (state.iteration % SIMP_EVERY == 0):
        state.simp_penalties = raised_simp_penalties(state.physics, state.simp_penalties)
        state.physics.set_simp_penalties(state.solver, state.simp_penalties)
    if state.to_params.HeavisideProjection:
        beta_new, beta_updated = update_beta(
            beta_proj=state.beta,
            mmaIterations=state.iteration,
            beta_schedule=state.beta_schedule,
        )
        if beta_updated:
            state.beta = beta_new
            if state.verbose:
                print(f"[PROJ] Updated beta -> {state.beta:.4g} (eta={s.eta_proj}, at most {s.beta_max:g})")


ITERATION_STEPS = [
    stop_if_asked,
    analyse_design,
    objective_and_constraints,
    latent_sensitivity_filter,
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

    Reads state.iteration, state.last_zeta, state.beta, state.during_run, state.simp_penalties. Sets state.during_run
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
    project = state.project
    run_config = _save.build_run_config(
        example_name=getattr(project, "name", str(project)),
        run_args=_run_args, to_params=to_params, vae_params=state.vae_params,
        beta_schedule=state.beta_schedule, mesh=state.mesh, bcs=state.bcs, fields=state.physics.FIELDS,
        mat_encoder=state.encoder, z_real_points=state.z_real,
        material_colors=state.material_colors, material_labels=state.material_labels,
        mma_settings=method_settings,
        runtime_sec=(tEnd - tStart), iterations=_iters,
        comment=getattr(to_params, "Comment", ""),
        pnorm_exponent=float(s.pnorm_exponent) if getattr(s, "pnorm_exponent", None) is not None else None)
    run_config["during_run"] = dict(state.during_run)
    run_config["network_errors_percent"] = state.network_errors   # what the run's materials were off by, per property
    run_config["network_reconstruction"] = state.encoder.reconstruction()   # and per material: database vs network
    run_config["provenance"] = state.provenance
    run_config["database"] = state.database_record                # every cell of the database the run used
    run_config["units"] = project.unit_record()
    from lsr_mmto.pyto_ad_imports import topopt_material_model
    run_config["fixed_in_code"] = {
        "mma_move_limit": MMA_MOVE_LIMIT, "mma_kkt_tol": MMA_KKT_TOL,
        "apply_latent_sensitivity_filter": APPLY_LATENT_SENS_FILTER,
        "initial_density": float(state.initial_density),
        **{f"simp_{name}_penalty_at_end": float(v) for name, v in state.simp_penalties.items()},
        "stress_relaxation_exponent": float(topopt_material_model._SIMP_STRESS_RELAXATION),
        "void_stiffness_relative": float(topopt_material_model._EVOID_RELATIVE),
    }

    model, matEncoder = state.model, state.encoder
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

"""
run_setup.py
============
Everything a run builds before its first design is evaluated, and what it keeps from then to its end:

setup(...)          the problem built, the material network read or trained (material_source), the solver
                    (problem_solver), the filters, the autodiff model (optimize/design_model.py), the start
                    design, the bounds and the schedules (optimize/schedules.py); returns the RunState
RunState            what setup built, and what changes from one iteration to the next

The loop (optimize/optimization_loop.py) runs from the RunState. Analyse in the GUI (optimize/solid_part_analysis.py)
builds its solver with problem_solver too, as a run does.
"""
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
import time
import os
import shutil

from lsr_mmto.materials.databases import in_shipped_folder, material_display
from lsr_mmto.pyto_ad_imports import linear_solvers, createFilters
from lsr_mmto.materials.encoder import MaterialEncoder, NetworkRunCancelled, network_warning
from lsr_mmto.physics.available_physics import physics_module
from lsr_mmto.design.latent_penalties import PENALTIES
from lsr_mmto.optimize.design_model import MMTOModelAD
from lsr_mmto.problem.settings import Z0InitMethod
from lsr_mmto.optimize.filters import create_density_and_material_filters
from lsr_mmto.formulation.quantities import find_quantity
from lsr_mmto.optimize.history import compute_domain_volume, init_run_history
from lsr_mmto.optimize.schedules import sharpness_schedule, simp_schedules

if TYPE_CHECKING:
    from types import ModuleType

    import pyvista

    from pyto.core.bc import BC
    from pyto.core.hex_mesher import HexMesher
    from pyto.core.mat_lib import Material
    from lsr_mmto.design.latent_penalties import Penalty
    from lsr_mmto.materials.databases import VAEParams
    from lsr_mmto.materials.material_source import MaterialSource
    from lsr_mmto.optimize.iterative_solve_check import CheckedDeflationSolver
    from lsr_mmto.optimize.mmto_to_params import MMTOParams
    from lsr_mmto.optimize.schedules import Schedule
    from lsr_mmto.problem.project import Project
    from lsr_mmto.problem.settings import RunSettings


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
    beta_schedule: "Schedule"                     # the Heaviside sharpness's (optimize/schedules.py)
    simp_schedules: "dict[str, Schedule]"         # each SIMP exponent's, by name
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
    trained, the solver, the filters, the autodiff model, the start design, the bounds and the schedules
    (optimize/schedules.py). Returns the
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

    beta_schedule = sharpness_schedule(s)

    simp_schedule = simp_schedules(physics, s.use_continuation)
    simp_penalties = {name: schedule.start for name, schedule in simp_schedule.items()}
    physics.set_simp_penalties(solver, simp_penalties)

    initialDensity = 0.5                       # or the bound of the first constraint that sets it
    for c in to_params.Constraints:
        if find_quantity(c[0]).sets_start_density:
            initialDensity = c[2]
            break
    if s.initial_density is not None:          # the Start density setting, when given, wins
        initialDensity = float(s.initial_density)
        if not 0.0 < initialDensity <= 1.0:
            raise ValueError(f"Start density must be above 0 and at most 1, not {initialDensity:g}")
    if not 0.0 < float(s.mma_move_limit) <= 1.0:
        raise ValueError(f"Step limit must be above 0 and at most 1, not {float(s.mma_move_limit):g}")

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
        beta_schedule=beta_schedule, simp_schedules=simp_schedule, initial_density=initialDensity,
        zeta0=zeta0, lower=lowerBound, upper=upperBound,
        # what changes from one iteration to the next
        iteration=0,                        # designs evaluated so far
        beta=float(beta_schedule.start),
        simp_penalties=simp_penalties,      # raised by SIMP continuation
        objective_scale=None,               # the first objective's size; later ones are divided by it
        penalties=penalties,                # the penalties of the run's physics, in the order they are added
        penalty_scales={},                  # the first value of each penalty scaled by it, by name
        gamma=float(s.gamma_init),          # the distance penalty's weight
        objective_history=[],               # scaled objectives, for the start of the weight's ramp
        gamma_ramp_started=False,
        use_z_smoothing=s.use_z_smoothing, use_distance_penalty=s.use_penalization,   # switched off on schedule
        during_run={},                      # iteration at which a setting changed during the run
        last_zeta=zeta0.flatten().copy(),
    )

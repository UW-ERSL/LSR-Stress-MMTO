"""
solid_part_analysis.py
======================
Analyse, the button beside Run: the project's physics solved once on the solid part, every element
full (x_phys = 1), in one material of its database, with that material's values as the database gives
them, in its own units (not through the material network). No optimizer, no iterations: it checks the
physics and the setup before a run.

Analyse this design (Results): a saved run's design as built, solved once the same way under the
project's supports and loads as they are now: each element with a density above 0.5 solid, in its
nearest database material (the run's material_indices, as Export STL takes them), the others empty.

    analyse_solid_part(project, material, save_dir=None, should_stop=None) -> Run (results/load_results.py)
    analyse_design(project, run_folder, save_dir=None, should_stop=None) -> Run

It takes from the physics only its front page's entries, as a run does:
    make_solver      the solver, built as a run builds it (run_setup.problem_solver)
    analyse          the solve the project's objective and constraints (QUANTITIES) read; their values
                     are printed as the final check prints its numbers
    final_summary    the final check of the design analysed, printed, and its results (RESULT_FIELDS)
When its solves are iterative, the record keeps whether they converged ("solver_check"), as a run's does
(iterative_solve_check.py), and the Summary says so when one did not.

Saved (save_dir), it is a results folder like a run's, its record marked as an analysis ("analysis" in
run_data.config.json; for a design, the run it analysed too). Not saved, nothing is written: the record is
made in memory and shown from there.
"""
import datetime
import json
import os
import time
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

if TYPE_CHECKING:
    from types import ModuleType

    from lsr_mmto.materials.material_source import DatabaseMaterials

    from pyto.core.hex_mesher import HexMesher
    from lsr_mmto.optimize.mmto_to_params import MMTOParams
    from lsr_mmto.problem.project import Project
    from lsr_mmto.problem.settings import RunSettings
    from lsr_mmto.results.load_results import Run

_DTYPE = torch.float64
NOT_GIVEN = "not given by an analysis: it follows the material network"   # a quantity of the latent points


class AnalysisStopped(Exception):
    """Stop was pressed before the analysis was done; nothing was saved."""


class NoLatentPoints(Exception):
    """A quantity read the latent points (z, z_real), which an analysis does not have: it uses no network."""


class _Analysed:
    """What the physics' analyse and the quantities read of an iteration's model (optimize/design_model.py),
    for the design analysed: x_phys its densities (1 solid, 0 empty), props each element's material's database
    values. It has no latent points: a quantity that reads them raises NoLatentPoints."""

    fe: Any                              # the physics' solver
    physics: "ModuleType"
    enc: "DatabaseMaterials"
    settings: "RunSettings"
    N: int
    x_phys: torch.Tensor                 # (N,), 1 solid and 0 empty
    props: dict[str, torch.Tensor]
    elem_volume: float

    def __init__(self, solver: Any, physics: "ModuleType", materials: "DatabaseMaterials", settings: "RunSettings",
                 elem_volume: float, x: np.ndarray) -> None:
        self.fe, self.physics, self.enc, self.settings = solver, physics, materials, settings
        self.N = int(solver.mesh.num_elems)
        self.x_phys = torch.as_tensor(np.asarray(x, dtype=float), dtype=_DTYPE)
        self.props = materials.properties(self.N)
        self.elem_volume = float(elem_volume)

    @property
    def z(self) -> torch.Tensor:
        """Raises NoLatentPoints: an analysis has none (z_real too)."""
        raise NoLatentPoints

    z_real = z


def _quantities(m: _Analysed, to_params: "MMTOParams",
                labels: list[str]) -> list[tuple[str, float | None, float | None, str]]:
    """[(name as the app shows it, value or None, limit or None, op)] of the objective and then each constraint,
    at the solid part; None for a quantity that reads the latent points."""
    from lsr_mmto.formulation.quantities import find_quantity, op_of, quantity_label
    out = []
    for spec in [tuple(to_params.Objective)] + [tuple(c) for c in to_params.Constraints]:
        name, opt, limit = spec[0], spec[1], (float(spec[2]) if len(spec) > 2 else None)
        try:
            with torch.no_grad():
                value = float(find_quantity(name).fn(m, opt))
        except NoLatentPoints:
            value = None
        out.append((quantity_label(name, opt, labels), value, limit, op_of(spec)))
    return out


def _print_quantities(title: str, values: list[tuple[str, float | None, float | None, str]]) -> None:
    """The objective and the constraints, as the final check prints its numbers."""
    print("\n" + "=" * 70)
    print(f"[ANALYSIS] {title}: the objective and the constraints")
    for name, value, limit, op in values:
        bound = "" if limit is None else f" ({'limit' if op == '<=' else 'at least'} {limit:.6g})"
        print(f" {name}{bound} : " + (NOT_GIVEN if value is None else f"{value:.6g}"))
    print("=" * 70 + "\n")


def analyse_solid_part(project: "Project", material: int, save_dir: str | None = None,
                       should_stop: Callable[[], bool] | None = None) -> "Run":
    """The project's physics solved once on the solid part in one material of its database (material: its
    row), with the project's supports, loads, mesh and settings, checked as a run is checked (validate's
    errors refuse it). The record of the analysis, as a Run: written into save_dir and read back when it is
    given, else made in memory with nothing written. should_stop() True between its steps ends it with
    AnalysisStopped."""
    from lsr_mmto.materials.databases import database_table, material_display

    def design(mesh: "HexMesher", xlsx: str) -> tuple[np.ndarray, np.ndarray, str, str, dict[str, Any]]:
        names = database_table(xlsx)[0]
        if isinstance(material, bool) or not isinstance(material, (int, np.integer)) or \
                not 0 <= int(material) < len(names):
            raise ValueError(f"material {material!r}: a row of the database, 0 to {len(names) - 1}")
        row, label = int(material), material_display(xlsx)[1][int(material)]
        n = int(mesh.num_elems)
        return np.ones(n), np.full(n, row), xlsx, f"Solid part in {label}", {"material": label, "material_row": row}
    return _analyse(project, design, save_dir, should_stop)


def _same_part(a: "Project", b: "Project") -> bool:
    """Whether two projects have the same part: the same shapes for parts made of shapes, the same STL file
    content otherwise."""
    import filecmp
    from lsr_mmto.problem.shapes import plain
    if a.part_shapes is not None or b.part_shapes is not None:
        return a.part_shapes is not None and b.part_shapes is not None and plain(a.part_shapes) == plain(b.part_shapes)
    return os.path.isfile(a.stl_path()) and os.path.isfile(b.stl_path()) and \
        filecmp.cmp(a.stl_path(), b.stl_path(), shallow=False)


def analyse_design(project: "Project", run_folder: str, save_dir: str | None = None,
                   should_stop: Callable[[], bool] | None = None) -> "Run":
    """Analyses a saved run's design as built, solved once under the project's supports and loads as they are
    now, with its settings, checked as a run is checked. Each element with a density above 0.5 is solid, in its
    nearest material of the database the run used (the run's material_indices, as Export STL takes them), with
    that material's values as the database gives them; the others are empty. The run's mesh must be the
    project's now: the same mesh dofs, single element layer and part, else a ValueError says what differs; and
    the database the run used must be as the run read it (its record of every cell, kept by runs that have one;
    else its material names), else a ValueError names the cells changed.
    Returns the analysis as a Run (load), its record saying which run it analysed (Run.analysed_design).

    project: the project, a Project, with the supports and loads to analyse the design under
    run_folder: the run's folder (run returns it), one of those in results in the project's folder
    save_dir: the folder the analysis goes into, kept with the runs; None saves nothing
    should_stop: a function asked between its steps; when it returns True the analysis stops and saves
        nothing; or None
    Example: a = analyse_design(p, folder, save_dir=folder + " design")
    """
    from lsr_mmto.problem.project import Project
    from lsr_mmto.problem.project_folder import run_facts
    from lsr_mmto.materials.databases import database_table
    from lsr_mmto.results import load_results
    run_folder = os.path.abspath(run_folder)
    run = load_results.load(os.path.join(run_folder, "run_data"))
    spec = ((run.config or {}).get("arguments_resolved") or {}).get("to_problem")
    if not isinstance(spec, dict) or "project" not in spec:
        raise ValueError("this run's record does not hold its project, so its part and materials are not known")
    ran = Project.from_record(spec)
    if run.analysed_material is not None:
        raise ValueError("this is an analysis, not a run: analyse the run it came from")
    differ = [what for what, same in (("mesh dofs", ran.n_dof == project.n_dof),
                                      ("single element layer", ran.single_layer == project.single_layer),
                                      ("part", _same_part(ran, project)))
              if not same]
    if differ:
        raise ValueError(f"the run was made on another mesh than the project's now ({', '.join(differ)} changed "
                         f"since): set them back as the run had them to analyse its design")
    when = run_facts(run_folder)[0]

    def design(mesh: "HexMesher", xlsx: str) -> tuple[np.ndarray, np.ndarray, str, str, dict[str, Any]]:
        if int(mesh.num_elems) != int(run.n_elems):
            raise ValueError(f"the run was made on another mesh than the project's now: {run.n_elems} elements, "
                             f"the project's mesh {int(mesh.num_elems)}")
        names = [str(n).strip() for n in run.material_names]
        used = ran.database_paths()[0]
        if not os.path.isfile(used) and project.database is not None:
            used = project.database_paths()[0]          # the same database renamed since (Rename... in Materials)
        if not os.path.isfile(used) or database_table(used)[0] != names:
            raise ValueError(f"the database the run used ({os.path.basename(used)}) is not there as it was, so its "
                             f"materials are not known")
        then = (run.config or {}).get("database")
        if isinstance(then, dict) and isinstance(then.get("main"), list):      # every cell, as the run read them
            from lsr_mmto.materials.network_stamp import changed_cells, database_stamp, listed
            cells = changed_cells(then["main"], database_stamp(used)["main"])
            if cells:
                raise ValueError(f"the database the run used ({os.path.basename(used)}) has changed since the run: "
                                 f"{listed(cells)}; bring back the version the run used (Versions... in Materials) "
                                 f"to analyse its design")
        x = (np.asarray(run.x_cont, dtype=float) > 0.5).astype(float)
        return (x, np.asarray(run.material_indices, dtype=int), used, f"Design of {when}",
                {"material": f"design of {when}", "design_of": os.path.basename(run_folder)})
    return _analyse(project, design, save_dir, should_stop)


def _analyse(project: "Project",
             design: "Callable[[HexMesher, str], tuple[np.ndarray, np.ndarray, str, str, dict[str, Any]]]",
             save_dir: str | None,
             should_stop: Callable[[], bool] | None) -> "Run":
    """The project's physics solved once on a design: design(mesh, the project's database) gives each element's
    density (1 or 0), its material's row in the database, that database, the design's title, and the record's
    "analysis" entry. See analyse_solid_part."""
    from lsr_mmto.formulation.quantities import constraint_g
    from lsr_mmto.materials.databases import material_display
    from lsr_mmto.materials.material_source import DatabaseMaterials
    from lsr_mmto.optimize import save_results
    from lsr_mmto.optimize.final_design_analysis import checked_results, saved_names, solver_check
    from lsr_mmto.optimize.iterative_solve_check import CheckedDeflationSolver
    from lsr_mmto.optimize.history import compute_domain_volume
    from lsr_mmto.optimize.run_setup import problem_solver
    from lsr_mmto.physics.available_physics import physics_module
    from lsr_mmto.problem.project import _ready_to_run
    from lsr_mmto.problem.settings import run_settings
    from lsr_mmto.results import load_results

    def stop_if_asked() -> None:
        if should_stop is not None and should_stop():
            raise AnalysisStopped("the analysis was stopped; nothing was saved")

    started = time.time()
    project = _ready_to_run(project, network=False)      # as a run, but no network: it uses the database's values
    s = run_settings(project)
    physics = physics_module(project.physics)
    mesh, mat_prop, bcs, elem_body_force, to_params, _ = project.build(network=False)
    x, rows, xlsx, title, analysis = design(mesh, to_params.MaterialsExcelFile)
    materials = DatabaseMaterials(xlsx, rows)
    colours, labels = material_display(xlsx)
    stop_if_asked()

    checked = CheckedDeflationSolver()              # keeps whether each iterative solve converged
    checked.step = title                            # every solve of an analysis is of its one design
    solver = problem_solver(physics, mesh, mat_prop, bcs, elem_body_force, to_params, checked)
    elem_volume = float(np.prod(solver.mesh.elem_size))
    domain_volume = compute_domain_volume(solver)
    m = _Analysed(solver, physics, materials, s, elem_volume, x)
    own: dict[str, list[float]] = {}                 # the physics' own numbers (its Summary rows), recorded as
    with torch.no_grad():                            # one iteration of a run records them
        physics.analyse(m)
        physics.iteration_diagnostics(own, 0, m, np.asarray(x, dtype=float),
                                      {k: v.detach() for k, v in m.props.items()}, elem_volume, domain_volume,
                                      verbose=False)
    values = _quantities(m, to_params, labels)
    _print_quantities(title, values)
    stop_if_asked()

    z = np.zeros((m.N, materials.latent_dim))        # no latent points: each element is its own material
    results = physics.final_summary(
        [(title, x, z)], solver, materials.decode, s, materials, materials.latent_dim, elem_volume, domain_volume)
    checked_results(physics, results, m.N, int(mesh.num_nodes))
    check = solver_check(checked, [title], saved_names(physics))
    stop_if_asked()

    nan = float("nan")
    history = {**own, "J_phys": [nan if values[0][1] is None else values[0][1]],
               "constraints": [[nan if v is None else constraint_g(v, op, limit) for _, v, limit, op in values[1:]]]}
    config = {
        "run": {"example": project.name, "comment": project.comment or project.name,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "runtime_seconds": time.time() - started, "iterations": None},
        "analysis": analysis,
        "arguments_resolved": {"to_problem": save_results._to_plain(project)},
        "topopt_params": save_results._params_dict(to_params),
        "materials": {"material_names": list(materials.materialNames), "material_count": len(materials.materialNames),
                      "material_colors": list(colours), "material_labels": list(labels),
                      "real_material_properties": {a: materials.rawData[:, c["idx"]].astype(float).tolist()
                                                   for a, c in materials.materialAttributes.items()}},
        "fea": {"mesh": save_results._mesh_summary(mesh),
                "loading_and_bc": {f: save_results._loading_bc_summary(mesh, bc, physics.FIELDS[f])
                                   for f, bc in bcs.items()}},
        "units": project.unit_record(),
        "provenance": save_results.provenance(),
    }
    if check is not None:
        config["solver_check"] = check
    data = save_results.run_data(
        config=config,
        fields={"x_cont": x, "x_bin": x, "z_elem": np.zeros((m.N, 0)), "material_indices": np.asarray(rows)},
        vae_state_dict=None, history=history, example_name=project.name,
        z_real_points=np.zeros((len(materials.materialNames), 0)), material_colors=colours,
        material_names=materials.materialNames, latent_dim=0, num_elems=m.N, num_nodes=int(mesh.num_nodes),
        material_labels=labels,
        results={name: dict(r, value=np.asarray(results[name])) for name, r in physics.RESULT_FIELDS.items()})
    if save_dir is not None and save_results.WRITE_FILES:
        os.makedirs(save_dir, exist_ok=True)
        save_results.write_run_data(save_dir, data)
        print(f"[SAVE] {', '.join(save_results.files_written())} -> {save_dir}")
        return load_results.load(os.path.join(save_dir, "run_data"))
    arrays, manifest = save_results.record_arrays(save_dir or "analysis", data)
    return load_results.from_arrays("", arrays, json.loads(json.dumps(manifest, default=str)))

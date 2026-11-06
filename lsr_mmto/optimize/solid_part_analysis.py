"""
solid_part_analysis.py
======================
Analyse, the button beside Run: the project's physics solved once on the solid part, every element
full (x_phys = 1), in one material of its database, with that material's values as the database gives
them, in its own units (not through the material network). No optimizer, no iterations: it checks the
physics and the setup before a run.

    analyse_solid_part(project, material, save_dir=None, should_stop=None) -> Run (results/load_results.py)

It takes from the physics only its front page's entries, as a run does:
    make_solver      the solver, built as a run builds it (optimization_loop.problem_solver)
    analyse          the solve the project's objective and constraints (QUANTITIES) read; their values
                     are printed as the final check prints its numbers
    final_summary    the final check of the solid part, printed, and its per-element results (RESULT_FIELDS)

Saved (save_dir), it is a results folder like a run's, its record marked as an analysis ("analysis" in
run_data.config.json). Not saved, nothing is written: the record is made in memory and shown from there.
"""
import datetime
import json
import os
import time
import types

import numpy as np
import torch

_DTYPE = torch.float64
NOT_GIVEN = "not given by an analysis: it follows the material network"   # a quantity of the latent points


class AnalysisStopped(Exception):
    """Stop was pressed before the analysis was done; nothing was saved."""


class NoLatentPoints(Exception):
    """A quantity read the latent points (z, z_real), which an analysis does not have: it uses no network."""


class _OneMaterial:
    """The database's materials as the quantities and the physics' final_summary read them from the material
    encoder (materialNames, rawData, materialAttributes), with every latent point decoded to the one material
    analysed: decoder and getMaterialProperties give every element that material's database values. So a
    quantity of a database material (Mass fraction) reads the database, and the final check's mass is the
    material's own."""

    def __init__(self, xlsx, row):
        from lsr_mmto.materials.databases import database_table
        names, attributes, units, values = database_table(xlsx)
        if isinstance(row, bool) or not isinstance(row, (int, np.integer)) or not 0 <= int(row) < len(names):
            raise ValueError(f"material {row!r}: a row of the database, 0 to {len(names) - 1}")
        self.materialNames, self.rawData, self.row = names, values, int(row)
        self.materialAttributes = {a: {"idx": j, "unit": units[j]} for j, a in enumerate(attributes)}
        self.vaeNet = types.SimpleNamespace(decoder=lambda z: z)

    def properties(self, n):
        """{column: the material's value in each of n elements}, as torch, the way a run's decode gives them."""
        return {a: torch.full((int(n),), float(self.rawData[self.row, c["idx"]]), dtype=_DTYPE)
                for a, c in self.materialAttributes.items()}

    def getMaterialProperties(self, decoded):
        return self.properties(len(decoded))


class _SolidPart:
    """What the physics' analyse and the quantities read of an iteration's model (optimize/design_model.py),
    for the solid part: x_phys 1 in every element, props the material's database values. It has no latent
    points: a quantity that reads them raises NoLatentPoints."""

    def __init__(self, solver, physics, one, settings, elem_volume):
        self.fe, self.physics, self.enc, self.settings = solver, physics, one, settings
        self.N = int(solver.mesh.num_elems)
        self.x_phys = torch.ones(self.N, dtype=_DTYPE)
        self.props = one.properties(self.N)
        self.elem_volume = float(elem_volume)

    @property
    def z(self):
        raise NoLatentPoints

    z_real = z


def _quantities(m, to_params, labels):
    """[(name as the app shows it, value or None, limit or None)] of the objective and then each constraint,
    at the solid part; None for a quantity that reads the latent points."""
    from lsr_mmto.formulation.quantities import find_quantity, quantity_label
    out = []
    for spec in [tuple(to_params.Objective)] + [tuple(c) for c in to_params.Constraints]:
        name, opt, limit = spec[0], spec[1], (float(spec[2]) if len(spec) > 2 else None)
        try:
            with torch.no_grad():
                value = float(find_quantity(name).fn(m, opt))
        except NoLatentPoints:
            value = None
        out.append((quantity_label(name, opt, labels), value, limit))
    return out


def _print_quantities(title, values):
    """The objective and the constraints, as the final check prints its numbers."""
    print("\n" + "=" * 70)
    print(f"[ANALYSIS] {title}: the objective and the constraints")
    for name, value, limit in values:
        bound = "" if limit is None else f" (limit {limit:.6g})"
        print(f" {name}{bound} : " + (NOT_GIVEN if value is None else f"{value:.6g}"))
    print("=" * 70 + "\n")


def analyse_solid_part(project, material, save_dir=None, should_stop=None):
    """The project's physics solved once on the solid part in one material of its database (material: its
    row), with the project's supports, loads, mesh and settings, checked as a run is checked (validate's
    errors refuse it). The record of the analysis, as a Run: written into save_dir and read back when it is
    given, else made in memory with nothing written. should_stop() True between its steps ends it with
    AnalysisStopped."""
    from lsr_mmto.materials.databases import material_display
    from lsr_mmto.optimize import save_results
    from lsr_mmto.optimize.final_design_analysis import checked_results
    from lsr_mmto.optimize.history import compute_domain_volume
    from lsr_mmto.optimize.optimization_loop import problem_solver
    from lsr_mmto.physics.available_physics import physics_module
    from lsr_mmto.problem.project import _prepared
    from lsr_mmto.problem.settings import run_settings
    from lsr_mmto.results import load_results

    def stop_if_asked():
        if should_stop is not None and should_stop():
            raise AnalysisStopped("the analysis was stopped; nothing was saved")

    started = time.time()
    project = _prepared(project)                     # as a run: settings completed, validate's errors refuse
    s = run_settings(project)
    physics = physics_module(project.physics)
    mesh, mat_prop, bcs, elem_body_force, to_params, _ = project.build()
    xlsx = to_params.MaterialsExcelFile
    one = _OneMaterial(xlsx, material)
    colours, labels = material_display(xlsx)
    title = f"Solid part in {labels[one.row]}"
    stop_if_asked()

    solver = problem_solver(physics, mesh, mat_prop, bcs, elem_body_force, to_params)
    elem_volume = float(np.prod(solver.mesh.elem_size))
    m = _SolidPart(solver, physics, one, s, elem_volume)
    with torch.no_grad():
        physics.analyse(m)
    values = _quantities(m, to_params, labels)
    _print_quantities(title, values)
    stop_if_asked()

    x = np.ones(m.N)
    z = np.zeros((m.N, 1))                           # one latent coordinate for final_summary to reshape; the
    results = physics.final_summary(                 # properties ignore it: every element is the one material
        [(title, x, z)], solver, lambda z: one.properties(len(z)), s, one, 1, elem_volume,
        compute_domain_volume(solver))
    checked_results(physics, results, m.N, int(mesh.num_nodes))
    stop_if_asked()

    nan = float("nan")
    history = {"J_phys": [nan if values[0][1] is None else values[0][1]],
               "constraints": [[nan if v is None else v / limit - 1.0 for _, v, limit in values[1:]]]}
    config = {
        "run": {"example": project.name, "comment": project.comment or project.name,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "runtime_seconds": time.time() - started, "iterations": None},
        "analysis": {"material": labels[one.row], "material_row": one.row},
        "arguments_resolved": {"to_problem": save_results._to_plain(project)},
        "topopt_params": save_results._params_dict(to_params),
        "materials": {"material_names": list(one.materialNames), "material_count": len(one.materialNames),
                      "material_colors": list(colours), "material_labels": list(labels),
                      "real_material_properties": {a: one.rawData[:, c["idx"]].astype(float).tolist()
                                                   for a, c in one.materialAttributes.items()}},
        "fea": {"mesh": save_results._mesh_summary(mesh),
                "loading_and_bc": {f: save_results._loading_bc_summary(mesh, bc, physics.FIELDS[f])
                                   for f, bc in bcs.items()}},
        "units": project.unit_record(),
        "provenance": save_results.provenance(),
    }
    data = save_results.run_data(
        config=config,
        fields={"x_cont": x, "x_bin": x, "z_elem": np.zeros((m.N, 0)),
                "material_indices": np.full(m.N, one.row)},
        vae_state_dict=None, history=history, example_name=project.name,
        z_real_points=np.zeros((len(one.materialNames), 0)), material_colors=colours,
        material_names=one.materialNames, latent_dim=0, num_elems=m.N, num_nodes=int(mesh.num_nodes),
        material_labels=labels,
        results={name: dict(r, value=np.asarray(results[name])) for name, r in physics.RESULT_FIELDS.items()})
    if save_dir is not None and save_results.WRITE_FILES:
        os.makedirs(save_dir, exist_ok=True)
        save_results.write_run_data(save_dir, data)
        print(f"[SAVE] {', '.join(save_results.files_written())} -> {save_dir}")
        return load_results.load(os.path.join(save_dir, "run_data"))
    arrays, manifest = save_results.record_arrays(save_dir or "analysis", data)
    return load_results.from_arrays("", arrays, json.loads(json.dumps(manifest, default=str)))

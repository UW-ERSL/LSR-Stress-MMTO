"""
final_design_analysis.py
========================
The final design analysed three ways (continuous, binarized, binarized with snapped materials) by
the physics (its final_summary), and the run saved for later postprocessing. When the run's solves are
iterative, the record also keeps which of them did not converge (iterative_solve_check.py): a design whose
final-check solve failed still has its numbers printed and saved, marked as not reliable.
"""
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

if TYPE_CHECKING:
    from types import ModuleType

    from lsr_mmto.optimize.iterative_solve_check import CheckedDeflationSolver
    from pyto.autodiff.material_model import MaterialModel
    from lsr_mmto.materials.material_source import MaterialSource
    from lsr_mmto.problem.settings import RunSettings


def checked_results(physics: "ModuleType", results: dict[str, np.ndarray] | None, n_elems: int,
                    n_nodes: int) -> dict[str, np.ndarray] | None:
    """The physics' final_summary results, refused with a clear message when one of its RESULT_FIELDS is
    missing, or its values are not of the length and shape its entry declares (where and shape:
    load_results.result_info). Returns results itself, unchanged."""
    from lsr_mmto.results.load_results import result_info, result_shape
    missing = sorted(set(physics.RESULT_FIELDS) - set(results or {}))
    if missing:
        raise ValueError(f"the {physics.NAME} physics' final_summary returned no {', '.join(missing)}: it returns "
                         f"one value per element for each entry of its RESULT_FIELDS")
    for name, entry in physics.RESULT_FIELDS.items():
        info = result_info(entry)
        want, got = result_shape(info, n_elems, n_nodes), np.shape(results[name])
        if got != want:
            raise ValueError(f"the {physics.NAME} physics' final_summary returned {name} with shape {got}, but its "
                             f"RESULT_FIELDS entry declares a {info['shape']} at each {info['where']}: shape {want}")
    return results


def saved_names(physics: "ModuleType") -> list[str]:
    """The results a run of the physics saves, by the names the Results panel shows them under (RESULT_FIELDS'
    view): "Stress", "Displacement" for structural."""
    return [entry.get("view") or name for name, entry in physics.RESULT_FIELDS.items()]


class _Named(list):
    """The final check's designs, which tell the run's iterative solver (checked) the label of the design being
    solved: the physics' final_summary goes through them with "for label, x, z in designs" and solves each inside
    its pass (template_results.py; tests/code_rules/test_physics_folders.py checks that it reads them only so),
    and a design taken by its index (designs[-1]) is the one then solved. Plain list otherwise."""

    def __init__(self, designs: list, checked: "CheckedDeflationSolver | None") -> None:
        super().__init__(designs)
        self.checked = checked

    def _solving(self, design: tuple) -> tuple:
        if self.checked is not None:
            self.checked.step = design[0]
        return design

    def __iter__(self):
        for design in list.__iter__(self):
            yield self._solving(design)

    def __getitem__(self, i):
        item = list.__getitem__(self, i)
        return item if isinstance(i, slice) else self._solving(item)


def solver_check(checked: "CheckedDeflationSolver | None", labels: list[str], saved: list[str]) \
        -> dict[str, Any] | None:
    """What the run's iterative solves did not reach, for the run record: {"iterations_not_converged": how many
    solves of the iterations did not converge, "final_check_not_converged": {design label: largest relative
    residual}, "results_from": the label of the design whose results the run saves}, printed too when something did not converge; None when no solve was iterative. labels: the
    designs checked, in order; saved: the names of the results the run saves (those of the last design)."""
    from lsr_mmto.optimize.iterative_solve_check import short_label
    if checked is None or not checked.solves:
        return None
    during = sum(1 for step, _, ok in checked.solves if step is None and not ok)
    failed = {}
    for step, residual, ok in checked.solves:
        if step is not None and not ok:
            seen = failed.get(step, 0.0)
            failed[step] = residual if residual != residual or seen != seen else max(seen, residual)    # NaN kept
    if during:
        print(f"[SOLVER CHECK] the iterative solver did not converge in {during} solve(s) during the iterations: "
              f"the design those iterations reached may not be the optimum.")
    if failed:
        which = ", ".join(f"{short_label(k)} (relative residual {v:.3g})" for k, v in failed.items())
        last = labels[-1] in failed
        saved_too = f", nor are the results saved ({', '.join(saved)})" + \
            (", which are those of the last design." if len(labels) > 1 else ".")
        print(f"[SOLVER CHECK] the iterative solver did not converge in the final check for: {which}. Their "
              f"numbers above are not reliable" + (saved_too if last else "."))
    return {"iterations_not_converged": during, "final_check_not_converged": failed, "results_from": labels[-1]}


def run_postprocessing(history: dict[str, list[float | np.ndarray]], zetaOptimal: np.ndarray, fe_solver: Any,
                       matEncoder: "MaterialSource", mat_lib: "ModuleType",
                       MaterialModel: "type[MaterialModel]", physical_density: Callable[[np.ndarray], np.ndarray],
                       zRealPoints: torch.Tensor | np.ndarray, material_colors: list[str], num_elems: int,
                       latentDim: int, elem_volume: float, design_domain_volume: float,
                       binarize_topology: bool, snap_to_real_material: bool, save_dir: str | None = None,
                       run_config: dict[str, Any] | None = None, example_name: str | None = None,
                       material_labels: list[str] | None = None, physics: "ModuleType | None" = None,
                       properties_of: Callable[[torch.Tensor], dict[str, torch.Tensor]] | None = None,
                       settings: "RunSettings | None" = None,
                       checked: "CheckedDeflationSolver | None" = None) -> None:
    """Reconstruct the optimal fields, have the physics analyse the continuous / binarized /
    snapped designs (its final_summary), and save the run.

    properties_of(z) -> properties is the optimization's own decode, and settings its settings, so the
    final check analyses the design with the same materials as the optimization;
    physical_density(x) -> x_phys is its own density filter and projection, at the last beta.
    zetaOptimal is the design vector [x (N) ; z latent-major], flat. mat_lib and MaterialModel are not used.
    checked: the run's deflation solver (run_setup.problem_solver), whose solves are labelled with the
    design they are for and recorded as run_config["solver_check"]; None checks nothing.
    Saves nothing when save_dir is None; the physics and properties_of are required (ValueError)."""
    if physics is None or properties_of is None:
        raise ValueError("run_postprocessing needs the physics and properties_of, the optimization's decode")
    import os
    if save_dir is not None:
        pass  # plot autosave disabled; figures are drawn by results/run_plots.py from the saved record

    def _p(name: str) -> str | None:
        return os.path.join(save_dir, name) if save_dir else None

    # Reconstruct physical fields
    xOptimal_raw = zetaOptimal[0:num_elems].copy()
    zOptimal = zetaOptimal[num_elems:].reshape(latentDim, num_elems).T   # numpy, element-major

    xOptimal_cont = physical_density(xOptimal_raw)   # the optimization's own filter and projection

    xOptimal = xOptimal_cont.copy()
    if binarize_topology:
        x_sorted = np.sort(xOptimal)
        threshold = x_sorted[int((1 - np.mean(xOptimal)) * len(xOptimal))]
        xOptimal = np.where(xOptimal < threshold, 0.0, 1.0)

    zOptimalPts = torch.tensor(zOptimal, dtype=torch.get_default_dtype())
    if snap_to_real_material:
        zSnappedPts = torch.tensor(matEncoder.getClosestRealMaterialZValues(zOptimalPts))
        zOptimalPts = zSnappedPts

    designs = [("Continuous (x_phys projected, raw z)", xOptimal_cont, zOptimal),
               ("Binarized topology (x projected -> binarized, raw z)", xOptimal, zOptimal),
               ("Binarized + snapped materials (x binarized, snapped z)", xOptimal, zOptimalPts)]
    try:
        results = physics.final_summary(_Named(designs, checked), fe_solver, properties_of, settings, matEncoder,
                                        latentDim, elem_volume, design_domain_volume)
    finally:
        if checked is not None:
            checked.step = None
    checked_results(physics, results, num_elems, int(fe_solver.mesh.num_nodes))
    check = solver_check(checked, [label for label, _, _ in designs], saved_names(physics))
    if run_config is not None and check is not None:
        run_config["solver_check"] = check

    material_properties = matEncoder.decode(zOptimalPts)
    # each element's value of every property the physics reads (its PROPERTIES), as property_<name>
    element_properties = {f"property_{name}": material_properties[name].detach().cpu().numpy()
                          for name in physics.PROPERTIES}

    # --- persist everything for offline postprocessing (results/load_results.py, results/run_plots.py) ---
    material_indices = matEncoder.getClosestRealMaterialIndex(zOptimalPts)
    if hasattr(material_indices, "cpu"):
        material_indices = material_indices.cpu().numpy()
    material_names = [matEncoder.materialNames[i]
                      for i in range(len(matEncoder.materialNames))]

    if save_dir is not None:
        from lsr_mmto.optimize import save_results as run_save
        _zreal = (zRealPoints.detach().cpu().numpy()
                  if hasattr(zRealPoints, "detach") else np.asarray(zRealPoints))
        run_save.save_run_data(
            save_dir=save_dir,
            config=run_config if run_config is not None else {},
            fields={
                "zeta": np.asarray(zetaOptimal),
                "x_cont": np.asarray(xOptimal_cont),
                "x_bin": np.asarray(xOptimal),
                "z_elem": np.asarray(zOptimal),
                **{k: np.asarray(v) for k, v in element_properties.items()},
                "material_indices": np.asarray(material_indices),
            },
            results={name: dict(r, value=np.asarray(results[name])) for name, r in physics.RESULT_FIELDS.items()},
            vae_state_dict=matEncoder.network_weights(),
            history=history,
            example_name=example_name,
            z_real_points=_zreal,
            material_colors=material_colors,
            material_names=material_names,
            material_labels=material_labels,
            latent_dim=latentDim,
            num_elems=num_elems,
            num_nodes=int(fe_solver.mesh.num_nodes),
        )
        print(f"[SAVE] {', '.join(run_save.files_written())} -> {save_dir}")

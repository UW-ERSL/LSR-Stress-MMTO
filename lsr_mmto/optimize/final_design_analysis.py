"""
final_design_analysis.py
========================
The final design analysed three ways (continuous, binarized, binarized with snapped materials) by
the physics (its final_summary), and the run saved for later postprocessing.
"""
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

if TYPE_CHECKING:
    from types import ModuleType

    from pyto.autodiff.material_model import MaterialModel
    from lsr_mmto.materials.encoder import MaterialEncoder
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


def run_postprocessing(history: dict[str, list[float | np.ndarray]], zetaOptimal: np.ndarray, fe_solver: Any,
                       matEncoder: "MaterialEncoder", mat_lib: "ModuleType",
                       MaterialModel: "type[MaterialModel]", physical_density: Callable[[np.ndarray], np.ndarray],
                       zRealPoints: torch.Tensor | np.ndarray, material_colors: list[str], num_elems: int,
                       latentDim: int, elem_volume: float, design_domain_volume: float,
                       binarize_topology: bool, snap_to_real_material: bool, save_dir: str | None = None,
                       run_config: dict[str, Any] | None = None, example_name: str | None = None,
                       material_labels: list[str] | None = None, physics: "ModuleType | None" = None,
                       properties_of: Callable[[torch.Tensor], dict[str, torch.Tensor]] | None = None,
                       settings: "RunSettings | None" = None) -> None:
    """Reconstruct the optimal fields, have the physics analyse the continuous / binarized /
    snapped designs (its final_summary), and save the run.

    properties_of(z) -> properties is the optimization's own decode, and settings its settings, so the
    final check analyses the design with the same materials as the optimization;
    physical_density(x) -> x_phys is its own density filter and projection, at the last beta.
    zetaOptimal is the design vector [x (N) ; z latent-major], flat. mat_lib and MaterialModel are not used.
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
    zOptimal = zetaOptimal[num_elems:].reshape(latentDim, -1).T   # numpy, element-major

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

    results = physics.final_summary(
        [("Continuous (x_phys projected, raw z)", xOptimal_cont, zOptimal),
         ("Binarized topology (x projected -> binarized, raw z)", xOptimal, zOptimal),
         ("Binarized + snapped materials (x binarized, snapped z)", xOptimal, zOptimalPts)],
        fe_solver, properties_of, settings, matEncoder, latentDim, elem_volume, design_domain_volume)
    checked_results(physics, results, num_elems, int(fe_solver.mesh.num_nodes))

    decoded = matEncoder.vaeNet.decoder(zOptimalPts)
    material_properties = matEncoder.getMaterialProperties(decoded)
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
            vae_state_dict=matEncoder.vaeNet.state_dict(),
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

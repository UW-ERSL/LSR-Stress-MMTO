"""
thermal_results.py
==================
What the thermal module records and reports: the temperature a run saves (RESULT_FIELDS) and its Summary
row (SUMMARY_ROWS), the highest temperature of each iteration recorded into the history
(iteration_diagnostics), and the final designs' thermal compliance, printed by the final check
(final_summary).
"""
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path
from lsr_mmto.pyto_ad_imports import topopt_material_model as ad_mm
from lsr_mmto.physics.thermal.thermal_physics import CONDUCTIVITY

if TYPE_CHECKING:
    import pyvista

    from lsr_mmto.materials.encoder import MaterialEncoder
    from lsr_mmto.optimize.solid_part_analysis import _Materials
    from lsr_mmto.physics.thermal.thermal_physics import ConductionHexFEA
    from lsr_mmto.problem.settings import RunSettings

# the results a run saves, by name: the view that shows it in the Results panel, its title on the colour scale,
# what it measures (for its unit), and where its values live
RESULT_FIELDS = {
    "temperature": dict(view="Temperature", title="Temperature", measures="temperature", where="node"),
}

# the numbers of its own a run's Summary shows: the last value of each history key, when the run recorded it
SUMMARY_ROWS = {
    "max_temperature": dict(label="Max temperature", unit="{temperature}", format=".4g"),
}


# OPTIONAL: iteration_diagnostics, what each iteration records
def iteration_diagnostics(history: dict[str, list[float | np.ndarray]], iteration: int, solver: "ConductionHexFEA",
                          x_phys: np.ndarray, properties: dict[str, torch.Tensor], elem_volume: float,
                          design_domain_volume: float,
                          plot_progress: bool = False, plotter: "pyvista.Plotter | None" = None,
                          verbose: bool = True) -> None:
    """After the iteration's solve: its highest temperature recorded into history (max_temperature),
    printed when verbose."""
    history.setdefault("max_temperature", [])
    if solver.sol is None:
        return
    highest = float(torch.max(solver.sol.detach()))
    history["max_temperature"].append(highest)
    if verbose:
        print(f"[THERMAL] iteration {iteration}: max temperature {highest:.4g}")


# ESSENTIAL: final_summary, the final check and the temperature a run saves
def final_summary(designs: "list[tuple[str, np.ndarray, np.ndarray | torch.Tensor]]", solver: "ConductionHexFEA",
                  properties_of: Callable[[torch.Tensor], dict[str, torch.Tensor]], settings: "RunSettings",
                  encoder: "MaterialEncoder | _Materials", latent_dim: int, elem_volume: float,
                  design_domain_volume: float) -> dict[str, np.ndarray]:
    """The final check's numbers: each design (label, x, z element-major) solved, its thermal compliance and
    highest temperature printed. properties_of(z) gives the decoded properties of latent points z. Returns
    the results a run saves (RESULT_FIELDS), of the last design: the temperature at each node."""
    for label, x, z in designs:
        with torch.no_grad():
            z = z if torch.is_tensor(z) else torch.as_tensor(np.asarray(z), dtype=torch.get_default_dtype())
            solver.set_conductivity(properties_of(z)[CONDUCTIVITY])
            T = solver.solve(x, ad_mm.MaterialModel.SIMP).detach()
            compliance = float(torch.dot(solver.total_force, T))
        print(f"[THERMAL] {label}: thermal compliance {compliance:.6g}, max temperature {float(T.max()):.4g}")
    return {"temperature": T.numpy()}

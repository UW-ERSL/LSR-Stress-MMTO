"""
template_results.py
===================
What this file is for: what the template physics records and saves. The per-element results a run
saves (RESULT_FIELDS), what each iteration records (iteration_diagnostics), and the numbers of the
final design (final_summary).

Who reads it:
    RESULT_FIELDS          optimize/final_design_analysis.py saves each one; the Results panel shows it
    iteration_diagnostics  optimize/optimization_loop.py, after each iteration's solve
    final_summary          optimize/final_design_analysis.py, once, on the final designs; it returns
                           the RESULT_FIELDS values that are saved

To make your own, change the lines marked FILL.
"""
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  # keep as is
from lsr_mmto.pyto_ad_imports import topopt_material_model as ad_mm

from lsr_mmto.physics.template.template_physics import COEFFICIENT

if TYPE_CHECKING:
    import pyvista

    from lsr_mmto.materials.encoder import MaterialEncoder
    from lsr_mmto.optimize.solid_part_analysis import _Materials
    from lsr_mmto.physics.template.template_physics import DiffusionHexFEA
    from lsr_mmto.problem.settings import RunSettings


# ----------------------------------------------------------------------------- results a run saves
# {name: dict(view, title, measures, where, shape)}: view is the Results panel's choice that shows it, title
# the colour scale's title, measures what it is (for its unit, None for no unit). where is "element" (one
# value per element) or "node" (one per node), shape "scalar" (one number) or "vector" (x, y and z); an
# entry that does not say them is a number at each element, as here. The values come from final_summary
# below, which returns one array for each name here, (elements,) or (nodes,) with a last axis of 3 for a
# vector; a name it does not return, or values of another length or shape, stop the save with a message
# that says so. A vector at each node, such as a displacement:
#     "displacement": dict(view="Displacement", title="Displacement", measures="length", where="node",
#                          shape="vector"),
RESULT_FIELDS = {
    "mean_value": dict(view="Value", title="Value of c", measures=None),      # FILL
}


def _element_mean(solver: "DiffusionHexFEA", c: torch.Tensor) -> torch.Tensor:
    """c at each element: the mean of its 8 nodes."""
    return c[solver.edof].mean(1)


# ----------------------------------------------------------------------------- each iteration, and the end
def iteration_diagnostics(history: dict[str, list[float | np.ndarray]], iteration: int, solver: "DiffusionHexFEA",
                          x_phys: np.ndarray, properties: dict[str, torch.Tensor], elem_volume: float,
                          design_domain_volume: float,
                          plot_progress: bool = False, plotter: "pyvista.Plotter | None" = None,
                          verbose: bool = True) -> None:
    """After each iteration's solve. history is the run's record ({name: list}); append to it what you want
    to keep, and print a line when verbose. Here: the largest c of the last solve."""
    if solver.sol is None:
        return
    largest = float(torch.max(solver.sol.detach()))
    history.setdefault("template_max_value", []).append(largest)             # FILL
    if verbose:
        print(f"[TEMPLATE] iteration {iteration}: largest value {largest:.4g}")


def final_summary(designs: "list[tuple[str, np.ndarray, np.ndarray | torch.Tensor]]", solver: "DiffusionHexFEA",
                  properties_of: Callable[[torch.Tensor], dict[str, torch.Tensor]], settings: "RunSettings",
                  encoder: "MaterialEncoder | _Materials", latent_dim: int, elem_volume: float,
                  design_domain_volume: float) -> dict[str, np.ndarray]:
    """Once, at the end. designs is [(label, x, z)] for the continuous, binarized and binarized-and-snapped
    final designs (x per element, z per element and latent dimension); properties_of(z) decodes latent
    points to properties. Solve each, print its numbers, and return the results a run saves,
    {name: numpy array}, one for each RESULT_FIELDS entry, of the last design: one value per element here,
    or as the entry's where and shape say."""
    for label, x, z in designs:
        with torch.no_grad():
            z = z if torch.is_tensor(z) else torch.as_tensor(np.asarray(z), dtype=torch.get_default_dtype())
            props = properties_of(z)
            solver.set_coefficient(props[COEFFICIENT] * float(settings.template_coefficient_scale))
            c = solver.solve(x, ad_mm.MaterialModel.SIMP).detach()
            compliance = float(torch.dot(solver.total_force, c))
        print(f"[TEMPLATE] {label}: diffusion compliance {compliance:.6g}, largest value {float(c.max()):.4g}")
    return {"mean_value": _element_mean(solver, c).numpy()}                # FILL: one per RESULT_FIELDS entry

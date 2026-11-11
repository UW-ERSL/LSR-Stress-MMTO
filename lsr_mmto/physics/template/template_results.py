"""
template_results.py
===================
The job of this file: what the physics records and saves. The results a run saves (RESULT_FIELDS) and the
final check of the final designs (final_summary); optionally what each iteration records
(iteration_diagnostics) and the physics' own rows in a run's Summary (SUMMARY_ROWS). Thermal's
thermal_results.py is this file filled in.

Who reads it:
    RESULT_FIELDS          optimize/final_design_analysis.py saves each one; the Results panel shows it
    final_summary          optimize/final_design_analysis.py, once, on the final designs (and Analyse, on
                           the design analysed); it returns the RESULT_FIELDS values that are saved
    iteration_diagnostics  optimize/optimization_loop.py, after each iteration's solve
    SUMMARY_ROWS           results/tables.py, the Summary and Compare of a run
"""
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

if TYPE_CHECKING:
    from lsr_mmto.materials.encoder import MaterialEncoder
    from lsr_mmto.optimize.solid_part_analysis import _Materials
    from lsr_mmto.problem.settings import RunSettings

# RESULT_FIELDS: the results a run saves, {name: dict(view, title, measures, where, shape)}:
#   view      the Results panel's Show choice that draws it
#   title     the colour scale's title
#   measures  what it is, for its unit: a unit kind, or None for no unit
#   where     "element" (one value per element, the default) or "node"
#   shape     "scalar" (one number, the default) or "vector" (x, y and z)
# final_summary returns one array for each name: (elements,) or (nodes,), with a last axis of 3 for a vector;
# a name it does not return, or values of another length or shape, stop the save with a message saying so.
# Thermal's: {"temperature": dict(view="Temperature", title="Temperature", measures="temperature", where="node")}
RESULT_FIELDS = ...                                                 # FILL


# ESSENTIAL: final_summary, the final check and the results a run saves
def final_summary(designs: "list[tuple[str, np.ndarray, np.ndarray | torch.Tensor]]", solver: Any,
                  properties_of: Callable[[torch.Tensor], dict[str, torch.Tensor]], settings: "RunSettings",
                  encoder: "MaterialEncoder | _Materials", latent_dim: int, elem_volume: float,
                  design_domain_volume: float) -> dict[str, np.ndarray]:
    """Once, at the end. designs is [(label, x, z)] for the final designs (x per element, z per element and
    latent dimension); properties_of(z) decodes latent points to properties. Solve each design with the
    solver, print its numbers, and return the results a run saves, {name: numpy array}, one for each
    RESULT_FIELDS entry, of the last design."""
    raise NotImplementedError("FILL: final_summary, the final check of this physics")


# ----------------------------------------------------------------------------- OPTIONAL entries of this file
# OPTIONAL: SUMMARY_ROWS, the numbers of its own a run's Summary and Compare show: {history key: dict(label,
# unit, format)}, the last value of that key in the run's history, when the run recorded it. unit is text with
# unit kinds in braces, or "" for none. Thermal's:
#   SUMMARY_ROWS = {"max_temperature": dict(label="Max temperature", unit="{temperature}", format=".4g")}

# OPTIONAL: iteration_diagnostics, what each iteration records
# def iteration_diagnostics(history: dict[str, list[float | np.ndarray]], iteration: int, solver: Any,
#                           x_phys: np.ndarray, properties: dict[str, torch.Tensor], elem_volume: float,
#                           design_domain_volume: float, plot_progress: bool = False,
#                           plotter: "pyvista.Plotter | None" = None, verbose: bool = True) -> None:
#     """After each iteration's solve. history is the run's record ({name: list}): add your keys with
#     history.setdefault(key, []) and append to them, and print a line when verbose. Left out, nothing is
#     recorded or printed besides what every run has."""

# ----------------------------------------------------------------------------- Helpers
# Anything final_summary or iteration_diagnostics needs (a stress check, a value per element from one per
# node). Mark each one "# HELPER: of <what uses it>".

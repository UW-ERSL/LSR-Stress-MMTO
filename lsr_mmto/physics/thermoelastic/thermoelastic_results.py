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
from lsr_mmto.pyto_ad_imports import topopt_material_model as ad_mm
from lsr_mmto.physics.thermoelastic.thermoelastic_physics import elastic_constants, thermal_force
if TYPE_CHECKING:
    from lsr_mmto.materials.encoder import MaterialEncoder
    from lsr_mmto.optimize.solid_part_analysis import _Materials
    from lsr_mmto.problem.settings import RunSettings
    import pyvista
    from lsr_mmto.optimize.design_model import MMTOModelAD
    from lsr_mmto.physics.thermoelastic.thermoelastic_physics import ThermoelasticHexFEA
# RESULT_FIELDS: the results a run saves, {name: dict(view, title, measures, where, shape)}:
#   view      the Results panel's Show choice that draws it
#   title     the colour scale's title
#   measures  what it is, for its unit: a unit kind, or None for no unit
#   where     "element" (one value per element, the default) or "node"
#   shape     "scalar" (one number, the default) or "vector" (x, y and z)
# final_summary returns one array for each name: (elements,) or (nodes,), with a last axis of 3 for a vector;
# a name it does not return, or values of another length or shape, stop the save with a message saying so.
# Thermal's: {"temperature": dict(view="Temperature", title="Temperature", measures="temperature", where="node")}
_STRAIN_WARNING = 0.01          # the final check warns when the largest thermal strain is above this (small strain)
RESULT_FIELDS = {
    "temperature": dict(view="Temperature", title="Temperature", measures="temperature", where="node"),
    "displacement": dict(view="Displacement", title="Displacement", measures="length", where="node", shape="vector"),
}


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


# the numbers of its own a run's Summary shows: the last value of each history key, when the run recorded it
SUMMARY_ROWS = {
    "max_temperature": dict(label="Max temperature", unit="{temperature}", format=".4g"),
    "max_thermal_strain": dict(label="Max thermal strain", unit="", format=".3%"),
    "max_displacement": dict(label="Max displacement", unit="{length}", format=".4g"),
    "thermal_share": dict(label="Thermal share of compliance", unit="", format=".3f"),
}

# OPTIONAL: iteration_diagnostics, what each iteration records
def iteration_diagnostics(history: dict[str, list[float | np.ndarray]], iteration: int, m: "MMTOModelAD",
                          x_phys: np.ndarray, properties: dict[str, torch.Tensor], elem_volume: float,
                          design_domain_volume: float, plot_progress: bool = False,
                          plotter: "pyvista.Plotter | None" = None, verbose: bool = True) -> None:
    """After the iteration's solve: the highest temperature, the largest thermal strain (each element's
    expansion coefficient times its largest |T - T_ref|), the largest displacement and the thermal share of
    the compliance (f_th . u over (f + f_th) . u), recorded into history and printed when verbose."""
    for key in SUMMARY_ROWS:
        history.setdefault(key, [])
    T, u, f_th = m.T.detach(), m.u.detach(), m.f_th.detach()
    dT = (T[m.fe.thermal.edof] - float(m.settings.reference_temperature)).abs().amax(dim=1)   # (N,)
    strain = float(torch.max(properties["Thermal_Expansion"] * dT))
    total = float(torch.dot(m.fe.total_force.detach(), u))
    share = float(torch.dot(f_th, u)) / total if total != 0.0 else float("nan")
    history["max_temperature"].append(float(T.max()))
    history["max_thermal_strain"].append(strain)
    history["max_displacement"].append(float(u.view(-1, 3).norm(dim=1).max()))
    history["thermal_share"].append(share)
    if verbose:
        print(f"[THERMOELASTIC] iteration {iteration}: max temperature {float(T.max()):.4g}, "
              f"max thermal strain {strain:.3%}, thermal share {share:.3f}")

# ESSENTIAL: final_summary, the final check and the results a run saves
def final_summary(designs: "list[tuple[str, np.ndarray, np.ndarray | torch.Tensor]]", solver: "ThermoelasticHexFEA",
                  properties_of: Callable[[torch.Tensor], dict[str, torch.Tensor]], settings: "RunSettings",
                  encoder: "MaterialEncoder | _Materials", latent_dim: int, elem_volume: float,
                  design_domain_volume: float) -> dict[str, np.ndarray]:
    """The final check's numbers: each design (label, x, z element-major) solved as analyse does (temperature,
    thermal force, displacement), its thermoelastic compliance, thermal share and highest temperature printed,
    with a warning when its largest thermal strain is above _STRAIN_WARNING. properties_of(z) gives the decoded
    properties of latent points z. Returns the results a run saves (RESULT_FIELDS), of the last design: the
    temperature at each node, and the displacement (x, y, z) at each node."""
    T_ref = float(settings.reference_temperature)
    for label, x, z in designs:
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(x), dtype=torch.float64)
            z = z if torch.is_tensor(z) else torch.as_tensor(np.asarray(z), dtype=torch.get_default_dtype())
            props = properties_of(z)
            solver.thermal.set_conductivity(props["Conductivity"])
            T = solver.thermal.solve(x, ad_mm.MaterialModel.SIMP).detach()
            E, nu = elastic_constants(props, settings)
            f_th = thermal_force(solver, T, x, E, nu, props["Thermal_Expansion"], T_ref)
            solver.set_material_fields(E, nu)
            u = solver.solve(x, ad_mm.MaterialModel.SIMP, extra_force=f_th).detach()
            compliance = float(torch.dot(solver.total_force, u))
            share = float(torch.dot(f_th, u)) / compliance if compliance != 0.0 else float("nan")
            dT = (T[solver.thermal.edof] - T_ref).abs().amax(dim=1)
            strain = float(torch.max(props["Thermal_Expansion"] * dT))
        print(f"[THERMOELASTIC] {label}: thermoelastic compliance {compliance:.6g}, thermal share {share:.3f}, "
              f"max temperature {float(T.max()):.4g}")
        if strain > _STRAIN_WARNING:
            print(f"Warning: the largest thermal strain is {strain:.1%}, more than {_STRAIN_WARNING:.0%}. "
                  f"The results assume small strains.")
    return {"temperature": T.numpy(), "displacement": u.numpy().reshape(-1, 3)}
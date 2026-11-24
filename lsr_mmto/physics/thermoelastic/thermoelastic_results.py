"""
thermoelastic_results.py
========================
What the thermoelastic module records and reports: the temperature and displacement a run saves
(RESULT_FIELDS) and its Summary rows (SUMMARY_ROWS); each iteration's highest temperature, largest thermal
strain, largest displacement and the thermal force's share of the compliance, recorded into the history
(iteration_diagnostics); and the final designs' compliance, printed by the final check (final_summary),
with a warning when the thermal strain is too large for small strain.
"""
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
if TYPE_CHECKING:
    from lsr_mmto.materials.material_source import MaterialSource
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
_STRAIN_WARNING = 0.01          # the final check warns when the largest thermal strain is above this (small strain)
RESULT_FIELDS = {
    "temperature": dict(view="Temperature", title="Temperature", measures="temperature", where="node"),
    "displacement": dict(view="Displacement", title="Displacement", measures="length", where="node", shape="vector"),
}


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
                  encoder: "MaterialSource", latent_dim: int, elem_volume: float,
                  design_domain_volume: float) -> dict[str, np.ndarray]:
    """The final check's numbers: each design (label, x, z element-major) solved as analyse does (temperature,
    thermal force, displacement), its thermoelastic compliance, thermal share and highest temperature printed,
    with a warning when its largest thermal strain is above _STRAIN_WARNING. properties_of(z) gives the decoded
    properties of latent points z. Returns the results a run saves (RESULT_FIELDS), of the last design: the
    temperature at each node, and the displacement (x, y, z) at each node."""
    from lsr_mmto.physics.available_physics import design_for_analyse
    from lsr_mmto.physics.thermoelastic.thermoelastic_physics import analyse

    T_ref = float(settings.reference_temperature)
    for label, x, z in designs:
        with torch.no_grad():
            z = z if torch.is_tensor(z) else torch.as_tensor(np.asarray(z), dtype=torch.get_default_dtype())
            props = properties_of(z)
            design = design_for_analyse(solver, settings, x, props)
            analyse(design)                                         # the run's solves, temperature first
            T, f_th, u = design.T.detach(), design.f_th.detach(), design.u.detach()
            compliance = float(torch.dot(solver.total_force, u))
            share = float(torch.dot(f_th, u)) / compliance if compliance != 0.0 else float("nan")
            dT = (T[solver.thermal.edof] - T_ref).abs().amax(dim=1)
            strain = float(torch.max(design.props["Thermal_Expansion"] * dT))   # at temperature
        print(f"[THERMOELASTIC] {label}: thermoelastic compliance {compliance:.6g}, thermal share {share:.3f}, "
              f"max temperature {float(T.max()):.4g}")
        if strain > _STRAIN_WARNING:
            print(f"Warning: the largest thermal strain is {strain:.1%}, more than {_STRAIN_WARNING:.0%}. "
                  f"The results assume small strains.")
    return {"temperature": T.numpy(), "displacement": u.numpy().reshape(-1, 3)}
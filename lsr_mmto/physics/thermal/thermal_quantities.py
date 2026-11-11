"""
thermal_quantities.py
=====================
What the thermal module optimizes: its objective and constraint (QUANTITIES: thermal compliance). What
it records and reports is in thermal_results.py.
"""
from typing import TYPE_CHECKING

import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path
from lsr_mmto.physics.thermal.thermal_physics import CONDUCTIVITY

if TYPE_CHECKING:
    from lsr_mmto.optimize.design_model import MMTOModelAD
    from lsr_mmto.optimize.solid_part_analysis import _Analysed


# ESSENTIAL: the fn of the ThermalCompliance quantity (QUANTITIES)
def _thermal_compliance(m: "MMTOModelAD | _Analysed", opt: dict[str, int | float | str] | None) -> torch.Tensor:
    """f . T, a torch scalar, from the solver's heat at each node and the temperature m.T: large when the
    heat put in sits far, in conduction terms, from where the temperature is held. Zero for a problem with
    no heat source, and the loop divides every objective by its first value, so such a problem needs
    another objective."""
    return torch.dot(m.fe.total_force, m.T)


# the thermal objective and constraint, as the arguments of a Quantity (formulation/quantities.py)
QUANTITIES = [
    dict(name="ThermalCompliance", objective=True, constraint=True, needs=[CONDUCTIVITY], fn=_thermal_compliance,
         doc="f . T", shown="Thermal compliance",
         help="How hot the part gets where the heat goes in. Lower carries the heat away better.",
         unit="{heat}*{temperature}", scales_with={CONDUCTIVITY: -1}),
]

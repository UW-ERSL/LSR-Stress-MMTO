"""
thermoelastic_quantities.py
===========================
What the thermoelastic module optimizes: its objectives and constraints (QUANTITIES: thermoelastic
compliance). What it records and reports is in thermoelastic_results.py.
"""
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from lsr_mmto.optimize.design_model import MMTOModelAD
    from lsr_mmto.optimize.solid_part_analysis import _Analysed


# ESSENTIAL: the fn of the ThermoelasticCompliance quantity (QUANTITIES)
def _thermoelastic_compliance(m: "MMTOModelAD | _Analysed", opt: dict[str, int | float | str] | None) -> torch.Tensor:
    """(F + F_th) . u, a torch scalar: the solver's total force (the applied loads plus the thermal force,
    as analyse passes it) dotted with the displacement m.u."""
    return torch.dot(m.fe.total_force, m.u)


# the thermoelastic objectives and constraints, as the arguments of a Quantity (formulation/quantities.py)
QUANTITIES = [
    dict(name="ThermoelasticCompliance", objective=True, constraint=True, needs=[], fn=_thermoelastic_compliance,
         doc="(F + F_th) . u, the total load including the thermal force", shown="Thermoelastic compliance",
         help="How much the part gives under the loads and the heating. Lower is stiffer.",
         unit="{force}*{length}"),
]
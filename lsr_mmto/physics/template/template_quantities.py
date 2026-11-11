"""
template_quantities.py
======================
The job of this file: what the physics optimizes, its objectives and constraints (QUANTITIES). What a
run records and saves is in template_results.py. Thermal's thermal_quantities.py is this file filled in.

Who reads it:
    QUANTITIES   formulation/quantities.py turns each entry into a Quantity; MMTOModelAD
                 (optimize/design_model.py) calls its fn every iteration; the Formulation panel, the
                 reference page and the user guide list it

How a quantity gets its gradient: its fn computes the value from m with torch operations only: what
analyse left on m comes out of the solver's differentiable solve, m.x_phys out of the filter and
projection, m.props out of the decoder. So autograd follows the value back to the densities and the
latent points; nothing is differentiated by hand. numpy, .item(), .detach() or float() inside fn cut
the tree, and the gradient of whatever comes after is silently zero.
"""
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from lsr_mmto.optimize.design_model import MMTOModelAD
    from lsr_mmto.optimize.solid_part_analysis import _Analysed


# ESSENTIAL: the fn of your quantity (QUANTITIES); one function for each quantity
def _your_quantity(m: "MMTOModelAD | _Analysed", opt: dict[str, int | float | str] | None) -> torch.Tensor:
    """The quantity's value, a torch scalar, from what analyse left on m. opt holds the options written in
    the formula, None when it takes none. Thermal's thermal compliance: torch.dot(m.fe.total_force, m.T)."""
    raise NotImplementedError("FILL: a quantity of this physics, in torch")


# QUANTITIES: each entry the arguments of a Quantity (formulation/quantities.py):
#   name        the name a formula uses, "ThermalCompliance()"; unique across every physics
#   objective   may it be the objective
#   constraint  may it be a constraint
#   needs       the database columns fn reads; a database without them gives a clear error
#   fn          fn(m, opt) -> a torch scalar, a function above
#   doc         one line for a reader of the code (and the reference page)
#   shown       its name in the GUI
#   help        one plain sentence for a user
#   unit        its unit, with unit kinds in braces filled in from the project's unit system
#               ("{heat}*{temperature}"), or text shown as it is ("no units")
#   scales_with optional: how the value follows a database column given in the wrong unit, {column: power};
#               thermal compliance goes as 1 / conductivity, so {"Conductivity": -1}
# Thermal's:
#   QUANTITIES = [dict(name="ThermalCompliance", objective=True, constraint=True, needs=["Conductivity"],
#                      fn=_thermal_compliance, doc="f . T", shown="Thermal compliance",
#                      help="How hot the part gets where the heat goes in.", unit="{heat}*{temperature}",
#                      scales_with={"Conductivity": -1})]
QUANTITIES = ...                                                    # FILL

# ----------------------------------------------------------------------------- Helpers
# Anything the quantities share (a p-norm, say). Mark each one "# HELPER: of <the quantity's fn>".

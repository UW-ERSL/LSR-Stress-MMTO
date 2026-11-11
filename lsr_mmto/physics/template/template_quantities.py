"""
template_quantities.py
======================
What this file is for: what the template physics optimizes, its objectives and constraints
(QUANTITIES). What a run records and saves is in template_results.py.

Who reads it:
    QUANTITIES             formulation/quantities.py turns each entry into a Quantity; MMTOModelAD
                           (optimize/design_model.py) calls its fn every iteration; the Formulation
                           panel, the reference page and the user guide list it

How a quantity gets its gradient: fn computes its value from m with torch operations only. m.c comes
out of the solver's differentiable solve, m.x_phys out of the filter and projection, m.props out of
the decoder. So torch's autograd can follow the value back to the densities and the latent points;
nothing here is differentiated by hand. Avoid numpy, .item(), .detach() or float() inside fn: they cut
the tree, and the gradient of whatever comes after is silently zero.

To make your own, change the lines marked FILL.
"""
from typing import TYPE_CHECKING

import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  # keep as is

from lsr_mmto.physics.template.template_physics import COEFFICIENT

if TYPE_CHECKING:
    from lsr_mmto.optimize.design_model import MMTOModelAD
    from lsr_mmto.optimize.solid_part_analysis import _Analysed


# ----------------------------------------------------------------------------- objectives and constraints
def _diffusion_compliance(m: "MMTOModelAD | _Analysed", opt: dict[str, int | float | str] | None) -> torch.Tensor:
    """f . c: large when the sources sit far (in diffusion terms) from where c is held. The diffusion
    counterpart of structural compliance F . u. It is zero for a problem with no source (held values
    only), and the loop divides every objective by its first value, so such a problem needs another
    objective. m: the iteration's model (see analyse in template_physics.py); opt: the options written
    in the formula, None here (it takes none)."""
    return torch.dot(m.fe.total_force, m.c)                  # FILL: your quantity, torch only


# Each entry is the arguments of a Quantity (formulation/quantities.py):
#   name        the name a formula uses, "DiffusionCompliance()"; unique across every physics
#   objective   may it be the objective
#   constraint  may it be a constraint
#   needs       the database columns fn reads; a database without them gives a clear error
#   fn          fn(m, opt) -> a torch scalar
#   doc         one line for a reader of the code (and the reference page)
#   shown       its name in the GUI
#   help        one plain sentence for a user
#   unit        its unit, with unit kinds in braces filled in from the project's unit system
#               ("{force}*{length}"), or text shown as it is ("no units")
#   scales_with how the value follows a database column given in the wrong unit, {column: power}: the
#               value of f . c goes as 1 / coefficient, so a coefficient 1000 times too large makes it
#               1000 times too small (optional; the project's unit warnings read it)
QUANTITIES = [
    dict(name="DiffusionCompliance",                          # FILL
         objective=True, constraint=True,
         needs=[COEFFICIENT],
         fn=_diffusion_compliance,
         doc="f . c",
         shown="Diffusion compliance",                        # FILL
         help="How hard the sources push the value up. Lower means the part carries it away better.",
         unit="no units",
         scales_with={COEFFICIENT: -1}),
]

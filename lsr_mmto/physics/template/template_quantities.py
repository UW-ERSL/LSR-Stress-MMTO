"""
template_quantities.py
======================
What this file is for: what the template physics measures. Its objectives and constraints
(QUANTITIES), the per-element results a run saves (RESULT_FIELDS), what each iteration records
(iteration_diagnostics), and the numbers of the final design (final_summary).

Who reads it:
    QUANTITIES             formulation/quantities.py turns each entry into a Quantity; MMTOModelAD
                           (optimize/design_model.py) calls its fn every iteration; the Formulation
                           panel, the reference page and the user guide list it
    RESULT_FIELDS          optimize/final_design_analysis.py saves each one; the Results panel shows it
    iteration_diagnostics  optimize/optimization_loop.py, after each iteration's solve
    final_summary          optimize/final_design_analysis.py, once, on the final designs; it returns
                           the RESULT_FIELDS values that are saved

How a quantity gets its gradient: fn computes its value from m with torch operations only. m.c comes
out of the solver's differentiable solve, m.x_phys out of the filter and projection, m.props out of
the decoder. So torch's autograd can follow the value back to the densities and the latent points;
nothing here is differentiated by hand. Avoid numpy, .item(), .detach() or float() inside fn: they cut
the tree, and the gradient of whatever comes after is silently zero.

To make your own, change the lines marked FILL.
"""
import numpy as np
import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  # keep as is
from lsr_mmto.pyto_ad_imports import topopt_material_model as ad_mm

from lsr_mmto.physics.template.template_physics import COEFFICIENT


# ----------------------------------------------------------------------------- objectives and constraints
def _diffusion_compliance(m, opt):
    """f . c: large when the sources sit far (in diffusion terms) from where c is held. The diffusion
    counterpart of structural compliance F . u. It is zero for a problem with no source (held values
    only), and the loop divides every objective by its first value, so such a problem needs another
    objective. m: the iteration's model (see analyse in template_physics.py); opt: the options written
    in the formula, {} here."""
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


def _element_mean(solver, c):
    """c at each element: the mean of its 8 nodes."""
    return c[solver.edof].mean(1)


# ----------------------------------------------------------------------------- each iteration, and the end
def iteration_diagnostics(history, iteration, solver, x_phys, properties, elem_volume, design_domain_volume,
                          plot_progress=False, plotter=None, verbose=True):
    """After each iteration's solve. history is the run's record ({name: list}); append to it what you want
    to keep, and print a line when verbose. Here: the largest c of the last solve."""
    if solver.sol is None:
        return
    largest = float(torch.max(solver.sol.detach()))
    history.setdefault("template_max_value", []).append(largest)             # FILL
    if verbose:
        print(f"[TEMPLATE] iteration {iteration}: largest value {largest:.4g}")


def final_summary(designs, solver, properties_of, settings, encoder, latent_dim, elem_volume,
                  design_domain_volume):
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

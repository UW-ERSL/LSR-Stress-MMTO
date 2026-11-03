"""
structural_module.py
====================
The structural module's front page: everything the rest of LSR MMTO takes from the structural
physics, one entry each, as physics/available_physics.py lists them (MODULE_ENTRIES). Small-strain
linear elasticity with a different material in each element; the stresses are von Mises.

    structural_physics.py              the solver, one iteration's solve, element stress
    structural_supports_and_loads.py   fixed, displacement, force, traction
    structural_quantities.py           compliance, p-norm stress and failure factor, the stress
                                       diagnostics of each iteration and of the final design
"""
from lsr_mmto.physics.structural.structural_physics import (analyse, deflation_space, make_solver,  # noqa: F401
                                                             set_pnorm_exponent)
from lsr_mmto.physics.structural.structural_quantities import (QUANTITIES, final_summary,  # noqa: F401
                                                               iteration_diagnostics)
from lsr_mmto.physics.structural.structural_supports_and_loads import (SUPPORT_AND_LOAD_KINDS,  # noqa: F401
                                                                      supports_and_loads_error)

NAME = "structural"
DOC = "linear elasticity, small strain"
FIELDS = {"u": 3}                                   # the displacement, three unknowns per node
PROPERTIES = ["Youngs_Modulus"]                     # Poisson's ratio may be one value for all (SETTINGS)
NEEDS = ("support", "load")                         # a structural problem with no load has nothing to solve

# the structural settings, as the arguments of a Setting (problem/settings.py)
SETTINGS = [
    dict(section="common", key="pnorm_exponent", default=6.0, label="p-norm exponent",
         help="How closely the single stress value follows the highest stress in the part. Larger is closer, "
              "but harder to optimize.", group="Main", at_build=set_pnorm_exponent),
    dict(section="advanced", key="use_constant_poissons_ratio", default=False, label="Constant Poisson's ratio",
         help="One Poisson's ratio for every material, instead of each material's own.", group="Material model"),
    dict(section="advanced", key="constant_poissons_ratio", default=0.3, label="Poisson's ratio value",
         help="The Poisson's ratio used when Constant Poisson's ratio is ticked.", group="Material model"),
]

# the per-element results a run saves, by name: the view that shows it in the Results panel, its title on the
# colour scale, what it measures (for its unit), and its value from the solver after the final check
RESULT_FIELDS = {
    "von_mises": dict(view="Stress", title="von Mises stress", measures="stress"),
}

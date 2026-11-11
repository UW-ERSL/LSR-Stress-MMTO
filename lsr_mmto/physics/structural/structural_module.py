"""
structural_module.py
====================
The structural module's front page: everything the rest of LSR MMTO takes from the structural
physics, one entry each, as physics/available_physics.py lists them (MODULE_ENTRIES). Small-strain
linear elasticity with a different material in each element; the stresses are von Mises.

    structural_physics.py              the solver, one iteration's solve, element stress
    structural_supports_and_loads.py   fixed, displacement, force, traction
    structural_quantities.py           compliance, p-norm stress and failure factor (QUANTITIES)
    structural_results.py              what each iteration records, the final check, the results a run
                                       saves and its Summary rows

Each top-level function in the folder has a marker on the line above it:

    # ESSENTIAL: ...    a required entry of this page, or the function one of them names
    # OPTIONAL: ...     an optional entry (available_physics.OPTIONAL_ENTRIES); a physics that leaves it
                        out gets a default that does nothing
    # HELPER: ...       used only inside this folder, by the entry or function it names

The helpers behind each entry:

    make_solver              MultiMaterialHexFEA (its postprocess uses von_mises)
    analyse                  elastic_constants, von_mises
    QUANTITIES               _compliance, _stress_pnorm (pnorm), _failure_factor_pnorm
    SUPPORT_AND_LOAD_KINDS   _apply_support (_dofs), _apply_force and _apply_traction (_add_load),
                             _picked_force, _check_direction
    iteration_diagnostics    record_fea_diagnostics: compute_max_failure_factor, update_paper_history,
                             compute_stress_violation_diagnostics, print_stress_violation_diagnostics
    final_summary            elastic_constants, eval_stress_compliance_metrics (compute_max_failure_factor),
                             compute_stress_violation_diagnostics, print_stress_violation_diagnostics
"""
from lsr_mmto.physics.structural.structural_physics import (analyse, deflation_space, make_solver,  # noqa: F401
                                                             set_pnorm_exponent)
from lsr_mmto.physics.structural.structural_quantities import QUANTITIES  # noqa: F401
from lsr_mmto.physics.structural.structural_results import (RESULT_FIELDS, SUMMARY_ROWS,  # noqa: F401
                                                            final_summary, iteration_diagnostics)
from lsr_mmto.physics.structural.structural_supports_and_loads import (SUPPORT_AND_LOAD_KINDS,  # noqa: F401
                                                                      supports_and_loads_error)

NAME = "structural"
DOC = "linear elasticity, small strain"
FIELDS = {"u": 3}                                   # the displacement, three unknowns per node
PROPERTIES = ["Youngs_Modulus"]                     # Poisson's ratio may be one value for all (SETTINGS)
OPTIONAL_PROPERTIES = ["Poissons_Ratio"]            # each material's own when the database has it, else SETTINGS'

# the material properties it brings, each as the arguments of an Attribute (materials/databases.py); the core
# has Density and Cost
NEW_PROPERTIES = [
    dict(name="Youngs_Modulus", measures="stress", doc="Young's modulus, the stiffness", shown="Young's modulus",
         low=0, aliases=("Youngs Modulus", "Young_Modulus", "E"),
         wrong_unit_effect="Displacement and compliance come out {times} times too small; stress and failure "
                           "factor are unaffected.",
         legend_symbol="E", legend_rank=2),
    dict(name="Poissons_Ratio", measures="ratio", doc="Poisson's ratio", shown="Poisson's ratio", low=-1, high=0.5,
         aliases=("Poisson_Ratio", "Poissons Ratio", "nu")),
    dict(name="Yield_Strength", measures="stress", doc="yield strength, the stress at which the material yields",
         shown="Yield strength", low=0, aliases=("Yield Strength", "Yield"),
         legend_symbol=r"\sigma_Y", legend_rank=1, iteration_label="Y", iteration_rank=1),
]
NEEDS = ("support", "load")                         # a structural problem with no load has nothing to solve
PENALTIES = []                                      # none besides those every physics has

# the kinds of quantity of its own values and quantities (forces, stresses), with their units' values in SI
# and each unit system's unit; the core has length, density, mass and cost per mass already
_IN, _LBF = 0.0254, 4.4482216152605                 # inch in m, pound-force in N
UNIT_KINDS = {
    "force": {"units": {"N": 1.0, "lbf": _LBF}, "MKS": "N", "mmKS": "N", "IPS": "lbf"},
    "stress": {"units": {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9, "psi": _LBF / _IN ** 2,
                         "ksi": 1e3 * _LBF / _IN ** 2},
               "MKS": "Pa", "mmKS": "MPa", "IPS": "psi"},
}

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



"""
thermoelastic_module.py
=======================
The thermoelastic module's front page: everything the rest of LSR MMTO takes from the thermoelastic
physics, one entry each, as physics/available_physics.py lists them (MODULE_ENTRIES). Linear isotropic
thermoelasticity with a different material in each element, coupled one way: a steady temperature, solved
by conduction, drives the displacement through each element's thermal expansion.

It is not in the list of physics (physics/available_physics.py) yet.

    thermoelastic_physics.py              the solvers, one iteration's solve
    thermoelastic_supports_and_loads.py   its kinds of support and load
    thermoelastic_quantities.py           its objectives and constraints (QUANTITIES)
    thermoelastic_results.py              what a run saves and the final check

Each top-level function in the folder has a marker on the line above it:

    # ESSENTIAL: ...    a required entry of this page, or the function one of them names
    # OPTIONAL: ...     an optional entry (available_physics.OPTIONAL_ENTRIES); a physics that leaves it
                        out gets a default that does nothing
    # HELPER: ...       used only inside this folder, by the entry or function it names

It reads thermal's material property and units (Conductivity, temperature) and structural's (Young's
modulus, Poisson's ratio); the property it brings is the thermal expansion coefficient.
"""
from lsr_mmto.physics.thermoelastic.thermoelastic_physics import analyse, make_solver  # noqa: F401
from lsr_mmto.physics.thermoelastic.thermoelastic_quantities import QUANTITIES  # noqa: F401
from lsr_mmto.physics.thermoelastic.thermoelastic_results import (RESULT_FIELDS, SUMMARY_ROWS,  # noqa: F401
                                                                  final_summary, iteration_diagnostics)
from lsr_mmto.physics.thermoelastic.thermoelastic_supports_and_loads import (SUPPORT_AND_LOAD_KINDS,  # noqa: F401
                                                                            supports_and_loads_error)

NAME = "thermoelastic"
DOC = "linear isotropic thermoelasticity, one-way coupled: the temperature drives the displacement"
FIELDS = {"u": 3, "T": 1}                           # the displacement, three unknowns per node; the temperature, one
PROPERTIES = ["Youngs_Modulus", "Conductivity", "Thermal_Expansion"]   # Poisson's ratio may be one value (SETTINGS)
OPTIONAL_PROPERTIES = ["Poissons_Ratio"]            # each material's own when the database has it, else SETTINGS'

# the material property it brings, as the arguments of an Attribute (materials/databases.py); Young's modulus and
# Poisson's ratio are structural's, Conductivity is thermal's
NEW_PROPERTIES = [
    dict(name="Thermal_Expansion", measures="expansion",
         doc="coefficient of thermal expansion, the strain per degree of temperature rise",
         shown="Thermal expansion coefficient", low=0, aliases=("Thermal Expansion", "CTE", "alpha")),
]

NEEDS = ("support",)                                # heating alone is a problem; a mechanical load is not needed

_RANKINE = 5.0 / 9.0                                # one degree Rankine in K

# the kind of quantity of its own values (per degree, for the expansion coefficient), with its units' values in
# SI and each unit system's unit; the temperature itself is thermal's
UNIT_KINDS = {
    "expansion": {"units": {"1/K": 1.0, "1/°R": 1.0 / _RANKINE},
                  "MKS": "1/K", "mmKS": "1/K", "IPS": "1/°R"},
}

# the thermoelastic settings, as the arguments of a Setting (problem/settings.py); Poisson's ratio has keys of its
# own, separate from structural's
SETTINGS = [
    dict(section="common", key="reference_temperature", default=293.15, label="Reference temperature",
         help="The temperature at which the part has no thermal strain, in the project's temperature unit.",
         group="Main"),
    dict(section="advanced", key="te_use_constant_poissons_ratio", default=False, label="Constant Poisson's ratio",
         help="One Poisson's ratio for every material, instead of each material's own.", group="Material model"),
    dict(section="advanced", key="te_constant_poissons_ratio", default=0.3, label="Poisson's ratio value",
         help="The Poisson's ratio used when Constant Poisson's ratio is ticked.", group="Material model"),
]

# OPTIONAL: PENALTIES, terms added to the objective besides the two every physics has (latent smoothing and
# the distance penalty), each the arguments of a Penalty (design/latent_penalties.py, which says what each
# piece is): its name, its name in the GUI and its help, its value fn(m) in torch, its weight(state, it) this
# iteration (0 is off: its schedule goes there), its scale ("raw" or "first value") and its history key. Its
# settings go in SETTINGS. A push towards fully solid or fully empty elements, off from iteration 100 on:
#   PENALTIES = [dict(name="HeatGreyness", shown="Greyness", help="Pushes each element towards solid or empty.",
#                     fn=lambda m: torch.mean(m.x_phys * (1 - m.x_phys)),
#                     weight=lambda state, it: 0.0 if state.iteration >= 100 else state.s.heat_greyness_weight,
#                     scale="first value", key="heat_greyness")]

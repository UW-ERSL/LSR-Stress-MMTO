"""
thermal_module.py
=================
The thermal module's front page: everything the rest of LSR MMTO takes from the thermal physics,
one entry each, as physics/available_physics.py lists them (MODULE_ENTRIES). Steady heat conduction
with a different material in each element: one temperature per node, each element's conductivity
from its material.

It is not in the list of physics (physics/available_physics.py), so the GUI does not offer it yet;
tests/test_thermal_physics.py registers it and runs it, so it keeps working as the code changes. It is
also the worked example of a physics: physics/template/ is the blank form, and this folder is the same
form filled in.

    thermal_physics.py              the solver, one iteration's solve
    thermal_supports_and_loads.py   fixed temperature, heat source
    thermal_quantities.py           thermal compliance (QUANTITIES)
    thermal_results.py              what each iteration records, the final check, the temperature a
                                    run saves and its Summary row
    example_materials.xlsx, .nt     four materials (Density, Conductivity) and their trained network

Each top-level function in the folder has a marker on the line above it:

    # ESSENTIAL: ...    a required entry of this page, or the function one of them names
    # OPTIONAL: ...     an optional entry (available_physics.OPTIONAL_ENTRIES); a physics that leaves it
                        out gets a default that does nothing
    # HELPER: ...       used only inside this folder, by the entry or function it names

The helpers behind each entry:

    make_solver              ConductionHexFEA
    SUPPORT_AND_LOAD_KINDS   _apply_fixed_temperature, _apply_heat_source, _check_total

Left out, so the defaults apply (OPTIONAL_ENTRIES): OPTIONAL_PROPERTIES, SETTINGS, PENALTIES,
deflation_space (the direct solver at every size).
"""
from lsr_mmto.physics.thermal.thermal_physics import CONDUCTIVITY, analyse, make_solver  # noqa: F401
from lsr_mmto.physics.thermal.thermal_quantities import QUANTITIES  # noqa: F401
from lsr_mmto.physics.thermal.thermal_results import (RESULT_FIELDS, SUMMARY_ROWS,  # noqa: F401
                                                      final_summary, iteration_diagnostics)
from lsr_mmto.physics.thermal.thermal_supports_and_loads import (SUPPORT_AND_LOAD_KINDS,  # noqa: F401
                                                                supports_and_loads_error)

NAME = "thermal"
DOC = "steady heat conduction, one temperature per node"
FIELDS = {"T": 1}                                   # the temperature, one unknown per node
PROPERTIES = [CONDUCTIVITY]

# the material property it brings, as the arguments of an Attribute (materials/databases.py); the core has
# Density and Cost
NEW_PROPERTIES = [
    dict(name=CONDUCTIVITY, measures="conductivity", doc="thermal conductivity, how well heat flows through it",
         shown="Thermal conductivity", low=0, aliases=("Thermal_Conductivity", "Thermal Conductivity", "k")),
]

NEEDS = ("support",)                                # held temperatures alone are a problem; a source is not needed

_BTU = 1055.05585262                                # one BTU in J
_RANKINE = 5.0 / 9.0                                # one degree Rankine in K

# the kinds of quantity its values and quantities measure that the core does not have: each unit's value in SI,
# and the kind's unit in each unit system (mmKS: mm, t, s, so a power is N mm / s = mW)
UNIT_KINDS = {
    "temperature": {"units": {"K": 1.0, "°R": _RANKINE},
                    "MKS": "K", "mmKS": "K", "IPS": "°R"},
    "heat": {"units": {"W": 1.0, "mW": 1e-3, "BTU/s": _BTU},
             "MKS": "W", "mmKS": "mW", "IPS": "BTU/s"},
    "conductivity": {"units": {"W/(m*K)": 1.0, "mW/(mm*K)": 1.0, "BTU/(s*in*°R)": _BTU / (0.0254 * _RANKINE)},
                     "MKS": "W/(m*K)", "mmKS": "mW/(mm*K)", "IPS": "BTU/(s*in*°R)"},
}

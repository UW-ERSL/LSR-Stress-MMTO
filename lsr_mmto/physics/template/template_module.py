"""
template_module.py
==================
What this file is for: the front page of the template physics, a small working steady diffusion
(one unknown per node), kept as the place to start a physics of your own. It is not in the list of
physics (physics/available_physics.py), so the GUI never offers it; tests/test_template_physics.py
registers it and runs it, so it keeps working as the code changes.

Who reads it: physics/available_physics.py checks that a front page gives every entry of
MODULE_ENTRIES and makes the core's lists from it (quantities, kinds of support and load, settings);
the rest of LSR MMTO only ever reaches a physics through these names.

    template_physics.py              the solver, and one iteration's solve (start here next)
    template_supports_and_loads.py   Set value and Source, and how they reach the solve
    template_quantities.py           Diffusion compliance, the saved result, the diagnostics

To make a physics of your own:
    1. Copy this folder, e.g. to physics/heat/.
    2. Rename the four files (heat_module.py, heat_physics.py, heat_supports_and_loads.py,
       heat_quantities.py) and the imports between them.
    3. Change every line marked FILL. The lines marked "keep as is" stay.
    4. Give it names of its own: NAME below, each kind in SUPPORT_AND_LOAD_KINDS, each quantity and
       each setting key must not be used by another physics.
    5. Add one line to physics/available_physics.py: import your front page, and put it in MODULES.
    6. Copy tests/test_template_physics.py for your physics and run it.

Large problems: above the direct solver's limit (DIRECT_SOLVER_DOF_CUTOFF in PyTO-AD) the loop uses
DPCG only for a physics that builds its own deflation space (deflation_space below); the template has
none, so it keeps the direct solver at every size.
"""
# keep as is: the entries the other three files give
from lsr_mmto.physics.template.template_physics import analyse, make_solver  # noqa: F401
from lsr_mmto.physics.template.template_quantities import (QUANTITIES, RESULT_FIELDS,  # noqa: F401
                                                           final_summary, iteration_diagnostics)
from lsr_mmto.physics.template.template_supports_and_loads import (SUPPORT_AND_LOAD_KINDS,  # noqa: F401
                                                                  supports_and_loads_error)
from lsr_mmto.physics.template.template_physics import COEFFICIENT

# NAME: the name a project file gives ("physics": "template"); read by project.py and the GUI.
NAME = "template"                                                  # FILL

# DOC: one line on what the physics is; the reference page shows it.
DOC = "steady diffusion, one unknown per node (the template to copy)"   # FILL

# FIELDS: what is solved for, {name: unknowns per node}. Read by project.py (one boundary condition
# per field) and by the loop (the size of the problem). Structural is {"u": 3}.
FIELDS = {"c": 1}                                                  # FILL

# PROPERTIES: the database columns every problem of this physics reads; a database without them is
# refused before the run.
PROPERTIES = [COEFFICIENT]                                         # FILL

# deflation_space: deflation_space(dsolver, mesh) builds the deflation space of this physics' system on
# dsolver, so problems above the direct solver's limit use DPCG (structural: the rigid-body motions of
# groups of nodes). None keeps the direct solver at every size. Read by optimize/optimization_loop.py.
deflation_space = None                                             # FILL, if your problems get large

# NEEDS: what every project of this physics must have, as roles of its kinds of support and load;
# Project.validate refuses a project without them ("no supports", "no loads"). Diffusion held at
# two values needs no load, so only a support is needed. Structural is ("support", "load").
NEEDS = ("support",)                                               # FILL

# UNIT_KINDS: the kinds of quantity this physics' own values and quantities measure that the core does
# not have (the core has length, density, mass, cost per mass and ratio), each with its units' values in
# SI and its unit in each unit system (MKS, mmKS, IPS). Read by materials/databases.py, which adds them to
# UNITS and UNIT_SYSTEMS. Structural gives force and stress. Diffusion's values have no unit, so none.
UNIT_KINDS = {}                                                    # FILL, e.g. {"temperature": {...}}

# SETTINGS: settings of this physics only, each the arguments of a Setting (problem/settings.py): its
# section ("common" shows in Main, "advanced" in Advanced), key (read as m.settings.<key>), default,
# label and help in the Settings panel, and the group it sits in. May be empty.
SETTINGS = [
    dict(section="advanced", key="template_coefficient_scale", default=1.0, label="Coefficient scale",
         help="Multiplies every material's diffusion coefficient.", group="Material model"),   # FILL
]

# The other entries come from the other files (imported above):
#   make_solver, analyse                         template_physics.py
#   SUPPORT_AND_LOAD_KINDS, supports_and_loads_error   template_supports_and_loads.py
#   QUANTITIES, RESULT_FIELDS, iteration_diagnostics, final_summary   template_quantities.py

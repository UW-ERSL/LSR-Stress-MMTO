"""
template_module.py
==================
The blank form of a physics: copy this folder to start one of your own. Every line marked FILL is a
blank; the functions raise until they are filled in. physics/thermal/ is the same form filled in, a
working steady heat conduction, and physics/structural/ a larger one.

What a physics is
-----------------
A folder in lsr_mmto/physics/ that tells the core what to solve on the voxel mesh and what to measure.
The core (the loop, the filter, the material network, the GUI) has no physics written in it: it reaches
a physics only through the names on its front page, this file, which physics/available_physics.py lists
(MODULE_ENTRIES says what each one is).

The five jobs, one file each
----------------------------
    template_module.py              the front page: its name, what it solves for, the material properties
                                    and units it reads, and the entries the other four files give
    template_physics.py             the solver (make_solver) and one iteration's solve (analyse)
    template_supports_and_loads.py  its kinds of support and load (SUPPORT_AND_LOAD_KINDS)
    template_quantities.py          its objectives and constraints (QUANTITIES)
    template_results.py             what a run saves (RESULT_FIELDS) and the final check (final_summary)

Each top-level function in a physics' folder has a marker on the line above it:

    # ESSENTIAL: ...    a required entry, or the function one of them names: every physics has these
    # OPTIONAL: ...     an optional entry (available_physics.OPTIONAL_ENTRIES); a physics that leaves it
                        out gets a default that does nothing. Here they are commented out, each with what it
                        does: uncomment the ones your physics needs
    # HELPER: ...       used only inside the folder, by the entry or function it names; each file ends with
                        a Helpers section saying what goes there

How to start
------------
    1. Copy this folder, e.g. to physics/heat/, and rename the five files (heat_module.py, heat_physics.py,
       heat_supports_and_loads.py, heat_quantities.py, heat_results.py) and the imports between them.
    2. Fill every FILL, file by file in the order above. thermal's file of the same job shows each one done.
    3. Give it names of its own: NAME, each kind of support and load, each quantity, each penalty and each
       setting key must not be used by another physics.
    4. Add one line to physics/available_physics.py: import your front page and put it in MODULES. Until
       then a test can register it, as tests/test_thermal_physics.py does with thermal.
    5. Copy tests/test_thermal_physics.py for your physics: a case whose answer is known, and a short run.

The bounds of a physics
-----------------------
    1. The core reaches a physics only through its front page; nothing outside the folder imports its
       other files (tests/test_physics_folders.py checks it).
    2. More files inside the folder are fine, when the name says the job (heat_radiation.py).
    3. A physics may import and reuse another physics' code: a thermoelastic physics can build on
       thermal's solver and structural's.
    4. Needing to change a file outside the folder means the core lacks something: propose the change to
       the core, never patch it in from the physics.
"""
# the entries the other four files give
from lsr_mmto.physics.template.template_physics import analyse, make_solver  # noqa: F401
from lsr_mmto.physics.template.template_quantities import QUANTITIES  # noqa: F401
from lsr_mmto.physics.template.template_results import RESULT_FIELDS, final_summary  # noqa: F401
from lsr_mmto.physics.template.template_supports_and_loads import SUPPORT_AND_LOAD_KINDS  # noqa: F401
# OPTIONAL entries of the other files, once you uncomment them there:
#   from lsr_mmto.physics.template.template_physics import deflation_space
#   from lsr_mmto.physics.template.template_supports_and_loads import supports_and_loads_error
#   from lsr_mmto.physics.template.template_results import SUMMARY_ROWS, iteration_diagnostics

# NAME: the name a project file gives ("physics": "heat"); read by project.py and the GUI. Thermal: "thermal".
NAME = ...                                                          # FILL

# DOC: one line on what the physics is; the reference page shows it.
DOC = ...                                                           # FILL

# FIELDS: what is solved for, {name: unknowns per node}. Read by project.py (one set of supports and loads per
# field) and by the loop (the size of the problem). Thermal: {"T": 1}; structural: {"u": 3}.
FIELDS = ...                                                        # FILL

# PROPERTIES: the database columns every problem of this physics reads; a database without them is refused
# before the run. Thermal: ["Conductivity"].
PROPERTIES = ...                                                    # FILL

# NEW_PROPERTIES: the material properties this physics brings, each the arguments of an Attribute
# (materials/databases.py): its name (the database column), what it measures (a unit kind: the core's, or
# one in UNIT_KINDS below), a one-line doc, its name in the GUI, its allowed range and other names it may be
# written under. The core has Density and Cost; a property another physics brings may be given again only
# exactly as that physics gives it (one entry, shared), so an empty list is fine when you read only those.
# Thermal:
#   NEW_PROPERTIES = [dict(name="Conductivity", measures="conductivity", doc="thermal conductivity",
#                          shown="Thermal conductivity", low=0, aliases=("k",))]
NEW_PROPERTIES = ...                                                # FILL

# NEEDS: what every project of this physics must have, as roles of its kinds of support and load;
# Project.validate refuses a project without them. Structural: ("support", "load"); thermal: ("support",),
# since held temperatures alone are a problem.
NEEDS = ...                                                         # FILL

# ----------------------------------------------------------------------------- OPTIONAL entries of this file
# OPTIONAL: OPTIONAL_PROPERTIES, the database columns it reads when the database has them and does without
# otherwise. Structural: ["Poissons_Ratio"] (else one value from its settings).
#   OPTIONAL_PROPERTIES = []

# OPTIONAL: UNIT_KINDS, the kinds of quantity its values and quantities measure that the core does not have
# (the core has length, density, mass, cost per mass and ratio): each unit's value in SI, and the kind's unit
# in each unit system (MKS, mmKS, IPS). Read by materials/databases.py. Thermal:
#   UNIT_KINDS = {"temperature": {"units": {"K": 1.0, "°R": 5 / 9}, "MKS": "K", "mmKS": "K", "IPS": "°R"}}

# OPTIONAL: SETTINGS, settings of this physics only, each the arguments of a Setting (problem/settings.py):
# its section ("common" shows in Main, "advanced" in Advanced), key (read as m.settings.<key>), default,
# label and help in the Settings panel, and the group it sits in. Structural has its p-norm exponent:
#   SETTINGS = [dict(section="common", key="pnorm_exponent", default=6.0, label="p-norm exponent",
#                    help="...", group="Main")]

# OPTIONAL: PENALTIES, terms added to the objective besides the two every physics has (latent smoothing and
# the distance penalty), each the arguments of a Penalty (design/latent_penalties.py, which says what each
# piece is): its name, its name in the GUI and its help, its value fn(m) in torch, its weight(state, it) this
# iteration (0 is off: its schedule goes there), its scale ("raw" or "first value") and its history key. Its
# settings go in SETTINGS. A push towards fully solid or fully empty elements, off from iteration 100 on:
#   PENALTIES = [dict(name="HeatGreyness", shown="Greyness", help="Pushes each element towards solid or empty.",
#                     fn=lambda m: torch.mean(m.x_phys * (1 - m.x_phys)),
#                     weight=lambda state, it: 0.0 if state.iteration >= 100 else state.s.heat_greyness_weight,
#                     scale="first value", key="heat_greyness")]

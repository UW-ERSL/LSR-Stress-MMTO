"""
available_physics.py
====================
The physics a project may name ("physics": "structural" in its file), one module each, listed in
MODULES by hand. A module is a folder in physics/ whose <name>_module.py is its front page: it
gives every entry in MODULE_ENTRIES, and the core takes what it needs of the physics only from
there. A new physics is one folder and one line in MODULES.

A module's files import the layers below them (elements/, design/, the PyTO-AD solvers), never
the core's lists: a module gives its quantities, its kinds of support and load and its settings
as plain arguments, and the core makes them into Quantity, BCKind and Setting entries.
"""
from lsr_mmto.physics.structural import structural_module

MODULES = [structural_module]

# what a module's front page gives, and what each is
MODULE_ENTRIES = {
    "NAME": "the name a project file uses",
    "DOC": "what the physics is, in one line",
    "FIELDS": "each field it solves for and its unknowns per node, e.g. {'u': 3}",
    "PROPERTIES": "the material database columns every problem of it reads",
    "make_solver": "make_solver(mesh, mat_prop, bcs, linear_solver, dsolver, elem_body_force): its finite "
                   "element solver; bcs holds one boundary condition per field",
    "analyse": "analyse(model): one iteration's solve inside the torch tree, from model.x_phys and model.props, "
               "leaving on the model what its quantities read",
    "SUPPORT_AND_LOAD_KINDS": "its kinds of support and load, {name: the arguments of a BCKind}, each naming "
                              "the field it acts on",
    "NEEDS": "what every project of it must have, as roles of its kinds: ('support', 'load') when a problem "
             "without a load has no meaning, ('support',) when held values alone make a problem",
    "supports_and_loads_error": "supports_and_loads_error(mesh, bcs): why the supports and loads leave the "
                                "problem without a solution (the part free to move), or None",
    "QUANTITIES": "its objectives and constraints, each as the arguments of a Quantity",
    "SETTINGS": "its own settings, each as the arguments of a Setting",
    "RESULT_FIELDS": "the per-element results a run saves, {name: dict(view, title, measures)}; final_summary "
                     "returns their values",
    "iteration_diagnostics": "iteration_diagnostics(history, iteration, solver, x_phys, properties, ...): what each "
                             "iteration records and prints besides the objective and the constraints",
    "final_summary": "final_summary(designs, solver, ...): prints the final designs' numbers and returns "
                     "{name: value per element} of the last design, one for each RESULT_FIELDS entry",
}


def _checked(module):
    missing = [k for k in MODULE_ENTRIES if not hasattr(module, k)]
    if missing:
        raise TypeError(f"physics module {module.__name__} does not give {', '.join(missing)}")
    return module


PHYSICS = {m.NAME: _checked(m) for m in MODULES}


def physics_module(name):
    """The front page of physics name; an unknown name is an error that lists the known ones."""
    if name not in PHYSICS:
        raise ValueError(f"physics {name!r}: one of {', '.join(PHYSICS)}")
    return PHYSICS[name]

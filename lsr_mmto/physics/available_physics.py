"""
available_physics.py
====================
The physics a project may name ("physics": "structural" in its file), one module each, listed in
MODULES by hand. A module is a folder in physics/ whose <name>_module.py is its front page: it
gives every entry in MODULE_ENTRIES, and the core takes what it needs of the physics only from
there. A new physics is one folder and one line in MODULES.

A module's files import the layers below them (elements/, design/, the PyTO-AD solvers), never
the core's lists: a module gives its quantities, its penalties, its kinds of support and load and
its settings as plain arguments, and the core makes them into Quantity, Penalty, BCKind and
Setting entries.

A front page is duck typed: any module that has every name of MODULE_ENTRIES below, checked only for
presence (_checked) when PHYSICS is made, not for signatures.
"""
from typing import TYPE_CHECKING, Any, Callable

from lsr_mmto.physics.structural import structural_module

if TYPE_CHECKING:
    from types import ModuleType

    from lsr_mmto.formulation.quantities import Quantity
    from lsr_mmto.materials.databases import Attribute

MODULES = [structural_module]

# what a module's front page gives, and what each is
MODULE_ENTRIES = {
    "NAME": "the name a project file uses",
    "DOC": "what the physics is, in one line",
    "FIELDS": "each field it solves for and its unknowns per node, e.g. {'u': 3}; the mesh size and the "
              "supports and loads (one boundary condition per field) are counted from it",
    "UNIT_KINDS": "the kinds of quantity its values and quantities measure that the core does not have, "
                  "{kind: {'units': {unit: value in SI}, 'MKS': unit, 'mmKS': unit, 'IPS': unit}}",
    "PROPERTIES": "the material database columns every problem of it reads",
    "NEW_PROPERTIES": "the material properties it brings that the core does not have (the core has Density and "
                      "Cost), each as the arguments of an Attribute (materials/databases.py); a property another "
                      "physics brings too is one entry, shared, when both give it the same way; may be empty",
    "make_solver": "make_solver(mesh, mat_prop, bcs, linear_solver, dsolver, elem_body_force): its finite "
                   "element solver; bcs holds one boundary condition per field",
    "analyse": "analyse(model): one iteration's solve inside the torch tree, from model.x_phys and model.props, "
               "leaving on the model what its quantities read",
    "SUPPORT_AND_LOAD_KINDS": "its kinds of support and load, {name: the arguments of a BCKind}, each naming "
                              "the field it acts on",
    "NEEDS": "what every project of it must have, as roles of its kinds: ('support', 'load') when a problem "
             "without a load has no meaning, ('support',) when held values alone make a problem",
    "deflation_space": "deflation_space(dsolver, mesh): builds the deflation space of its system on dsolver, so "
                       "problems above the direct solver's limit use DPCG; None for a physics without one, which "
                       "then keeps the direct solver at every size",
    "supports_and_loads_error": "supports_and_loads_error(mesh, bcs): why the supports and loads leave the "
                                "problem without a solution (the part free to move), or None",
    "QUANTITIES": "its objectives and constraints, each as the arguments of a Quantity",
    "PENALTIES": "the terms it adds to the objective besides those every physics has, each as the arguments "
                 "of a Penalty (design/latent_penalties.py); may be empty",
    "SETTINGS": "its own settings, each as the arguments of a Setting",
    "RESULT_FIELDS": "the results a run saves, {name: dict(view, title, measures, where, shape)}: view the Results "
                     "panel's choice that shows it, title its colour scale's, measures what it is (for its unit), "
                     "where 'element' (one value per element, the default) or 'node' (one per node), shape "
                     "'scalar' (one number, the default) or 'vector' (x, y and z); final_summary returns their "
                     "values",
    "iteration_diagnostics": "iteration_diagnostics(history, iteration, solver, x_phys, properties, ...): what each "
                             "iteration records and prints besides the objective and the constraints",
    "final_summary": "final_summary(designs, solver, ...): prints the final designs' numbers and returns "
                     "{name: values} of the last design, one for each RESULT_FIELDS entry: an array of "
                     "(elements,) or (nodes,), with a last axis of 3 for a vector",
}


def _checked(module: "ModuleType") -> "ModuleType":
    """Returns the module; raises TypeError naming every MODULE_ENTRIES entry it does not give."""
    missing = [k for k in MODULE_ENTRIES if not hasattr(module, k)]
    if missing:
        raise TypeError(f"physics module {module.__name__} does not give {', '.join(missing)}")
    return module


PHYSICS = {m.NAME: _checked(m) for m in MODULES}


def joined(what: str, core: dict[str, Any], modules: "list[ModuleType]",
           entries: "Callable[[ModuleType], dict[str, Any]]") -> dict[str, tuple[Any, list[str]]]:
    """{name: (entry, the names of the physics that give it)}: the core's entries (core: {name: entry}, every
    physics has them; tagged with none) and each module's (entries(module): {name: entry}) in one list, as the
    core's lists of properties, quantities and kinds of support and load are made. An entry several physics
    give the same way (the same object, or equal) is one entry tagged with all of them; two different
    entries under one name are an error naming both places. what names the kind of entry in that error.
    Entries are of any kind (Quantity, Attribute, the arguments of one as a dict), compared with is or ==."""
    out, where = {}, {}
    for name, entry in core.items():
        out[name], where[name] = (entry, []), "the core"
    for m in modules:
        for name, entry in entries(m).items():
            here = f"the {m.NAME} physics"
            if name not in out:
                out[name], where[name] = (entry, [m.NAME]), here
                continue
            have, tags = out[name]
            if not (have is entry or have == entry):
                raise ValueError(f"two different {what} are named {name!r}: one in {where[name]}, one in {here}. "
                                 f"Rename one of them, or give the same entry in both.")
            if where[name] != "the core" and m.NAME not in tags:
                tags.append(m.NAME)
    return out


def undefined_properties(modules: "list[ModuleType]", quantities: "list[Quantity]",
                         properties: "dict[str, Attribute]") -> None:
    """Raises, naming the place, when a physics' PROPERTIES or a quantity's needs names a material property
    that is not in properties (the joined list): every property is defined once, in the core or a physics."""
    named = [(f"the {m.NAME} physics' PROPERTIES", c) for m in modules for c in m.PROPERTIES] + \
        [(f"the quantity {q.name}", c) for q in quantities for c in q.needs]
    for place, column in named:
        if column not in properties:
            raise ValueError(f"{place} names the material property {column!r}, which neither the core nor any "
                             f"physics defines; add it to that physics' NEW_PROPERTIES")


def physics_module(name: str) -> "ModuleType":
    """The front page of physics name; an unknown name is an error that lists the known ones."""
    if name not in PHYSICS:
        raise ValueError(f"physics {name!r}: one of {', '.join(PHYSICS)}")
    return PHYSICS[name]

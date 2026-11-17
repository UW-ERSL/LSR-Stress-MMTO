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
presence (_checked) when PHYSICS is made, not for signatures. The entries in OPTIONAL_ENTRIES may be left
out: _checked then gives the module the "none" value listed there.
"""
from typing import TYPE_CHECKING, Any, Callable

from lsr_mmto.physics.structural import structural_module
from lsr_mmto.physics.thermal import thermal_module
from lsr_mmto.physics.thermoelastic import thermoelastic_module
if TYPE_CHECKING:
    from types import ModuleType

    from lsr_mmto.formulation.quantities import Quantity
    from lsr_mmto.materials.databases import Attribute

MODULES = [structural_module, thermal_module,thermoelastic_module]

# what a module's front page gives, and what each is
MODULE_ENTRIES = {
    "NAME": "the name a project file uses",
    "DOC": "what the physics is, in one line",
    "FIELDS": "each field it solves for and its unknowns per node, e.g. {'u': 3}; the mesh size and the "
              "supports and loads (one boundary condition per field) are counted from it",
    "UNIT_KINDS": "the kinds of quantity its values and quantities measure that the core does not have, "
                  "{kind: {'units': {unit: value in SI}, 'MKS': unit, 'mmKS': unit, 'IPS': unit}}",
    "PROPERTIES": "the material database columns every problem of it reads",
    "OPTIONAL_PROPERTIES": "the material database columns it reads when the database has them and replaces "
                           "otherwise (structural: Poisson's ratio, else one value from its settings); a new "
                           "database starts with these after PROPERTIES",
    "NEW_PROPERTIES": "the material properties it brings that the core does not have (the core has Density and "
                      "Cost), each as the arguments of an Attribute (materials/databases.py); a property another "
                      "physics brings too is one entry, shared, when both give it the same way; may be empty",
    "make_solver": "make_solver(mesh, mat_prop, bcs, linear_solver, dsolver, elem_body_force): its finite "
                   "element solver; bcs holds one boundary condition per field. The core also reads its .bc "
                   "(.bc.num_dofs), .mesh and .plot_mesh; a physics of several fields gives one field's",
    "analyse": "analyse(model): one iteration's solve inside the torch tree, from model.x_phys and model.props, "
               "leaving on the model what its quantities read",
    "SUPPORT_AND_LOAD_KINDS": "its kinds of support and load, {name: the arguments of a BCKind}, each naming "
                              "the field it acts on",
    "NEEDS": "what every project of it must have, as roles of its kinds: ('support', 'load') when a problem "
             "without a load has no meaning, ('support',) when held values alone make a problem",
    "deflation_space": "deflation_space(dsolver, mesh): builds the deflation space of each of its systems on "
                       "dsolver (one per field), so problems above the direct solver's limit use DPCG; None for a "
                       "physics without one, which then keeps the direct solver at every size",
    "SIMP_PENALTIES": "the SIMP exponents its solve uses, {name: (start, start with SIMP continuation, largest)}: "
                      "the run starts each and, with SIMP continuation on, raises it by 0.25 every 10 "
                      "iterations up to its largest, handing them to set_simp_penalties; the record keeps each "
                      "at the end as simp_<name>_penalty_at_end; may be empty",
    "set_simp_penalties": "set_simp_penalties(solver, values): puts each SIMP exponent ({name: value} of "
                          "SIMP_PENALTIES) where its solver reads it",
    "supports_and_loads_error": "supports_and_loads_error(mesh, bcs): why the supports and loads leave the "
                                "problem without a solution (the part free to move), or None",
    "QUANTITIES": "its objectives and constraints, each as the arguments of a Quantity",
    "PENALTIES": "the terms it adds to the objective besides those every physics has, each as the arguments "
                 "of a Penalty (design/latent_penalties.py), saying whether it reads the latent points; may be "
                 "empty",
    "SETTINGS": "its own settings, each as the arguments of a Setting",
    "RESULT_FIELDS": "the results a run saves, {name: dict(view, title, measures, where, shape)}: view the Results "
                     "panel's choice that shows it (not one of RESERVED_VIEWS), title its colour scale's, measures "
                     "what it is (for its unit), "
                     "where 'element' (one value per element, the default) or 'node' (one per node), shape "
                     "'scalar' (one number, the default) or 'vector' (x, y and z); final_summary returns their "
                     "values",
    "iteration_diagnostics": "iteration_diagnostics(history, iteration, m, x_phys, properties, ...): what each "
                             "iteration records and prints besides the objective and the constraints; m is the "
                             "model after its solve (its solver m.fe), live in the torch tree, so read .detach() "
                             "values only",
    "final_summary": "final_summary(designs, solver, ...): prints the final designs' numbers and returns "
                     "{name: values} of the last design, one for each RESULT_FIELDS entry: an array of "
                     "(elements,) or (nodes,), with a last axis of 3 for a vector",
    "SUMMARY_ROWS": "the numbers of its own a run's Summary and Compare show, {history key: dict(label, unit, "
                    "format)}: the key's last value (iteration_diagnostics records it), its row's label, its unit "
                    "as a quantity's unit is written (e.g. '{stress}', '' for none) and its format (default "
                    "'.4g'); a row a run did not record is left out",
}


def _always_held(mesh: Any, bcs: Any) -> None:
    """supports_and_loads_error of a physics that gives none: every problem is held."""
    return None


def _records_nothing(*args: Any, **kwargs: Any) -> None:
    """iteration_diagnostics of a physics that gives none: nothing recorded or printed."""
    return None


def _sets_nothing(solver: Any, values: dict[str, float]) -> None:
    """set_simp_penalties of a physics that gives none: nothing is set, so a solver that reads one of PyTO-AD's
    penalties reads whatever it holds, which another run in the same program may have changed."""
    return None


# the entries a front page may leave out, and the value it then has: "none" of each
OPTIONAL_ENTRIES = {
    "OPTIONAL_PROPERTIES": lambda: [],
    "UNIT_KINDS": lambda: {},
    "SETTINGS": lambda: [],
    "PENALTIES": lambda: [],
    "SUMMARY_ROWS": lambda: {},
    "deflation_space": lambda: None,
    "SIMP_PENALTIES": lambda: {},
    "set_simp_penalties": lambda: _sets_nothing,
    "supports_and_loads_error": lambda: _always_held,
    "iteration_diagnostics": lambda: _records_nothing,
}


# the Results panel's views every run has (gui/draw_design.py, views): a result shown as one of them would be hidden
RESERVED_VIEWS = ("Material", "Density", "Solid")


def _checked(module: "ModuleType") -> "ModuleType":
    """Returns the module, given the "none" value of each OPTIONAL_ENTRIES entry it leaves out; raises
    TypeError naming every other MODULE_ENTRIES entry it does not give, and ValueError for a result shown
    as one of RESERVED_VIEWS."""
    missing = [k for k in MODULE_ENTRIES if not hasattr(module, k) and k not in OPTIONAL_ENTRIES]
    if missing:
        raise TypeError(f"physics module {module.__name__} does not give {', '.join(missing)}")
    if hasattr(module, "SIMP_PENALTIES") != hasattr(module, "set_simp_penalties"):    # one is no use alone
        raise TypeError(f"physics module {module.__name__} gives one of SIMP_PENALTIES and set_simp_penalties: "
                        f"give both, or neither")
    for name, field in module.RESULT_FIELDS.items():
        if field.get("view") in RESERVED_VIEWS:
            raise ValueError(f"physics module {module.__name__}: the result {name!r} is shown as "
                             f"{field['view']!r}, a view every run already has ({', '.join(RESERVED_VIEWS)}); "
                             f"give it another view")
    for k, none in OPTIONAL_ENTRIES.items():
        if not hasattr(module, k):
            setattr(module, k, none())
    return module


PHYSICS = {m.NAME: _checked(m) for m in MODULES}


def one_each(what: str, module: "ModuleType", items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """{name: item} of a module's list of entries (each the arguments of one, with a "name"), for joined;
    ValueError when two of them share a name, which the dict would otherwise keep only the last of."""
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        if item["name"] in out:
            raise ValueError(f"the {module.NAME} physics gives two {what} named {item['name']!r}; rename one of them")
        out[item["name"]] = item
    return out


def joined(what: str, core: dict[str, Any], modules: "list[ModuleType]",
           entries: "Callable[[ModuleType], dict[str, Any]]") -> dict[str, tuple[Any, list[str]]]:
    """{name: (entry, the names of the physics that give it)}: the core's entries (core: {name: entry}, every
    physics has them; tagged with none) and each module's (entries(module): {name: entry}) in one list, as the
    core's lists of properties, quantities, penalties and kinds of support and load are made. An entry several physics
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
                fix = (f"rename the one in {here}" if where[name] == "the core" else
                       "rename one of them, or give the same entry in both")
                raise ValueError(f"two different {what} are named {name!r}: one in {where[name]}, one in {here}; "
                                 f"{fix}")
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

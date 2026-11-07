"""
quantities.py
=============
Every objective and constraint a formula can name, by name: QUANTITIES, find_quantity and
quantity_label. The quantities every physics has (mass, cost, volume fraction, material volume
fraction, mass fraction) are computed here; each physics module gives its own (the structural ones are in
physics/structural/structural_quantities.py), listed after these.
"""
import numpy as np
import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path (and its float64 default)
from lsr_mmto.materials.databases import once_per_file_version
from lsr_mmto.physics.available_physics import MODULES

DEFAULT_TAU = 0.5        # sharpness of the material assignment, the same for every material volume fraction


def material_volume_fraction(xb, z, z_real, material_id, tau):
    """V_k = (1/N) sum_e xb_e * softmax_k(-||z_e - z_k||^2 / tau)."""
    logits = -((z[:, None, :] - z_real[None, :, :]) ** 2).sum(-1) / tau
    p = torch.softmax(logits, dim=1)
    return torch.sum(xb * p[:, material_id]) / xb.shape[0]


# ----------------------------------------------------------------------------- quantities
# Every objective and constraint is one entry of QUANTITIES. One that every physics has is written
# here; one of a physics is written in its module (QUANTITIES on its front page, as the arguments
# below without physics). Write its function of the model's current state (m.x_phys, m.z, m.props,
# and what the physics' analyse leaves, e.g. m.u, m.svm) in torch, add the entry, and add it to
# tests/test_ad_quantities.py. Nothing else changes: the gradient comes from autograd.
#
#   name         the name a project file uses (P1); PyTO-AD's name where the meaning is the same
#   objective    True if it can be the objective
#   constraint   True if it can be a constraint
#   physics      the physics it applies to; empty: every physics
#   needs        material database columns it reads (checked, with a clear error)
#   fn(m, opt)   the value, a torch scalar; a constraint is fn / limit - 1 <= 0
#   doc          what fn computes, for the settings page
#   shown        its name in the app
#   help         what it is, in plain words: the app's tooltip and the settings page
#   unit         the unit of a bound on it, as the app shows it: names of the unit system in braces are
#                filled in ("{force}*{length}"); text without braces is shown as it is ("no units")
#   scales_with  how its value follows a database column given in the wrong unit: {column: power}; a column
#                whose numbers are k times what they should be makes the value k**power times its own
#                (mass {"Density": 1}, compliance {"Youngs_Modulus": -1}); the project's unit warnings read it
#   arguments    what its formula takes, in order, as (name, type, default) or (name, type, default, words):
#                type "material" (a material's row in the database, a whole number) or "number"; one without
#                a default is required and written by position, one with a default may be written by name:
#                Name(2, tau=0.5). words: the MATERIAL_WORDS a material argument also takes in place of a
#                row, written as the word: MassFraction(lightest)
#   sets_start_density  True for a constraint whose bound is the density every element starts the run at
#                (the volume fraction); the first such constraint sets it, else every element starts at 0.5
#   bound_warning  bound_warning(project, bound): a warning about a constraint's limit before a run (the
#                project's warnings show it), or None; e.g. a mass limit too low for any part to meet
#   start_bound  the limit a new constraint row in the Formulation panel starts at: a number, or a function of
#                the project giving one (None when it cannot tell); without it a new row starts at 0
#   start_note   how that starting limit is worked out, in plain words, for the row's (i) and the limit's tip;
#                {reference} in it is filled with start_reference(project), in the quantity's unit
#   start_reference  start_reference(project): the number start_note's {reference} names, or None


class Quantity:
    __slots__ = ("name", "objective", "constraint", "physics", "needs", "fn", "doc", "shown", "help", "unit",
                 "arguments", "scales_with", "sets_start_density", "bound_warning", "start_bound", "start_note",
                 "start_reference")

    def __init__(self, name, objective, constraint, physics, needs, fn, doc, shown, help, unit="", arguments=(),
                 scales_with=None, sets_start_density=False, bound_warning=None, start_bound=None, start_note="",
                 start_reference=None):
        self.name, self.objective, self.constraint = name, objective, constraint
        self.physics, self.needs, self.fn, self.doc = frozenset(physics), tuple(needs), fn, doc
        self.shown, self.help, self.unit = shown, help, unit
        self.arguments = tuple((a[0], a[1], a[2], tuple(a[3]) if len(a) > 3 else ()) for a in arguments)
        self.scales_with, self.sets_start_density = dict(scales_with or {}), bool(sets_start_density)
        self.bound_warning, self.start_bound = bound_warning, start_bound
        self.start_note, self.start_reference = start_note, start_reference

    def takes(self, argument):
        """True when the formula takes the argument of that name or type ("material", "tau")."""
        return any(argument in (name, typ) for name, typ, _, _ in self.arguments)

    def words(self, argument):
        """The MATERIAL_WORDS the argument of that name or type takes in place of a material's row."""
        return next((words for name, typ, _, words in self.arguments if argument in (name, typ)), ())

    def first_bound(self, project):
        """The limit a new constraint on it starts at (start_bound), or None when it declares none or the
        project cannot give it."""
        bound = self.start_bound
        if callable(bound):
            try:
                bound = bound(project)
            except Exception:                                                          # noqa: BLE001
                bound = None
        return None if bound is None else float(bound)

    def start_text(self, project):
        """Where a new constraint's starting limit comes from, in one line ("Starting limit 0.0035 kg: 40% of
        ... (0.00875 kg). Type your own limit to change it."), or None when it declares no start_note or the
        project cannot give the limit."""
        bound = self.first_bound(project)
        if bound is None or not self.start_note:
            return None
        try:
            unit = project.quantity_unit(self)
        except KeyError:                                                               # no unit system
            unit = ""
        note = self.start_note
        if "{reference}" in note:
            try:
                reference = self.start_reference(project)
            except Exception:                                                          # noqa: BLE001
                reference = None
            if reference is None:
                return None
            note = note.format(reference=f"{reference:.3g}{unit}")
        return f"Starting limit {bound:.3g}{unit}: {note}. Type your own limit to change it."

    def applies_to(self, physics):
        """True for a quantity of that physics, or one every physics has."""
        return not self.physics or physics in self.physics


# Words a material argument may take in place of a material's row, if it declares them: the word, how the
# app shows it, and the row it means, picked from each material's value of a database column.
MATERIAL_WORDS = {
    "heaviest": ("the heaviest", "Density", np.argmax),
    "lightest": ("the lightest", "Density", np.argmin),
}


def material_row(value, values_of):
    """The database row a material argument means: the row itself, or the row a material word picks from
    values_of(column), each material's value of that column, in the database's order."""
    if isinstance(value, str):
        _, column, pick = MATERIAL_WORDS[value]
        return int(pick(np.asarray(values_of(column), dtype=float)))
    return int(value)


def material_text(value, material_labels=None):
    """A material argument as the app shows it: the material's label ('Ti-6Al-4V'), or 'the lightest'."""
    if isinstance(value, str):
        return MATERIAL_WORDS[value][0] if value in MATERIAL_WORDS else value
    i = int(value)
    return material_labels[i] if material_labels is not None and 0 <= i < len(material_labels) else f"material {i}"


def database_values(m, column):
    """Each database material's value of a column, as the database gives it (not through the network)."""
    a = m.enc.materialAttributes[column]
    return np.asarray(m.enc.rawData, dtype=float)[:, a["idx"]]


def _mass(m, opt):
    return torch.sum(m.props["Density"] * m.x_phys) * m.elem_volume


def _mass_fraction(m, opt):
    """mass / (reference density * design domain volume); the reference density is the database's, a constant."""
    k = material_row(opt["material_id"], lambda column: database_values(m, column))
    reference = float(database_values(m, "Density")[k])
    return _mass(m, opt) / (reference * m.x_phys.shape[0] * m.elem_volume)


@once_per_file_version
def part_volume(stl):
    """The volume of the part, its STL's, in the model's length unit cubed."""
    import pyvista
    return abs(float(pyvista.read(stl).volume))


def solid_part_mass(project, material):
    """The mass of the whole part (its STL's volume) solid in one material of the project's database, in the
    unit Mass is in: material is its row or a material word ("lightest"). None when the part or the database
    cannot be read or the database has no Density column."""
    from lsr_mmto.materials.databases import database_table
    try:
        _, attributes, _, values = database_table(project.database_paths()[0])
        values_of = lambda column: values[:, attributes.index(column)]                 # noqa: E731
        return float(values_of("Density")[material_row(material, values_of)]) * part_volume(project.stl_path())
    except Exception:                                                                  # noqa: BLE001
        return None


LOW_MASS_SHARE = 0.01        # a mass limit below this share of the lightest solid part is warned about


def _mass_limit_warning(project, bound):
    """A mass limit below 1 % of the part solid in the lightest material: a design this light is barely
    there, and usually the limit was typed in the wrong unit."""
    lightest = solid_part_mass(project, "lightest")
    if not lightest or not 0 < bound < LOW_MASS_SHARE * lightest:
        return None
    return f"This limit allows at most {100 * bound / lightest:.2g}% of the part even in the lightest material."


def _lightest_solid_part(project):
    """The mass of the whole part solid in the database's lightest material, or None."""
    return solid_part_mass(project, "lightest")


def _mass_start_bound(project):
    """40 % of the part solid in the lightest material: a limit any design can meet, far above the warning's."""
    lightest = _lightest_solid_part(project)
    return None if lightest is None else 0.4 * lightest


def _cost(m, opt):
    return torch.sum(m.props["Density"] * m.props["Cost"] * m.x_phys) * m.elem_volume


def _volume_fraction(m, opt):
    return torch.mean(m.x_phys)


def _material_volume_fraction(m, opt):
    return material_volume_fraction(m.x_phys, m.z, m.z_real,
                                    int(opt["material_id"]), float(opt.get("tau", DEFAULT_TAU)))


CORE_QUANTITIES = [
    Quantity("Mass", True, True, [], ["Density"],
             _mass, "sum of density * x_phys * element volume", "Mass",
             "The mass of the part, from each material's density.", "{mass}", scales_with={"Density": 1},
             bound_warning=_mass_limit_warning, start_bound=_mass_start_bound,
             start_note="40% of the mass of the whole part made solid in its lightest material ({reference})",
             start_reference=_lightest_solid_part),
    Quantity("Cost", False, True, [], ["Density", "Cost"],
             _cost, "sum of density * cost * x_phys * element volume", "Cost",
             "The cost of the part, from each material's density and cost per mass.", "USD",
             scales_with={"Cost": 1, "Density": 1}),
    Quantity("VolumeFraction", False, True, [], [],
             _volume_fraction, "mean of x_phys", "Volume fraction",
             "The share of the design space filled with material, 0 to 1.", "fraction of the design domain",
             sets_start_density=True, start_bound=0.4, start_note="40% of the design space filled"),
    Quantity("MaterialVolumeFraction", False, True, [], [],
             _material_volume_fraction, "volume fraction of one database material, softmax weights",
             "Material volume fraction", "The share of the design space filled with one chosen material, 0 to 1.",
             "fraction of the design domain", (("material_id", "material", None), ("tau", "number", DEFAULT_TAU)),
             start_bound=0.4, start_note="40% of the design space filled with that material"),
    Quantity("MassFraction", False, True, [], ["Density"],
             _mass_fraction, "mass / (reference density * design domain volume), reference from the database",
             "Mass fraction",
             "The part's mass compared with the whole part made solid in one material: 0.4 allows 40 % of that "
             "mass. Choose the material, or the heaviest or the lightest of the database.",
             "of a solid part in that material", (("material_id", "material", None, ("heaviest", "lightest")),),
             start_bound=0.4, start_note="40% of the mass of the whole part made solid in that material"),
]


def quantities_of(modules):
    """Every quantity: each module's (its QUANTITIES, as the arguments of a Quantity), then the core's, each
    once (available_physics.joined): a quantity several physics give the same way is one, tagged with all of
    them. Every material property a physics or a quantity names must be defined (undefined_properties)."""
    from lsr_mmto.materials.databases import attributes_of
    from lsr_mmto.physics.available_physics import joined, undefined_properties
    entries = joined("quantities", {q.name: q for q in CORE_QUANTITIES}, modules,
                     lambda m: {q["name"]: q for q in m.QUANTITIES})
    out = [Quantity(physics=tags, **entry) for entry, tags in entries.values() if not isinstance(entry, Quantity)] + \
        CORE_QUANTITIES
    undefined_properties(modules, out, attributes_of(modules))
    return out


QUANTITIES = quantities_of(MODULES)


def find_quantity(name):
    """The registry entry of a quantity, by the name a project file uses."""
    for q in QUANTITIES:
        if q.name == name:
            return q
    raise KeyError(f"unknown quantity {name!r}; known: {', '.join(q.name for q in QUANTITIES)}")


def takes(name, argument):
    """True when the quantity of that name takes the argument (by name or type); False for an unknown name."""
    return any(q.name == name and q.takes(argument) for q in QUANTITIES)


def default_of(name, argument):
    """The default of a quantity's argument (the tau of a material volume fraction)."""
    return next(d for q in QUANTITIES if q.name == name for a, _, d, _ in q.arguments if a == argument)


def quantity_label(name, opt=None, material_labels=None):
    """A quantity's name as the app shows it, with the material for a quantity of one material:
    'Material volume fraction of Ti-6Al-4V', 'Mass fraction of the lightest'."""
    shown = next((q.shown for q in QUANTITIES if q.name == name), str(name))
    if isinstance(opt, dict) and "material_id" in opt:
        shown += f" of {material_text(opt['material_id'], material_labels)}"
    return shown

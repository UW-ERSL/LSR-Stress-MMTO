"""
quantities.py
=============
Every objective and constraint a formula can name, by name: QUANTITIES, find_quantity and
quantity_label. The quantities every physics has (mass, cost, volume fraction, material volume
fraction, mass fraction) are computed here; each physics module gives its own (the structural ones are in
physics/structural/structural_quantities.py), listed after these.
"""
from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np
import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path (and its float64 default)
from lsr_mmto.materials.databases import once_per_file_version
from lsr_mmto.physics.available_physics import MODULES

if TYPE_CHECKING:
    from types import ModuleType

    from lsr_mmto.optimize.design_model import MMTOModelAD
    from lsr_mmto.optimize.solid_part_analysis import _Analysed
    from lsr_mmto.problem.project import Project

DEFAULT_TAU = 0.5        # sharpness of the material assignment, the same for every material volume fraction

# ----------------------------------------------------------------------------- at most or at least
# A constraint is "<=" (at most its limit) or ">=" (at least its limit), for any quantity of any physics. The
# optimizer sees one g that is met when g <= 0, so the two are written g = (value - limit) / |limit| ("<=") and
# g = (limit - value) / |limit| (">="): for a positive limit, value / limit - 1 and 1 - value / limit, and right
# for a negative one too. The run's record keeps g; what a person sees is value / limit, from g, op and limit.
OPS = ("<=", ">=")
SHOWN_OPS = {"<=": "≤", ">=": "≥"}
AT_MOST_SHARE = 0.4      # the 40% rule: an "at most" limit starts at 40% of its quantity's reference
AT_LEAST_SHARE = 0.1     # the 10% rule: an "at least" limit starts at 10% of the same reference


def op_of(spec: "list | tuple") -> str:
    """The op of a constraint as a run holds it, [name, options, limit, op]; "<=" for one recorded before ">="
    was there ([name, options, limit])."""
    return spec[3] if len(spec) > 3 and spec[3] in OPS else "<="


def constraint_g(value, op: str, limit: float):
    """The g the optimizer sees, met when g <= 0: (value - limit) / |limit| for "<=", (limit - value) / |limit|
    for ">=" (value / limit - 1 and 1 - value / limit for a positive limit). value may be a torch tensor, its
    gradient following."""
    return (value - limit) / abs(limit) if op == "<=" else (limit - value) / abs(limit)


def share_of_limit(g, op: str, limit: float = 1.0):
    """value / limit, from a recorded g (a number or an array), its op and the sign of its limit."""
    sign = 1.0 if limit >= 0 else -1.0
    return 1.0 + sign * g if op == "<=" else 1.0 - sign * g


def material_volume_fraction(xb: torch.Tensor, z: torch.Tensor, z_real: torch.Tensor, material_id: int,
                             tau: float) -> torch.Tensor:
    """V_k = (1/N) sum_e xb_e * softmax_k(-||z_e - z_k||^2 / tau). xb (N,), z (N, latentDim), z_real
    (materials, latentDim); k = material_id, a row of the database."""
    logits = -((z[:, None, :] - z_real[None, :, :]) ** 2).sum(-1) / tau
    p = torch.softmax(logits, dim=1)
    return torch.sum(xb * p[:, material_id]) / xb.shape[0]


# ----------------------------------------------------------------------------- quantities
# Every objective and constraint is one entry of QUANTITIES. One that every physics has is written
# here; one of a physics is written in its module (QUANTITIES on its front page, as the arguments
# below without physics). Write its function of the model's current state (m.x_phys, m.z, m.props,
# and what the physics' analyse leaves, e.g. m.u, m.svm) in torch, add the entry, and add it to
# tests/gradients/test_ad_quantities.py. Nothing else changes: the gradient comes from autograd.
#
#   name         the name a project file uses (P1); PyTO-AD's name where the meaning is the same
#   objective    True if it can be the objective
#   constraint   True if it can be a constraint
#   physics      the physics it applies to; empty: every physics
#   needs        material database columns it reads (checked, with a clear error)
#   fn(m, opt)   the value, a torch scalar; a constraint is fn <= limit or fn >= limit (constraint_g)
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
#                (the volume fraction); the first such constraint sets it, at most or at least, else every
#                element starts at 0.5
#   bound_warning  bound_warning(project, bound): a warning about an "at most" limit before a run (the
#                project's warnings show it), or None; e.g. a mass limit too low for any part to meet
#   start_reference  what a new constraint row's limit is a share of: a number (1.0 for a fraction of the
#                design space), or start_reference(project) giving one (None when it cannot tell). A row
#                starts at AT_MOST_SHARE of it for "<=" (the 40% rule) and AT_LEAST_SHARE for ">=" (the 10%
#                rule), and moves with it while the limit is the one it started at (restarted_limits)
#   start_bound  a fixed "<=" starting limit, for a quantity whose start is no share of anything (a failure
#                factor of 1); a ">=" row of it keeps the limit it has. Without either a new row starts at 0
#   start_note   how the starting limit is worked out, in plain words, for the row's (i) and the limit's tip;
#                {share} is filled with the share in % ("40%"), {reference} with the reference in the
#                quantity's unit, {material} with the material it is worked out in (design_material_text)
#   start_mix    what {material} says for a mix of materials (the reference's own choice): "its lightest
#                material" (the default), "its cheapest material"
#   reads_latent_points  True when fn reads the latent points (m.z, m.z_real); a single-material project
#                (every element one material: Project.single_material) refuses it. Default False


class Quantity:
    """One objective or constraint a formula can name, as the comment above describes its arguments."""
    __slots__ = ("name", "objective", "constraint", "physics", "needs", "fn", "doc", "shown", "help", "unit",
                 "arguments", "scales_with", "sets_start_density", "bound_warning", "start_bound", "start_note",
                 "start_reference", "start_mix", "reads_latent_points")

    name: str
    objective: bool
    constraint: bool
    physics: frozenset[str]
    needs: tuple[str, ...]
    fn: "Callable[[MMTOModelAD | _Analysed, dict[str, int | float | str] | None], torch.Tensor]"
    doc: str
    shown: str
    help: str
    unit: str
    arguments: tuple[tuple[str, str, float | None, tuple[str, ...]], ...]     # (name, type, default, words)
    scales_with: dict[str, float]
    sets_start_density: bool
    bound_warning: "Callable[[Project, float], str | None] | None"
    start_bound: "float | Callable[[Project], float | None] | None"
    start_note: str
    start_reference: "float | Callable[[Project], float | None] | None"
    start_mix: str
    reads_latent_points: bool

    def __init__(self, name: str, objective: bool, constraint: bool, physics: Iterable[str], needs: Iterable[str],
                 fn: "Callable[[MMTOModelAD | _Analysed, dict[str, int | float | str] | None], torch.Tensor]",
                 doc: str, shown: str, help: str,
                 unit: str = "", arguments: Iterable[tuple] = (), scales_with: dict[str, float] | None = None,
                 sets_start_density: bool = False,
                 bound_warning: "Callable[[Project, float], str | None] | None" = None,
                 start_bound: "float | Callable[[Project], float | None] | None" = None, start_note: str = "",
                 start_reference: "float | Callable[[Project], float | None] | None" = None,
                 start_mix: str = "its lightest material", reads_latent_points: bool = False) -> None:
        self.name, self.objective, self.constraint = name, objective, constraint
        self.physics, self.needs, self.fn, self.doc = frozenset(physics), tuple(needs), fn, doc
        self.shown, self.help, self.unit = shown, help, unit
        self.arguments = tuple((a[0], a[1], a[2], tuple(a[3]) if len(a) > 3 else ()) for a in arguments)
        self.scales_with, self.sets_start_density = dict(scales_with or {}), bool(sets_start_density)
        self.bound_warning, self.start_bound = bound_warning, start_bound
        self.start_note, self.start_reference, self.start_mix = start_note, start_reference, start_mix
        self.reads_latent_points = bool(reads_latent_points)

    def takes(self, argument: str) -> bool:
        """True when the formula takes the argument of that name or type ("material", "tau")."""
        return any(argument in (name, typ) for name, typ, _, _ in self.arguments)

    def words(self, argument: str) -> tuple[str, ...]:
        """The MATERIAL_WORDS the argument of that name or type takes in place of a material's row."""
        return next((words for name, typ, _, words in self.arguments if argument in (name, typ)), ())

    def reference(self, project: "Project") -> float | None:
        """What its starting limit is a share of (start_reference), or None when it declares none or the
        project cannot give it."""
        ref = self.start_reference
        if callable(ref):
            try:
                ref = ref(project)
            except Exception:                                                          # noqa: BLE001
                ref = None
        return None if ref is None else float(ref)

    def first_bound(self, project: "Project", op: str = "<=") -> float | None:
        """The limit a new constraint on it starts at: AT_MOST_SHARE of its reference for "<=", AT_LEAST_SHARE
        for ">=" (the 40% and 10% rules), else its fixed start_bound for "<="; None when it has neither, or the
        project cannot give it."""
        if self.start_reference is not None:
            ref = self.reference(project)
            return None if ref is None else (AT_MOST_SHARE if op == "<=" else AT_LEAST_SHARE) * ref
        if self.start_bound is None or op != "<=":
            return None
        bound = self.start_bound
        if callable(bound):
            try:
                bound = bound(project)
            except Exception:                                                          # noqa: BLE001
                bound = None
        return None if bound is None else float(bound)

    def start_text(self, project: "Project", op: str = "<=") -> str | None:
        """Where a constraint's starting limit comes from, in one line ("Starting limit 0.0035 kg: 40% of the
        mass of the whole part made solid in its lightest material (0.00875 kg). Type your own limit to change
        it."), or None when it declares no start_note or the project cannot give the limit."""
        bound = self.first_bound(project, op)
        if bound is None or not self.start_note:
            return None
        try:
            unit = project.quantity_unit(self)
        except KeyError:                                                               # no unit system
            unit = ""
        note = self.note(project, op, unit)
        rule = ("" if self.start_reference is None else
                f" An at-most limit starts at {100 * AT_MOST_SHARE:g}% and an at-least limit at "
                f"{100 * AT_LEAST_SHARE:g}%.")
        return f"Starting limit {bound:.3g}{unit}: {note}.{rule} Type your own limit to change it."

    def note(self, project: "Project", op: str = "<=", unit: str = "") -> str:
        """start_note filled in for the project: the share, the material, the reference with unit."""
        share = AT_MOST_SHARE if op == "<=" else AT_LEAST_SHARE
        reference = self.reference(project)
        return self.start_note.format(share=f"{100 * share:g}%", material=design_material_text(project, self.start_mix),
                                      reference="" if reference is None else f"{reference:.3g}{unit}")

    def applies_to(self, physics: str) -> bool:
        """True for a quantity of that physics, or one every physics has."""
        return not self.physics or physics in self.physics


# Words a material argument may take in place of a material's row, if it declares them: the word, how the
# app shows it, and the row it means, picked from each material's value of a database column.
MATERIAL_WORDS = {
    "heaviest": ("the heaviest", "Density", np.argmax),
    "lightest": ("the lightest", "Density", np.argmin),
}


def material_row(value: int | float | str, values_of: Callable[[str], np.ndarray]) -> int:
    """The database row a material argument means: the row itself, or the row a material word picks from
    values_of(column), each material's value of that column, in the database's order."""
    if isinstance(value, str):
        _, column, pick = MATERIAL_WORDS[value]
        return int(pick(np.asarray(values_of(column), dtype=float)))
    return int(value)


def material_text(value: int | float | str, material_labels: list[str] | None = None) -> str:
    """A material argument as the app shows it: the material's label ('Ti-6Al-4V'), or 'the lightest'."""
    if isinstance(value, str):
        return MATERIAL_WORDS[value][0] if value in MATERIAL_WORDS else value
    i = int(value)
    return material_labels[i] if material_labels is not None and 0 <= i < len(material_labels) else f"material {i}"


def database_values(m: "MMTOModelAD | _Analysed", column: str) -> np.ndarray:
    """Each database material's value of a column, as the database gives it (not through the network)."""
    a = m.enc.materialAttributes[column]
    return np.asarray(m.enc.rawData, dtype=float)[:, a["idx"]]


def _mass(m: "MMTOModelAD | _Analysed", opt: dict[str, int | float | str] | None) -> torch.Tensor:
    """sum of density * x_phys * element volume, a torch scalar."""
    return torch.sum(m.props["Density"] * m.x_phys) * m.elem_volume


def _mass_fraction(m: "MMTOModelAD | _Analysed", opt: dict[str, int | float | str]) -> torch.Tensor:
    """mass / (reference density * design domain volume); the reference density is the database's, a constant."""
    k = material_row(opt["material_id"], lambda column: database_values(m, column))
    reference = float(database_values(m, "Density")[k])
    return _mass(m, opt) / (reference * m.x_phys.shape[0] * m.elem_volume)


@once_per_file_version
def part_volume(stl: str) -> float:
    """The volume of the part, its STL's, in the model's length unit cubed."""
    import pyvista
    return abs(float(pyvista.read(stl).volume))


def solid_part_mass(project: "Project", material: int | str) -> float | None:
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


def design_material(project: "Project") -> int | None:
    """The material the design is made of when it is one: its row in the database, with Single material (the
    material chosen, or the database's only one) or a material fixed with Multi material; None for a mix of
    materials, or one that cannot be read."""
    from lsr_mmto.materials.databases import database_table
    if not project.single_material():
        return None
    try:
        names = [str(n).strip() for n in database_table(project.database_paths()[0])[0]]
    except Exception:                                                                  # noqa: BLE001
        return None
    if project.pinned_material is None:
        return 0 if len(names) == 1 else None
    return names.index(str(project.pinned_material)) if str(project.pinned_material) in names else None


def design_material_text(project: "Project", mix: str = "its lightest material") -> str:
    """The material a starting limit is worked out in, in words: "its material, Ti-6Al-4V" for a design of one
    material, else mix (the lightest, the cheapest)."""
    from lsr_mmto.materials.databases import material_display
    row = design_material(project)
    if row is None:
        return mix
    try:
        label = material_display(project.database_paths()[0])[1][row]
    except Exception:                                                                  # noqa: BLE001
        label = f"material {row}"
    return f"its material, {label}"


def _solid_part(project: "Project", per_volume: Callable[[Callable[[str], np.ndarray]], np.ndarray],
                pick: Callable[[np.ndarray], int]) -> float | None:
    """per_volume(values_of) (each material's amount per volume, from its column values) times the part's
    volume, for the design's material (design_material), else the material pick chooses from those amounts.
    None when the part or the database cannot be read or lacks a column."""
    from lsr_mmto.materials.databases import database_table
    try:
        _, attributes, _, values = database_table(project.database_paths()[0])
        amounts = np.asarray(per_volume(lambda column: values[:, attributes.index(column)]), dtype=float)
        row = design_material(project)
        return float(amounts[pick(amounts) if row is None else row]) * part_volume(project.stl_path())
    except Exception:                                                                  # noqa: BLE001
        return None


def _design_solid_mass(project: "Project") -> float | None:
    """The mass of the whole part solid in the design's material, or in the lightest of a mix."""
    return _solid_part(project, lambda v: v("Density"), np.argmin)


def _design_solid_cost(project: "Project") -> float | None:
    """The cost of the whole part solid in the design's material, or in the cheapest of a mix (the least
    density times cost per mass)."""
    return _solid_part(project, lambda v: v("Density") * v("Cost"), np.argmin)


LOW_MASS_SHARE = 0.01        # a mass limit below this share of the solid part is warned about


def _mass_limit_warning(project: "Project", bound: float) -> str | None:
    """A mass limit below 1 % of the part solid in the design's material (the lightest of a mix): a design this
    light is barely there, and usually the limit was typed in the wrong unit."""
    solid = _design_solid_mass(project)
    if not solid or not 0 < bound < LOW_MASS_SHARE * solid:
        return None
    if design_material(project) is None:
        return f"This limit allows at most {100 * bound / solid:.2g}% of the part even in the lightest material."
    return (f"This limit allows at most {100 * bound / solid:.2g}% of the part made solid in "
            f"{design_material_text(project)}.")


def _constraint_quantity(c: dict) -> "Quantity | None":
    """The quantity a constraint names, or None for one that cannot be read (its error is said elsewhere)."""
    from lsr_mmto.formulation.formulation_text import parse_expression
    try:
        return find_quantity(parse_expression(c["expression"])[0])
    except (ValueError, KeyError, TypeError):
        return None


def starting_limits(project: "Project") -> list[float | None]:
    """The limit each constraint of the project would start at now (its quantity's rule, for its op), None for
    one with no rule that follows the materials. Taken as Materials opens, before any change to the database's
    file, for restarted_limits."""
    out = []
    for c in project.formulation.get("constraints", []):
        q, op = _constraint_quantity(c), c.get("op", "<=")
        out.append(q.first_bound(project, op) if q is not None and callable(q.start_reference) and op in OPS
                   else None)
    return out


def restarted_limits(starts: list[float | None], after: "Project") -> list[str]:
    """The constraint limits of after still at the limit they started at (starts: starting_limits as the
    materials were, taken before they changed) set to the starting limit for after's materials, the database's
    values now included, with a line for each; a limit you typed stays as it is. Called when the materials of
    the design change (Materials)."""
    import math
    lines = []
    for c, old in zip(after.formulation.get("constraints", []), starts):
        q, op = _constraint_quantity(c), c.get("op", "<=")
        if q is None or old is None or op not in OPS:
            continue
        new = q.first_bound(after, op)
        if new is None or c.get("bound") is None or not math.isclose(c["bound"], old, rel_tol=1e-12):
            continue
        if not math.isclose(new, old, rel_tol=1e-12):
            c["bound"] = new
            try:
                unit = after.quantity_unit(q)
            except KeyError:                                                           # no unit system
                unit = ""
            lines.append(f"Starting limit of {q.shown} {op} now {new:.3g}{unit}: {q.note(after, op, unit)}")
    return lines


def _cost(m: "MMTOModelAD | _Analysed", opt: dict[str, int | float | str] | None) -> torch.Tensor:
    """sum of density * cost * x_phys * element volume, a torch scalar."""
    return torch.sum(m.props["Density"] * m.props["Cost"] * m.x_phys) * m.elem_volume


def _volume_fraction(m: "MMTOModelAD | _Analysed", opt: dict[str, int | float | str] | None) -> torch.Tensor:
    """The mean of x_phys, a torch scalar."""
    return torch.mean(m.x_phys)


def _material_volume_fraction(m: "MMTOModelAD | _Analysed", opt: dict[str, int | float | str]) -> torch.Tensor:
    """material_volume_fraction of the material opt["material_id"], tau opt["tau"] or DEFAULT_TAU. It reads the
    latent points, so at a solid part (_Analysed) it raises NoLatentPoints."""
    return material_volume_fraction(m.x_phys, m.z, m.z_real,
                                    int(opt["material_id"]), float(opt.get("tau", DEFAULT_TAU)))


CORE_QUANTITIES = [
    Quantity("Mass", True, True, [], ["Density"],
             _mass, "sum of density * x_phys * element volume", "Mass",
             "The mass of the part, from each material's density.", "{mass}", scales_with={"Density": 1},
             bound_warning=_mass_limit_warning, start_reference=_design_solid_mass,
             start_note="{share} of the mass of the whole part made solid in {material} ({reference})"),
    Quantity("Cost", False, True, [], ["Density", "Cost"],
             _cost, "sum of density * cost * x_phys * element volume", "Cost",
             "The cost of the part, from each material's density and cost per mass.", "USD",
             scales_with={"Cost": 1, "Density": 1}, start_reference=_design_solid_cost,
             start_note="{share} of the cost of the whole part made solid in {material} ({reference})",
             start_mix="its cheapest material"),
    Quantity("VolumeFraction", False, True, [], [],
             _volume_fraction, "mean of x_phys", "Volume fraction",
             "The share of the design space filled with material, 0 to 1.", "fraction of the design domain",
             sets_start_density=True, start_reference=1.0, start_note="{share} of the design space filled"),
    Quantity("MaterialVolumeFraction", False, True, [], [],
             _material_volume_fraction, "volume fraction of one database material, softmax weights",
             "Material volume fraction", "The share of the design space filled with one chosen material, 0 to 1.",
             "fraction of the design domain", (("material_id", "material", None), ("tau", "number", DEFAULT_TAU)),
             start_reference=1.0, start_note="{share} of the design space filled with that material",
             reads_latent_points=True),
    Quantity("MassFraction", False, True, [], ["Density"],
             _mass_fraction, "mass / (reference density * design domain volume), reference from the database",
             "Mass fraction",
             "The part's mass compared with the whole part made solid in one material: 0.4 allows 40 % of that "
             "mass. Choose the material, or the heaviest or the lightest of the database.",
             "of a solid part in that material", (("material_id", "material", None, ("heaviest", "lightest")),),
             start_reference=1.0, start_note="{share} of the mass of the whole part made solid in that material"),
]


def quantities_of(modules: "list[ModuleType]") -> list[Quantity]:
    """Every quantity: each module's (its QUANTITIES, as the arguments of a Quantity), then the core's, each
    once (available_physics.joined): a quantity several physics give the same way is one, tagged with all of
    them. Every material property a physics or a quantity names must be defined (undefined_properties)."""
    from lsr_mmto.materials.databases import attributes_of
    from lsr_mmto.physics.available_physics import joined, one_each, undefined_properties
    entries = joined("quantities", {q.name: q for q in CORE_QUANTITIES}, modules,
                     lambda m: one_each("quantities", m, m.QUANTITIES))
    out = [Quantity(physics=tags, **entry) for entry, tags in entries.values() if not isinstance(entry, Quantity)] + \
        CORE_QUANTITIES
    undefined_properties(modules, out, attributes_of(modules))
    return out


QUANTITIES = quantities_of(MODULES)


def find_quantity(name: str) -> Quantity:
    """The registry entry of a quantity, by the name a project file uses; KeyError for an unknown name."""
    for q in QUANTITIES:
        if q.name == name:
            return q
    raise KeyError(f"unknown quantity {name!r}; known: {', '.join(q.name for q in QUANTITIES)}")


def takes(name: str, argument: str) -> bool:
    """True when the quantity of that name takes the argument (by name or type); False for an unknown name."""
    return any(q.name == name and q.takes(argument) for q in QUANTITIES)


def default_of(name: str, argument: str) -> float | None:
    """The default of a quantity's argument (the tau of a material volume fraction); None for a required
    argument. StopIteration when the quantity or the argument does not exist."""
    return next(d for q in QUANTITIES if q.name == name for a, _, d, _ in q.arguments if a == argument)


def quantity_label(name: str, opt: dict[str, int | float | str] | None = None,
                   material_labels: list[str] | None = None) -> str:
    """A quantity's name as the app shows it, with the material for a quantity of one material:
    'Material volume fraction of Ti-6Al-4V', 'Mass fraction of the lightest'."""
    shown = next((q.shown for q in QUANTITIES if q.name == name), str(name))
    if isinstance(opt, dict) and "material_id" in opt:
        shown += f" of {material_text(opt['material_id'], material_labels)}"
    return shown

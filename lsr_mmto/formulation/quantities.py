"""
quantities.py
=============
Every objective and constraint a formula can name, by name: QUANTITIES, find_quantity and
quantity_label. The quantities every physics has (mass, cost, volume fraction, material volume
fraction) are computed here; each physics module gives its own (the structural ones are in
physics/structural/structural_quantities.py), listed after these.
"""
import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path (and its float64 default)
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
#   arguments    what its formula takes, in order, as (name, type, default): type "material" (a material's
#                row in the database, a whole number) or "number"; one without a default is required and
#                written by position, one with a default may be written by name: Name(2, tau=0.5)
#   sets_start_density  True for a constraint whose bound is the density every element starts the run at
#                (the volume fraction); the first such constraint sets it, else every element starts at 0.5


class Quantity:
    __slots__ = ("name", "objective", "constraint", "physics", "needs", "fn", "doc", "shown", "help", "unit",
                 "arguments", "scales_with", "sets_start_density")

    def __init__(self, name, objective, constraint, physics, needs, fn, doc, shown, help, unit="", arguments=(),
                 scales_with=None, sets_start_density=False):
        self.name, self.objective, self.constraint = name, objective, constraint
        self.physics, self.needs, self.fn, self.doc = frozenset(physics), tuple(needs), fn, doc
        self.shown, self.help, self.unit, self.arguments = shown, help, unit, tuple(arguments)
        self.scales_with, self.sets_start_density = dict(scales_with or {}), bool(sets_start_density)

    def takes(self, argument):
        """True when the formula takes the argument of that name or type ("material", "tau")."""
        return any(argument in (name, typ) for name, typ, _ in self.arguments)

    def applies_to(self, physics):
        """True for a quantity of that physics, or one every physics has."""
        return not self.physics or physics in self.physics


def _mass(m, opt):
    return torch.sum(m.props["Density"] * m.x_phys) * m.elem_volume


def _cost(m, opt):
    return torch.sum(m.props["Density"] * m.props["Cost"] * m.x_phys) * m.elem_volume


def _volume_fraction(m, opt):
    return torch.mean(m.x_phys)


def _material_volume_fraction(m, opt):
    return material_volume_fraction(m.x_phys, m.z, m.z_real,
                                    int(opt["material_id"]), float(opt.get("tau", DEFAULT_TAU)))


QUANTITIES = [Quantity(physics=[m.NAME], **q) for m in MODULES for q in m.QUANTITIES] + [
    Quantity("Mass", True, True, [], ["Density"],
             _mass, "sum of density * x_phys * element volume", "Mass",
             "The mass of the part, from each material's density.", "{mass}", scales_with={"Density": 1}),
    Quantity("Cost", False, True, [], ["Density", "Cost"],
             _cost, "sum of density * cost * x_phys * element volume", "Cost",
             "The cost of the part, from each material's density and cost per mass.", "USD",
             scales_with={"Cost": 1, "Density": 1}),
    Quantity("VolumeFraction", False, True, [], [],
             _volume_fraction, "mean of x_phys", "Volume fraction",
             "The share of the design space filled with material, 0 to 1.", "fraction of the design domain",
             sets_start_density=True),
    Quantity("MaterialVolumeFraction", False, True, [], [],
             _material_volume_fraction, "volume fraction of one database material, softmax weights",
             "Material volume fraction", "The share of the design space filled with one chosen material, 0 to 1.",
             "fraction of the design domain", (("material_id", "material", None), ("tau", "number", DEFAULT_TAU))),
]


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
    return next(d for q in QUANTITIES if q.name == name for a, _, d in q.arguments if a == argument)


def quantity_label(name, opt=None, material_labels=None):
    """A quantity's name as the app shows it, with the material for a material volume fraction:
    'Material volume fraction of Ti-6Al-4V'."""
    shown = next((q.shown for q in QUANTITIES if q.name == name), str(name))
    if isinstance(opt, dict) and "material_id" in opt:
        i = int(opt["material_id"])
        shown += f" of {material_labels[i] if material_labels is not None and 0 <= i < len(material_labels) else f'material {i}'}"
    return shown

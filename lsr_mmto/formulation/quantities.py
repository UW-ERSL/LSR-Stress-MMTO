"""
quantities.py
=============
Every objective and constraint a formula can name, by name: QUANTITIES, find_quantity and
quantity_label. The quantities that need no physics solve (mass, cost, volume fraction,
material volume fraction) are computed here; the structural ones are computed in
physics/structural/structural_quantities.py.
"""
import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path (and its float64 default)
from lsr_mmto.physics.structural.structural_quantities import (_compliance, _failure_factor_pnorm,
                                                               _stress_pnorm)

DEFAULT_TAU = 0.5        # sharpness of the material assignment, the same for every material volume fraction


def material_volume_fraction(xb, z, z_real, material_id, tau):
    """V_k = (1/N) sum_e xb_e * softmax_k(-||z_e - z_k||^2 / tau)."""
    logits = -((z[:, None, :] - z_real[None, :, :]) ** 2).sum(-1) / tau
    p = torch.softmax(logits, dim=1)
    return torch.sum(xb * p[:, material_id]) / xb.shape[0]


# ----------------------------------------------------------------------------- quantities
# Every objective and constraint is one entry here. To add one: write its function of the
# model's current state (m.x_phys, m.z, m.u, m.svm, m.props, ...) in torch, add an entry, and
# add it to tests/test_ad_quantities.py. Nothing else changes: the gradient comes from autograd.
#
#   name         the name a project file uses (P1); PyTO-AD's name where the meaning is the same
#   objective    True if it can be the objective
#   constraint   True if it can be a constraint
#   physics      the physics it applies to
#   needs        material database columns it reads (checked, with a clear error)
#   fn(m, opt)   the value, a torch scalar; a constraint is fn / limit - 1 <= 0
#   doc          what fn computes, for the settings page
#   shown        its name in the app
#   help         what it is, in plain words: the app's tooltip and the settings page
STRUCTURAL = "structural"


class Quantity:
    __slots__ = ("name", "objective", "constraint", "physics", "needs", "fn", "doc", "shown", "help")

    def __init__(self, name, objective, constraint, physics, needs, fn, doc, shown, help):
        self.name, self.objective, self.constraint = name, objective, constraint
        self.physics, self.needs, self.fn, self.doc = frozenset(physics), tuple(needs), fn, doc
        self.shown, self.help = shown, help


def _mass(m, opt):
    return torch.sum(m.props["Density"] * m.x_phys) * m.elem_volume


def _cost(m, opt):
    return torch.sum(m.props["Density"] * m.props["Cost"] * m.x_phys) * m.elem_volume


def _volume_fraction(m, opt):
    return torch.mean(m.x_phys)


def _material_volume_fraction(m, opt):
    return material_volume_fraction(m.x_phys, m.z, m.z_real,
                                    int(opt["material_id"]), float(opt.get("tau", DEFAULT_TAU)))


QUANTITIES = [
    Quantity("Compliance", True, True, [STRUCTURAL], [],
             _compliance, "F . u", "Compliance",
             "How much the part gives under the loads. Lower is stiffer."),
    Quantity("Mass", True, True, [STRUCTURAL], ["Density"],
             _mass, "sum of density * x_phys * element volume", "Mass",
             "The mass of the part, from each material's density."),
    Quantity("Cost", False, True, [STRUCTURAL], ["Density", "Cost"],
             _cost, "sum of density * cost * x_phys * element volume", "Cost",
             "The cost of the part, from each material's density and cost per mass."),
    Quantity("StressPNorm", True, False, [STRUCTURAL], [],
             _stress_pnorm, "p-norm of the relaxed von Mises stress, no 1/N", "Stress (p-norm)",
             "One value close to the highest von Mises stress in the part."),
    Quantity("FailureFactorPNorm", True, True, [STRUCTURAL], ["Yield_Strength"],
             _failure_factor_pnorm, "mean p-norm of von Mises / yield strength, per element", "Failure factor (p-norm)",
             "One value close to the highest stress / yield strength in the part. Above 1 means some element "
             "yields."),
    Quantity("VolumeFraction", False, True, [STRUCTURAL], [],
             _volume_fraction, "mean of x_phys", "Volume fraction",
             "The share of the design space filled with material, 0 to 1."),
    Quantity("MaterialVolumeFraction", False, True, [STRUCTURAL], [],
             _material_volume_fraction, "volume fraction of one database material, softmax weights",
             "Material volume fraction", "The share of the design space filled with one chosen material, 0 to 1."),
]


def find_quantity(name):
    """The registry entry of a quantity, by the name a project file uses."""
    for q in QUANTITIES:
        if q.name == name:
            return q
    raise KeyError(f"unknown quantity {name!r}; known: {', '.join(q.name for q in QUANTITIES)}")


def quantity_label(name, opt=None, material_labels=None):
    """A quantity's name as the app shows it, with the material for a material volume fraction:
    'Material volume fraction of Ti-6Al-4V'."""
    shown = next((q.shown for q in QUANTITIES if q.name == name), str(name))
    if isinstance(opt, dict) and "material_id" in opt:
        i = int(opt["material_id"])
        shown += f" of {material_labels[i] if material_labels is not None and 0 <= i < len(material_labels) else f'material {i}'}"
    return shown

"""
structural_quantities.py
========================
What the structural module optimizes: its objectives and constraints (QUANTITIES: compliance,
p-norm stress, p-norm failure factor). What it records and reports is in structural_results.py.
"""
from typing import TYPE_CHECKING

import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path
from lsr_mmto.pyto_ad_imports import topopt_material_model as ad_mm
from lsr_mmto.physics.structural.structural_physics import pnorm

if TYPE_CHECKING:
    from lsr_mmto.optimize.design_model import MMTOModelAD
    from lsr_mmto.optimize.solid_part_analysis import _Analysed


# ESSENTIAL: the fn of the Compliance quantity (QUANTITIES)
def _compliance(m: "MMTOModelAD | _Analysed", opt: dict[str, int | float | str] | None) -> torch.Tensor:
    """F . u, a torch scalar, from the solver's total force and the displacement m.u."""
    return torch.dot(m.fe.total_force, m.u)               # F . u, as main's total_force . sol


# ESSENTIAL: the fn of the StressPNorm quantity (QUANTITIES)
def _stress_pnorm(m: "MMTOModelAD | _Analysed", opt: dict[str, int | float | str] | None) -> torch.Tensor:
    """(sum svm^p)^(1/p) of the relaxed von Mises stress m.svm, p the current p-norm exponent."""
    return pnorm(m.svm, ad_mm.get_pNorm_exponent())     # no 1/N, as PyTO's pNormStress


# ESSENTIAL: the fn of the FailureFactorPNorm quantity (QUANTITIES)
def _failure_factor_pnorm(m: "MMTOModelAD | _Analysed", opt: dict[str, int | float | str] | None) -> torch.Tensor:
    """((1/N) sum (svm / Y)^p)^(1/p), Y each element's yield strength (at least 1e-12)."""
    p = ad_mm.get_pNorm_exponent()                      # mean form (1/N), as the manual code on main
    Y = torch.clamp(m.props["Yield_Strength"], min=1e-12)
    return (torch.sum((m.svm / Y) ** p) / m.N) ** (1.0 / p)


# the structural objectives and constraints, as the arguments of a Quantity (formulation/quantities.py)
QUANTITIES = [
    dict(name="Compliance", objective=True, constraint=True, needs=[], fn=_compliance, doc="F . u",
         shown="Compliance", help="How much the part gives under the loads. Lower is stiffer.",
         unit="{force}*{length}", scales_with={"Youngs_Modulus": -1}),
    dict(name="StressPNorm", objective=True, constraint=False, needs=[], fn=_stress_pnorm,
         doc="p-norm of the relaxed von Mises stress, no 1/N", shown="Stress (p-norm)",
         help="One value close to the highest von Mises stress in the part.", unit="{stress}"),
    dict(name="FailureFactorPNorm", objective=True, constraint=True, needs=["Yield_Strength"],
         fn=_failure_factor_pnorm, doc="mean p-norm of von Mises / yield strength, per element",
         shown="Failure factor (p-norm)",
         help="One value close to the highest stress / yield strength in the part. Above 1 means some element "
              "yields.", unit="no units", scales_with={"Yield_Strength": -1}, start_bound=1.0,
         start_note="the highest stress allowed equals the yield strength"),
]

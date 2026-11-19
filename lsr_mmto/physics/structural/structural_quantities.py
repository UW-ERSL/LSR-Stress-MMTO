"""
structural_quantities.py
========================
What the structural module optimizes: its objectives and constraints (QUANTITIES: compliance,
p-norm stress, p-norm failure factor, highest failure factor). What it records and reports is in
structural_results.py.
"""
import math
from typing import TYPE_CHECKING

import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path
from lsr_mmto.formulation.quantity_states import run_state
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
    """(sum svm^p)^(1/p) of the relaxed von Mises stress m.svm, p the run's p-norm exponent (its setting)."""
    return pnorm(m.svm, float(m.settings.pnorm_exponent))     # no 1/N, as PyTO's pNormStress


# HELPER: of the failure factor quantities
def _failure_factors(m: "MMTOModelAD | _Analysed") -> torch.Tensor:
    """svm / Y at each element, (N,): the relaxed von Mises stress over the element's yield strength (at least
    1e-12)."""
    return m.svm / torch.clamp(m.props["Yield_Strength"], min=1e-12)


# ESSENTIAL: the fn of the FailureFactorPNorm quantity (QUANTITIES)
def _failure_factor_pnorm(m: "MMTOModelAD | _Analysed", opt: dict[str, int | float | str] | None) -> torch.Tensor:
    """((1/N) sum (svm / Y)^p)^(1/p), Y each element's yield strength (at least 1e-12)."""
    p = float(m.settings.pnorm_exponent)                # mean form (1/N), as the manual code on main
    return (torch.sum(_failure_factors(m) ** p) / m.N) ** (1.0 / p)


# ESSENTIAL: the fn of the FailureFactorMax quantity (QUANTITIES)
def _failure_factor_max(m: "MMTOModelAD | _Analysed", opt: dict[str, int | float | str] | None) -> torch.Tensor:
    """c * the p-norm failure factor (_failure_factor_pnorm), c a number held fixed in the gradient (Le et al.
    2010, adaptive normalization). In a run c is the run's scale (_carry_failure_factor_scale), 1 at the first
    iteration; outside a run it is the highest failure factor over the p-norm one at this design, which makes
    the value the highest failure factor itself."""
    value = _failure_factor_pnorm(m, opt)
    state = run_state(m, "FailureFactorMax", opt)
    if state is None:
        return value * (torch.max(_failure_factors(m)) / value).detach()
    return value * float(state.get("scale", 1.0))


# ESSENTIAL: the between_iterations of the FailureFactorMax quantity (QUANTITIES)
def _carry_failure_factor_scale(m: "MMTOModelAD", opt: dict[str, int | float | str] | None, state: dict,
                                iteration: int) -> dict[str, float]:
    """The next iteration's scale c from this one's design: c_next = a * r + (1 - a) * c, r the highest failure
    factor over the p-norm one, a the stress_scale_damping setting (0 < a <= 1, 1 taking r as it is). Returns
    the c this iteration used and r, for the history; c stays as it is when r cannot be worked out (no stress)."""
    a = float(m.settings.stress_scale_damping)
    if not 0.0 < a <= 1.0:
        raise ValueError(f"Stress scale damping must be above 0 and at most 1, not {a:g}")
    used = float(state.get("scale", 1.0))
    with torch.no_grad():
        value = float(_failure_factor_pnorm(m, opt))
        highest = float(torch.max(_failure_factors(m)))
    ratio = highest / value if value > 0.0 and math.isfinite(highest / value) else used
    state["scale"] = a * ratio + (1.0 - a) * used
    return {"failure_factor_scale": used, "failure_factor_ratio": ratio}


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
         help="An average of stress / yield strength over the whole part. It sits below the highest value, often "
              "well below, so 1 here does not mean no element yields.", unit="no units",
         scales_with={"Yield_Strength": -1}, start_bound=1.0,
         start_note="the highest stress allowed equals the yield strength"),
    dict(name="FailureFactorMax", objective=False, constraint=True, needs=["Yield_Strength"],
         fn=_failure_factor_max, between_iterations=_carry_failure_factor_scale,
         doc="the p-norm failure factor times a scale that tracks the highest failure factor (adaptive normalization)",
         shown="Failure factor (highest)",
         help="The highest stress / yield strength in the part. At most 1 means no element yields.",
         unit="no units", scales_with={"Yield_Strength": -1}, start_bound=1.0,
         start_note="the highest stress allowed equals the yield strength"),
]

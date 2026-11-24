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




# POLYSTRESS-INSPIRED LOCAL AUGMENTED LAGRANGIAN STRESS OBJECTIVE
# See Giraldo-Londono & Paulino (2021), Eqs. (19), (39)-(43), Appendix E.
# This is the *volume-fraction* version, matching the paper's objective.
# The current structural physics already gives RELAXED stress in m.svm.
# We therefore do NOT multiply the local constraint by a second SIMP
# stiffness factor mE(rho): that would double-relax the stress. The AL
# scheme is PolyStress's, while the stress relaxation remains LSR MMTO's.
# These initial values are Table 3 defaults; changing them is deliberately
# local to this feature until the project exposes them as settings.
_AL_MU0 = 10.0
_AL_MU_MAX = 10000.0
_AL_MU_GROWTH = 1.10
_AL_MMA_PER_STEP = 5


# HELPER: computes the elementwise polynomial stress constraints for AL
def _al_local_constraints(m: "MMTOModelAD | _Analysed") -> torch.Tensor:
    """PolyStress polynomial, one inequality per element (negative = satisfied).

    Lambda = relaxed von Mises / local yield strength - 1.
    Unlike PolyStress's solid-material stress, Lambda here uses LSR MMTO's
    preexisting stress relaxation. Polynomial Lambda*(1+Lambda**2) preserves
    the sign and the zero at yield, without another density multiplier.
    """
    excess = _failure_factors(m).reshape(-1) - 1.0
    return excess * (1.0 + excess.square())


# ESSENTIAL: the fn of the VolumeFractionALStress quantity (QUANTITIES)
def _volume_fraction_al_stress(m: "MMTOModelAD | _Analysed",
                               opt: dict[str, int | float | str] | None) -> torch.Tensor:
    """Physical volume fraction + mean local augmented-Lagrangian penalty.

    L = mean(x_phys) + (1/N) sum(lambda*h + mu/2*h**2),
    h = max(g, -lambda/mu).  Multipliers and mu are constant under autograd.
    """
    f = torch.mean(m.x_phys)
    state = run_state(m, "VolumeFractionALStress", opt)
    if state is None:
        # Outside an optimization run there are no AL multipliers; return
        # the physical volume fraction, not a fictional AL value.
        return f
    g = _al_local_constraints(m)
    lam = state.get("multipliers")
    if lam is None:
        lam = torch.zeros_like(g)
    else:
        lam = lam.to(device=g.device, dtype=g.dtype).detach()
        if lam.shape != g.shape:
            raise ValueError("AL multiplier count differs from element count")
    mu = float(state.get("mu", _AL_MU0))
    h = torch.maximum(g, -lam / mu)
    return f + torch.mean(lam * h + 0.5 * mu * h.square())


# ESSENTIAL: the between_iterations of the VolumeFractionALStress quantity (QUANTITIES)
def _carry_al_stress(m: "MMTOModelAD", opt: dict[str, int | float | str] | None,
                     state: dict, iteration: int) -> dict[str, float]:
    """Hold AL parameters for five MMA evaluations, then update both.

    State changes happen under no_grad AFTER the current objective was
    evaluated. The next evaluation sees the updated values. We report the
    physical maximum stress ratio, not just the polynomial g value.
    """
    with torch.no_grad():
        ratios = _failure_factors(m).reshape(-1)
        g = _al_local_constraints(m).detach()
        lam = state.get("multipliers")
        if lam is None:
            lam = torch.zeros_like(g)
        else:
            lam = lam.to(device=g.device, dtype=g.dtype).detach()
        if lam.shape != g.shape:
            raise ValueError("AL multiplier count differs from element count")
        mu = float(state.get("mu", _AL_MU0))
        h = torch.maximum(g, -lam / mu)
        penalty = torch.mean(lam * h + 0.5 * mu * h.square())
        maximum = float(torch.max(ratios))
        metrics = {
            "al_max_failure_factor": maximum,
            "al_max_stress_violation": max(0.0, maximum - 1.0),
            "al_penalty": float(penalty),
            "al_mu": mu,
            "al_active_elements": float(torch.count_nonzero(h > 0)),
        }
        # Count actual evaluations locally; independent of whether the run's
        # public iteration index starts at zero or one.
        evaluations = int(state.get("al_evaluations", 0)) + 1
        state["al_evaluations"] = evaluations
        if evaluations % _AL_MMA_PER_STEP == 0:
            state["multipliers"] = (lam + mu * h).detach().clone()
            state["mu"] = min(_AL_MU_GROWTH * mu, _AL_MU_MAX)
        else:
            state["multipliers"] = lam.detach().clone()
            state["mu"] = mu
        return metrics



# ESSENTIAL: the fn of the MassALStress quantity (QUANTITIES)
def _mass_al_stress(m: "MMTOModelAD | _Analysed",
                    opt: dict[str, int | float | str] | None) -> torch.Tensor:
    """Normalized physical mass plus the same elementwise AL stress penalty.

    The reference density is the mean decoded density in the INITIAL evaluation,
    detached from autograd and held constant afterwards. Hence this objective
    REALLY rewards lower-density material choices, not just lower volume.
    The mass contribution is M / (reference_density * full_design_volume),
    a dimensionless, fixed-scale physical-mass objective.

    The initial-reference rule is a convenience for this experiment, not a
    universal replacement for an explicit reference mass setting.
    """
    rho = m.props["Density"].reshape(-1)
    state = run_state(m, "MassALStress", opt)
    if state is None:
        # Outside a run there is no persistent reference or AL state.
        return torch.mean(rho * m.x_phys.reshape(-1)) / rho.detach().mean().clamp_min(1e-30)

    ref = state.get("mass_reference_density")
    if ref is None:
        ref = rho.detach().mean().clamp_min(1e-30)
    f = torch.mean(rho * m.x_phys.reshape(-1)) / ref

    g = _al_local_constraints(m)
    lam = state.get("multipliers")
    if lam is None:
        lam = torch.zeros_like(g)
    else:
        lam = lam.to(device=g.device, dtype=g.dtype).detach()
        if lam.shape != g.shape:
            raise ValueError("AL multiplier count differs from element count")
    mu = float(state.get("mu", _AL_MU0))
    h = torch.maximum(g, -lam / mu)
    return f + torch.mean(lam * h + 0.5 * mu * h.square())


# ESSENTIAL: the between_iterations of the MassALStress quantity (QUANTITIES)
def _carry_mass_al_stress(m: "MMTOModelAD", opt: dict[str, int | float | str] | None,
                          state: dict, iteration: int) -> dict[str, float]:
    """Freeze the first design's average decoded density; carry AL state.

    Multiplier updates and history metrics are otherwise the same as for
    VolumeFractionALStress, but belong to the separate MassALStress state.
    """
    if "mass_reference_density" not in state:
        with torch.no_grad():
            ref = float(m.props["Density"].reshape(-1).mean())
        if not math.isfinite(ref) or ref <= 0.0:
            raise ValueError("MassALStress requires a positive reference density")
        state["mass_reference_density"] = ref
    metrics = _carry_al_stress(m, opt, state, iteration)
    with torch.no_grad():
        # Physical mass in the model's mass unit; the optimizer sees
        # the normalized version from _mass_al_stress.
        mass = torch.sum(m.props["Density"].reshape(-1) * m.x_phys.reshape(-1)) * m.elem_volume
        ref_mass = state["mass_reference_density"] * m.N * m.elem_volume
        metrics["al_physical_mass"] = float(mass)
        metrics["al_mass_fraction_reference"] = float(mass) / float(ref_mass)
    return metrics

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
    dict(name="VolumeFractionALStress", objective=True, constraint=False,
         needs=["Yield_Strength"], fn=_volume_fraction_al_stress,
         between_iterations=_carry_al_stress,
         doc="physical volume fraction plus normalized local AL stress penalty",
         shown="Volume fraction (AL stress)",
         help="Minimize material volume while enforcing local relaxed-stress limits "
              "with augmented Lagrangian multipliers updated every five iterations.",
         unit="no units"),
    dict(name="MassALStress", objective=True, constraint=False,
         needs=["Density", "Yield_Strength"], fn=_mass_al_stress,
         between_iterations=_carry_mass_al_stress,
         doc="physical mass divided by fixed initial reference full-solid mass, plus normalized local AL stress penalty",
         shown="Mass (AL stress)",
         help="Minimize actual material mass, including material choices, while enforcing "
              "elementwise relaxed-stress limits using AL multipliers updated every five iterations.",
         unit="no units"),
]

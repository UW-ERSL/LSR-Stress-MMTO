"""
structural_quantities.py
========================
What the structural module measures: the structural objectives and constraints (compliance,
p-norm stress, p-norm failure factor), entered by name in formulation/quantities.py.
"""
import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path
import pyto.autodiff.material_model as ad_mm
from lsr_mmto.physics.structural.structural_physics import pnorm


def _compliance(m, opt):
    return torch.dot(m.fe.total_force, m.u)               # F . u, as main's total_force . sol


def _stress_pnorm(m, opt):
    return pnorm(m.svm, ad_mm.get_pNorm_exponent())     # no 1/N, as PyTO's pNormStress


def _failure_factor_pnorm(m, opt):
    p = ad_mm.get_pNorm_exponent()                      # mean form (1/N), as the manual code on main
    Y = torch.clamp(m.props["Yield_Strength"], min=1e-12)
    return (torch.sum((m.svm / Y) ** p) / m.N) ** (1.0 / p)

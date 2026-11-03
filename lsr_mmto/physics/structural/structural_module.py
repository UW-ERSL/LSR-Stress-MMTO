"""
structural_module.py
====================
What the structural module offers the optimization, the names physics/available_physics.py says
every physics offers: Solver (the finite element solver), Model (one iteration's physics:
decode, solve, objective, constraints and their slopes), decode, distance_penalty and
latent_smoothing.
"""
from lsr_mmto.design.latent_penalties import distance_penalty, latent_smoothing  # noqa: F401
from lsr_mmto.design.latent_to_properties import decode  # noqa: F401
from lsr_mmto.optimize.design_model import MMTOModelAD
from lsr_mmto.physics.structural.structural_physics import MultiMaterialHexFEA

Solver = MultiMaterialHexFEA
Model = MMTOModelAD

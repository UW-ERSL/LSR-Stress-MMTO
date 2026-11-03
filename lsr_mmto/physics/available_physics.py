"""
available_physics.py
====================
PHYSICS: each physics a project may name ("physics": "structural" in its file), the module that
implements it and the database columns every problem of it reads.

A physics module offers what the optimization builds from it: Solver (the finite element solver,
made from the mesh, the boundary conditions and the linear solver), Model (one iteration's
physics: decode, solve, objective, constraints and their slopes), decode, distance_penalty and
latent_smoothing. A new physics is one folder offering these, plus one entry here.
"""
import importlib


class Physics:
    __slots__ = ("name", "module", "needs", "doc")

    def __init__(self, name, module, needs, doc):
        self.name, self.module, self.needs, self.doc = name, module, tuple(needs), doc


PHYSICS = {p.name: p for p in [
    Physics("structural", "lsr_mmto.physics.structural.structural_module", ["Youngs_Modulus"], "linear elasticity, small strain"),
]}


def physics_module(name):
    """The file of physics name, imported; an unknown name is an error that lists the known ones."""
    if name not in PHYSICS:
        raise ValueError(f"physics {name!r}: one of {', '.join(PHYSICS)}")
    return importlib.import_module(PHYSICS[name].module)

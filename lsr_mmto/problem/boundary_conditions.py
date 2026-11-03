"""
boundary_conditions.py
======================
The kinds of support and load a project can use, BC_KINDS: one BCKind each, from the physics
modules (today the structural one, physics/structural/structural_supports_and_loads.py).
"""
from lsr_mmto.physics.structural.structural_supports_and_loads import STRUCTURAL_BC_KINDS


class BCKind:
    """One kind of boundary condition. role is "support" or "load"; check(bc) gives a list
    of errors; apply(bc, nodes, mesh, state, where) puts it on the mesh, in
    state["values"] (dof -> imposed value) or state["force"]; script(bc, on) is the
    Project call that makes it. Supports are applied before loads, each in the order given.
    fields: its values, as (name, kind, measures, doc); kind is "dofs" (some of x, y, z),
    "number", "vector" (three numbers) or "area" (a number or "bbox"); measures is what
    the number is (one of UNITS, or "area"), for its unit. The GUI's form is made from them."""

    def __init__(self, doc, role, check, apply, script, fields=()):
        self.doc, self.role, self.check, self.apply, self.script = doc, role, check, apply, script
        self.fields = tuple(fields)


BC_KINDS = {name: BCKind(**args) for name, args in STRUCTURAL_BC_KINDS.items()}

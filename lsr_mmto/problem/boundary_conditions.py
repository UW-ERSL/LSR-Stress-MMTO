"""
boundary_conditions.py
======================
The kinds of support and load a project can use, BC_KINDS: one BCKind each, from every physics
module's SUPPORT_AND_LOAD_KINDS (the structural ones are in
physics/structural/structural_supports_and_loads.py).
"""
from lsr_mmto.physics.available_physics import MODULES


class BCKind:
    """One kind of boundary condition. role is "support" or "load"; field is the field it acts on
    (one of its physics' FIELDS, e.g. "u"), and each field gets a boundary condition of its own; check(bc) gives a list
    of errors; apply(bc, nodes, mesh, state, where) puts it on the mesh, in
    state["values"] (dof -> imposed value) or state["force"]; script(bc, on) is the
    Project call that makes it. Supports are applied before loads, each in the order given.
    fields: its values, as (name, kind, measures, doc); kind is "dofs" (some of x, y, z),
    "number", "vector" (three numbers) or "area" (a number or "bbox"); measures is what
    the number is (one of UNITS, or "area"), for its unit. The GUI's form is made from them.
    picks: {type in the picking window: its values from the window's entry}, for a kind that can be
    picked on the part (None is an entry with no type)."""

    def __init__(self, doc, role, field, check, apply, script, fields=(), picks=None):
        self.doc, self.role, self.field, self.check, self.apply, self.script = doc, role, field, check, apply, script
        self.fields, self.picks = tuple(fields), dict(picks or {})


BC_KINDS = {name: BCKind(**args) for m in MODULES for name, args in m.SUPPORT_AND_LOAD_KINDS.items()}

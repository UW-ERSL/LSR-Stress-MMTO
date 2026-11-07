"""
boundary_conditions.py
======================
The kinds of support and load a project can use, BC_KINDS: one BCKind each, from every physics
module's SUPPORT_AND_LOAD_KINDS (the structural ones are in
physics/structural/structural_supports_and_loads.py). A project adds one with
p.add_bc(kind, on, **values), and its setup script is written the same way.
"""
import numpy as np

from lsr_mmto.physics.available_physics import MODULES

VALUE_KINDS = ("dofs", "number", "vector", "area")


class Value:
    """One value of a kind of support or load. kind is "dofs" (some of x, y, z, e.g. "xz"), "number",
    "vector" (three numbers) or "area" (a number, or "bbox": the selection's extent across the
    direction); measures is what the number is, for its unit (one of the unit kinds of
    materials/databases.py, "area", or None for no unit); label is what the GUI's form calls it,
    doc what it is, for the reference page."""
    __slots__ = ("name", "kind", "measures", "label", "doc")

    def __init__(self, name, kind, label, measures=None, doc=""):
        if kind not in VALUE_KINDS:
            raise ValueError(f"value {name}: kind must be one of {', '.join(VALUE_KINDS)}")
        self.name, self.kind, self.measures, self.label, self.doc = name, kind, measures, label, doc

    def errors(self, v):
        """What is wrong with v as this value, as a list of short texts (empty when it is fine)."""
        if self.kind == "dofs":
            if not isinstance(v, str) or not v or any(c not in "xyz" for c in v) or len(set(v)) != len(v):
                return [f"{self.name} must be letters from 'xyz', e.g. 'xyz' or 'y', not {v!r}"]
            return []
        if self.kind == "vector":
            try:
                a = np.asarray(v, dtype=float)
            except (TypeError, ValueError):
                a = np.zeros(0)
            if a.shape != (3,) or not np.any(a):
                return [f"{self.name} must be three numbers, not all zero"]
            return []
        if self.kind == "area" and v == "bbox":
            return []
        try:
            x = float(v)
        except (TypeError, ValueError):
            return [f"{self.name} must be a number" + (" or bbox" if self.kind == "area" else "")]
        if self.kind == "area" and not x > 0:
            return [f"{self.name} must be more than 0, or bbox"]
        return []

    def stored(self, v):
        """v as a project file keeps it: dofs as text, numbers as floats, "bbox" as it is."""
        if self.kind == "dofs":
            return str(v)
        if self.kind == "vector":
            return [float(x) for x in v]
        if self.kind == "area" and v == "bbox":
            return v
        return float(v)


class BCKind:
    """One kind of boundary condition, made from the arguments a physics gives for it:
    role     "support" or "load"; supports are applied before loads, each in the order given
    apply    apply(bc, nodes, mesh, state, where) puts it on the mesh, in state["values"]
             (dof -> imposed value) or state["force"], both of its field
    shown    its name in the GUI and the user guide
    help     what it does, in one plain sentence: the GUI's (i), the user guide, the reference page
    values   its values, each the arguments of a Value (name, kind, label, measures, doc); the GUI's
             form, the checks and the setup-script line are made from them
    field    the field it acts on (one of its physics' FIELDS); a physics of one field leaves it out
    check    optional: check(bc) -> errors its values' kinds do not catch (a list of short texts)
    picks    optional: {type in the picking window: its values from the window's entry}, for a
             kind that can be picked on the part (None is an entry with no type)"""

    def __init__(self, role, apply, shown, help, values=(), field=None, check=None, picks=None, physics=()):
        if role not in ("support", "load"):
            raise ValueError(f"role must be support or load, not {role!r}")
        self.role, self.apply, self.shown, self.help, self.field = role, apply, shown, help, field
        self.physics = tuple(physics)                       # the physics that give it
        self.values = tuple(Value(**v) for v in values)
        self.extra_check, self.picks = check, dict(picks or {})

    def value(self, name):
        """The Value of that name, or None."""
        return next((v for v in self.values if v.name == name), None)

    def errors(self, bc):
        """What is wrong with a support or load of this kind: its values, then the kind's own check."""
        out = []
        for v in self.values:
            out += [f"{v.name} is missing"] if v.name not in bc else v.errors(bc[v.name])
        unknown = [k for k in bc if k not in ("kind", "on") and self.value(k) is None]
        if unknown:
            out.append(f"not a value of this kind: {', '.join(unknown)}")
        if not out and self.extra_check is not None:
            out += self.extra_check(bc)
        return out

    def script(self, name, bc, on):
        """The line of a setup script that adds this support or load."""
        values = "".join(f", {v.name}={bc[v.name]!r}" for v in self.values if v.name in bc)
        return f"p.add_bc({name!r}, {on}{values})"


def kinds_of(module, physics=None):
    """{name: BCKind} of one physics module's SUPPORT_AND_LOAD_KINDS; a kind that names no field acts on
    the module's one field. physics: the names of the physics each kind is tagged with (by default the
    module's)."""
    out = {}
    for name, args in module.SUPPORT_AND_LOAD_KINDS.items():
        args = dict(args)
        if args.get("field") is None:
            if len(module.FIELDS) != 1:
                raise ValueError(f"{module.NAME} support or load {name}: say which field it acts on "
                                 f"({', '.join(module.FIELDS)})")
            args["field"] = next(iter(module.FIELDS))
        out[name] = BCKind(physics=(physics or {}).get(name, [module.NAME]), **args)
    return out


def bc_kinds_of(modules):
    """{name: BCKind} of every module's kinds, each once (available_physics.joined): a kind several physics give
    the same way is one, tagged with all of them; two different kinds under one name are an error."""
    from lsr_mmto.physics.available_physics import joined
    entries = joined("kinds of support and load", {}, modules, lambda m: m.SUPPORT_AND_LOAD_KINDS)
    tags = {name: t for name, (_, t) in entries.items()}
    out = {}
    for m in modules:
        out.update({name: kind for name, kind in kinds_of(m, tags).items() if name not in out})
    return out


BC_KINDS = bc_kinds_of(MODULES)

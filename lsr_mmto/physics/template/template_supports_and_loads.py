"""
template_supports_and_loads.py
==============================
What this file is for: the kinds of support and load of the template physics. A user puts one on a
selection of the part (in the Supports and loads panel, or in a setup script); this file says what
that does to the equations K c = f.

Who reads it:
    problem/boundary_conditions.py   turns each entry of SUPPORT_AND_LOAD_KINDS into a BCKind
    problem/project.py               checks a project's supports and loads (validate), and builds them
                                     on the mesh before the run (supports_and_loads)
    gui/supports_and_loads_panel.py  makes the form of each kind from its values and words
    gui/user_guide_window.py         lists the kinds in the user guide

Glossary:
    selection       a named part of the mesh (a face, a box, picked triangles); a support or load acts
                    on its nodes
    the bc dict     one support or load as a project stores it, e.g.
                        {"kind": "source", "on": "top", "total": 4.0}
                    "kind" is a key of SUPPORT_AND_LOAD_KINDS, "on" the selection's name, the rest
                    its values (the names in its values)
    state           what the supports and loads of one field build up, before the solve:
    state["values"] {dof: the value it is held at}; these dofs are not solved for
    state["force"]  f of K c = f: one number per dof, so one per node here

How a support or load reaches the solve (project.py, Project.supports_and_loads):
    1. For each field of the physics, an empty state: values {} and force zeros.
    2. Supports first, then loads, in the order the project lists them: the selection's nodes are
       found, and the kind's apply(bc, nodes, mesh, state, where) writes into the state of its field.
    3. Each field's state becomes PyTO-AD's BC(force, fixed_dofs, dirichlet_values).
    4. make_solver (template_physics.py) hands that BC to the solver.

Worked example, a source (below): a mesh of 4 nodes, a source with total 4 on the selection {1, 3}.
    nodes n = [1, 3], so each gets 4 / 2 = 2.
    state["force"] before:  [0, 0, 0, 0]
    state["force"] after:   [0, 2, 0, 2]
That f is what the solver uses.

To add a kind, copy one entry of SUPPORT_AND_LOAD_KINDS and change the lines marked FILL.
"""
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pyto.core.bc import BC
    from pyto.core.hex_mesher import HexMesher

FIELD = "c"                    # FILL: the field, as in FIELDS in template_module.py (read below)


# ----------------------------------------------------------------------------- what each kind does
def _check_total(bc: dict[str, Any]) -> list[str]:
    """Errors its values' kinds do not catch, as a list of short texts (empty when it is fine). Read by
    Project.validate and by the form before it accepts. That total is a number is checked already, from
    its value's kind; here only that it is not zero."""
    return [] if float(bc["total"]) != 0.0 else ["total must not be zero"]


def _apply_set_value(bc: dict[str, Any], n: np.ndarray, mesh: "HexMesher",
                     state: dict[str, Any], where: str) -> None:
    """Hold c at the value on the nodes n. Here dof = node (one unknown per node).
    where names the support in error messages ("boundary condition 2")."""
    values, v = state["values"], float(bc.get("value", 0.0))
    for d in np.asarray(n, dtype=int):
        if values.get(int(d), v) != v:                       # two supports, two values on one dof
            raise ValueError(f"{where}: node {int(d)} is already held at {values[int(d)]}, here {v}")
        values[int(d)] = v


def _apply_source(bc: dict[str, Any], n: np.ndarray, mesh: "HexMesher",
                  state: dict[str, Any], where: str) -> None:
    """Add the total to f, shared equally by the nodes n (the worked example above)."""
    state["force"][np.asarray(n, dtype=int)] += float(bc["total"]) / len(n)


# ----------------------------------------------------------------------------- the kinds
# Each entry is the arguments of a BCKind (problem/boundary_conditions.py). What each one is, and who
# reads it:
#   role     "support" or "load": supports are applied before loads (project.py, supports_and_loads);
#            the Pick window and the "is the part held" check use it too
#   apply    apply(bc, nodes, mesh, state, where): what it does to the equations (project.py,
#            supports_and_loads); the only physics in the entry
#   shown    its name in the GUI and the user guide
#   help     one plain sentence for a user: the form's (i), the user guide, the reference page
#   values   its values, each a dict: name (its key in the bc dict), kind ("number", "vector" for three
#            numbers, "dofs" for some of x, y, z, or "area": a number or bbox), label (its name in the
#            form), measures (what the number is, for its unit: one of the unit kinds of
#            materials/databases.py, or None for no unit) and doc (what it is, for the reference
#            page). From them come the form (gui/supports_and_loads_panel.py), the checks that a value
#            is there and is a number, three numbers or x, y, z (project.py, validate; the form), and
#            the setup-script line, p.add_bc("source", "top", total=4.0) (project.py, to_script)
#   check    optional: check(bc) -> errors the value kinds do not catch
#   field    left out here: a physics of one field (FIELDS in template_module.py) has its kinds act on
#            it; a physics of several fields names the field in each kind
#   picks    optional: how a face picked in the Pick window becomes this kind; left out here
SUPPORT_AND_LOAD_KINDS = {
    "set_value": dict(                                       # FILL: the kind's name in a project file
        role="support",
        apply=_apply_set_value,
        shown="Set value",                                   # FILL
        help="Holds the value at the selection.",            # FILL
        values=[dict(name="value", kind="number", label="Value", doc="the value held")],    # FILL
    ),
    "source": dict(
        role="load",
        apply=_apply_source,
        check=_check_total,
        shown="Source",
        help="Adds an amount at the selection, shared equally by its nodes.",
        values=[dict(name="total", kind="number", label="Total",
                     doc="the total amount, split equally over the nodes")],
    ),
}


def supports_and_loads_error(mesh: "HexMesher", bcs: "dict[str, BC]") -> str | None:
    """Why the supports and loads leave the problem without a solution, as one sentence, or None.
    Read by Project.build before the run (and by the preview). For diffusion: if c is held nowhere,
    K is singular (any constant can be added to c), so at least one Set value is needed."""
    if len(bcs[FIELD].fixed_dofs) == 0:
        return "the value is held nowhere; add a Set value on some selection"
    return None

"""
template_supports_and_loads.py
==============================
The job of this file: the kinds of support and load of the physics. A user puts one on a selection of the
part (in the Supports and loads panel, or in a setup script); this file says what that does to the
equations K u = f. Thermal's thermal_supports_and_loads.py is this file filled in.

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
                        {"kind": "heat_source", "on": "top", "total": 4.0}
                    "kind" is a key of SUPPORT_AND_LOAD_KINDS, "on" the selection's name, the rest its
                    values (the names in its values)
    state           what the supports and loads of one field build up, before the solve:
    state["values"] {dof: the value it is held at}; these dofs are not solved for
    state["force"]  f of K u = f: one number per dof

How a support or load reaches the solve (project.py, Project.supports_and_loads):
    1. For each field of the physics, an empty state: values {} and force zeros.
    2. Supports first, then loads, in the order the project lists them: the selection's nodes are found,
       and the kind's apply(bc, nodes, mesh, state, where) writes into the state of its field.
    3. Each field's state becomes PyTO-AD's BC(force, fixed_dofs, dirichlet_values).
    4. make_solver (template_physics.py) hands that BC to the solver.

Worked example, thermal's heat source: a mesh of 4 nodes, a source with total 4 on the selection {1, 3}.
    nodes n = [1, 3], so each gets 4 / 2 = 2.
    state["force"] before:  [0, 0, 0, 0]
    state["force"] after:   [0, 2, 0, 2]
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyto.core.bc import BC
    from pyto.core.hex_mesher import HexMesher

# SUPPORT_AND_LOAD_KINDS: {its name in a project file: the arguments of a BCKind (problem/boundary_conditions.py)}.
# What each argument is, and who reads it:
#   role     "support" or "load": supports are applied before loads (project.py, supports_and_loads); the
#            Pick window and the "is the part held" check use it too
#   apply    apply(bc, nodes, mesh, state, where): what it does to the equations; the only physics in the
#            entry, a Helper below
#   shown    its name in the GUI and the user guide
#   help     one plain sentence for a user: the form's (i), the user guide, the reference page
#   values   its values, each a dict: name (its key in the bc dict), kind ("number", "vector" for three
#            numbers, "dofs" for some of x, y, z, or "area": a number or bbox), label (its name in the form),
#            measures (what the number is, for its unit: a unit kind, or None for no unit) and doc (what it
#            is, for the reference page)
#   check    optional: check(bc) -> errors the value kinds do not catch, a list of short texts
#   field    optional: the field it acts on; a physics of one field leaves it out
#   picks    optional: how a face picked in the Pick window becomes this kind
# Thermal's heat source:
#   "heat_source": dict(role="load", apply=_apply_heat_source, check=_check_total, shown="Heat source",
#                       help="Puts heat into the selection, shared equally by its nodes.",
#                       values=[dict(name="total", kind="number", measures="heat", label="Total heat",
#                                    doc="the heat put in, split equally over the nodes")]),
SUPPORT_AND_LOAD_KINDS = ...                                        # FILL


# ----------------------------------------------------------------------------- OPTIONAL entries of this file
# OPTIONAL: supports_and_loads_error, the check that the supports hold the part
# def supports_and_loads_error(mesh: "HexMesher", bcs: "dict[str, BC]") -> str | None:
#     """Why the supports and loads leave the problem without a solution, as one sentence, or None. Read by
#     Project.build before the run (and by the preview). Thermal: the temperature held nowhere; structural:
#     the part free to move where the loads push. Left out, every problem is taken as held."""

# ----------------------------------------------------------------------------- Helpers
# Each kind's apply, and its check if it has one. apply(bc, n, mesh, state, where) -> None writes into the
# state of its field (the glossary above); where names the support in an error message. Mark each one
# "# HELPER: the <kind> kind's apply (SUPPORT_AND_LOAD_KINDS)".

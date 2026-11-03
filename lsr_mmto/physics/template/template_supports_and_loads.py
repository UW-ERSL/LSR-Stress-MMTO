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
                    its values (the names in its fields)
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
import numpy as np

FIELD = "c"                    # FILL: the field these kinds act on, one of FIELDS in template_module.py


# ----------------------------------------------------------------------------- what each kind does
def _check_value(bc):
    """Errors in a set-value support's values, as a list of short texts (empty when it is fine).
    Read by Project.validate and by the form before it accepts."""
    try:
        float(bc.get("value", 0.0))
        return []
    except (TypeError, ValueError):
        return ["value must be a number"]


def _check_total(bc):
    try:
        return [] if float(bc.get("total")) != 0.0 else ["total must not be zero"]
    except (TypeError, ValueError):
        return ["total must be a number"]


def _apply_set_value(bc, n, mesh, state, where):
    """Hold c at the value on the nodes n. Here dof = node (one unknown per node).
    where names the support in error messages ("boundary condition 2")."""
    values, v = state["values"], float(bc.get("value", 0.0))
    for d in np.asarray(n, dtype=int):
        if values.get(int(d), v) != v:                       # two supports, two values on one dof
            raise ValueError(f"{where}: node {int(d)} is already held at {values[int(d)]}, here {v}")
        values[int(d)] = v


def _apply_source(bc, n, mesh, state, where):
    """Add the total to f, shared equally by the nodes n (the worked example above)."""
    state["force"][np.asarray(n, dtype=int)] += float(bc["total"]) / len(n)


# ----------------------------------------------------------------------------- the kinds
# Each entry is the arguments of a BCKind (problem/boundary_conditions.py). What each one is, and who
# reads it (as the files stand now):
#   doc      one line on what it does, for a reader of the code; the Supports and loads form shows it
#            only when help is empty (gui/supports_and_loads_panel.py, the kind's (i))
#   role     "support" or "load": supports are applied before loads (project.py, supports_and_loads);
#            the Pick window and the "is the part held" check use it too
#   field    the field it writes into, FIELD above (project.py: validate and supports_and_loads)
#   check    check(bc) -> list of errors (project.py, validate; the form, before it accepts)
#   apply    apply(bc, nodes, mesh, state, where): what it does to the equations (project.py,
#            supports_and_loads); the only physics in the entry
#   script   script(bc, on) -> the line of a setup script that makes it (project.py, to_script)
#   fields   its values, as (name, type, measures, doc); type "number", "vector" (three numbers),
#            "dofs" (some of x, y, z) or "area"; measures is what the number is, for its unit (one
#            of the unit kinds of materials/databases.py, or None for no unit). The form is made
#            from them (gui/supports_and_loads_panel.py)
#   shown    its name in the GUI and the user guide
#   help     one plain sentence for a user, in the form and the user guide
#   labels   {value name: its label in the form}
#   picks    optional: how a face picked in the Pick window becomes this kind; leave it out
SUPPORT_AND_LOAD_KINDS = {
    "set_value": dict(                                       # FILL: the kind's name in a project file
        doc="c held at a value on the selection",            # FILL
        role="support",
        field=FIELD,
        check=_check_value,
        apply=_apply_set_value,
        script=lambda bc, on: f"p.bcs.append({{'kind': 'set_value', 'on': {on}, 'value': {bc.get('value', 0.0)!r}}})",
        fields=[("value", "number", None, "the value held")],
        shown="Set value",                                   # FILL
        help="Holds the value at the selection.",            # FILL
        labels={"value": "Value"},
    ),
    "source": dict(
        doc="a total amount, split equally over the nodes",
        role="load",
        field=FIELD,
        check=_check_total,
        apply=_apply_source,
        script=lambda bc, on: f"p.bcs.append({{'kind': 'source', 'on': {on}, 'total': {bc['total']!r}}})",
        fields=[("total", "number", None, "the total amount, split equally over the nodes")],
        shown="Source",
        help="Adds an amount at the selection, shared equally by its nodes.",
        labels={"total": "Total"},
    ),
}


def supports_and_loads_error(mesh, bcs):
    """Why the supports and loads leave the problem without a solution, as one sentence, or None.
    Read by Project.build before the run (and by the preview). For diffusion: if c is held nowhere,
    K is singular (any constant can be added to c), so at least one Set value is needed."""
    if len(bcs[FIELD].fixed_dofs) == 0:
        return "the value is held nowhere; add a Set value on some selection"
    return None

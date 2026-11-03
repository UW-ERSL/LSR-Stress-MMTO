"""
structural_supports_and_loads.py
================================
The structural supports and loads, each acting on a named selection, evaluated on the mesh nodes:
    fixed          dofs "x", "y", "z" or any combination, value 0
    displacement   dofs as above, a nonzero imposed value
    force          a total force [Fx, Fy, Fz], split equally over the nodes
    traction       pressure * area along a direction, split equally over the nodes;
                   area is a number, or "bbox": the bounding box of the loaded nodes
                   seen along an axis-aligned direction (each side at least half an
                   element). "bbox" is approximate; it is there for the airplane bracket.
Each kind is given as the arguments of a BCKind (problem/boundary_conditions.py), which turns
them into the kinds a project can use. picks: how a support or a load picked on the part in the
window (its type there) becomes this kind.
"""
import numpy as np

_AXES = {"x": 0, "y": 1, "z": 2}
_MARK_FIXED, _MARK_LOAD, _MARK_ROLLER = 1, 2, 3
# the directions each support type of the picking tools holds
_PICKED_SUPPORTS = {"Fixed XYZ": "xyz", "Fixed X": "x", "Fixed Y": "y", "Fixed Z": "z"}

def _dofs(dofs):
    if not dofs or any(c not in _AXES for c in dofs) or len(set(dofs)) != len(dofs):
        raise ValueError(f"dofs must be letters from 'xyz', e.g. 'xyz' or 'y', not {dofs!r}")
    return [_AXES[c] for c in sorted(dofs, key="xyz".index)]


def _kind(doc, role, check, apply, script, fields, shown, help, labels, picks=None):
    """The arguments of one BCKind; every structural kind acts on the displacement u."""
    return {"doc": doc, "role": role, "field": "u", "check": check, "apply": apply, "script": script,
            "fields": fields, "shown": shown, "help": help, "labels": labels, "picks": picks or {}}


def _picked_force(f):
    return {"total": [float(f.get("force_x", 0.0)), float(f.get("force_y", 0.0)), float(f.get("force_z", 0.0))]}


def _check_dofs(bc):
    try:
        _dofs(bc.get("dofs"))
        return []
    except ValueError as e:
        return [str(e)]


def _check_force(bc):
    f = np.asarray(bc.get("total", []), dtype=float)
    if f.shape != (3,) or not np.any(f):
        return ["total must be three numbers, not all zero"]
    return []


def _check_direction(bc):
    d = np.asarray(bc.get("direction", []), dtype=float)
    if d.shape != (3,) or not np.any(d):
        return ["direction must be three numbers, not all zero"]
    if bc.get("area") == "bbox" and np.count_nonzero(d) != 1:
        return ["area 'bbox' needs a direction along x, y or z; give a number instead"]
    return []


def supports_and_loads_error(mesh, bcs):
    """Why the supports let the part move where the loads push, or None (PyTO-AD's check)."""
    from lsr_mmto.pyto_ad_imports import bound_cond
    well_posed, worst = bound_cond.check_rigid_body_constraint(mesh, bcs["u"])
    if well_posed:
        return None
    size = "" if worst is None else f" (relative size {worst:.3g})"
    return (f"the supports leave the part free to move or rotate in a direction the loads push{size}; "
            f"add supports that hold it")


def _apply_support(bc, n, mesh, state, where):
    values, v = state["values"], float(bc.get("value", 0.0))
    for d in (3 * n[:, None] + np.array(_dofs(bc["dofs"]))[None, :]).ravel():
        if values.get(int(d), v) != v:
            raise ValueError(f"{where}: dof {int(d)} is already given the value {values[int(d)]}, here {v}")
        values[int(d)] = v
    mesh.node_indices[n, 3] = _MARK_FIXED if len(bc["dofs"]) == 3 else _MARK_ROLLER


def _add_load(total, n, mesh, state):
    state["force"].reshape(-1, 3)[n] += total / len(n)
    mesh.node_indices[n, 3] = _MARK_LOAD


def _apply_force(bc, n, mesh, state, where):
    _add_load(np.asarray(bc["total"], dtype=float), n, mesh, state)


def _apply_traction(bc, n, mesh, state, where):
    d = np.asarray(bc["direction"], dtype=float)
    d = d / np.linalg.norm(d)
    area = bc["area"]
    if area == "bbox":
        tol = 0.5 * float(np.min(mesh.elem_size))
        p = np.asarray(mesh.node_xyz, dtype=float)[n]
        a, b = [k for k in range(3) if d[k] == 0]
        area = max(np.ptp(p[:, a]), tol) * max(np.ptp(p[:, b]), tol)
    _add_load(bc["pressure"] * area * d, n, mesh, state)


SUPPORT_AND_LOAD_KINDS = {
    "fixed": _kind(
        "the given dofs (x, y, z) held at 0", "support", _check_dofs, _apply_support,
        lambda bc, on: f"p.add_support({on}, dofs={bc['dofs']!r})",
        [("dofs", "dofs", None, "the directions held")],
        "Fixed support", "Holds the selection still in the ticked directions.", {"dofs": "Held in"},
        {t: (lambda c, d=d: {"dofs": d}) for t, d in _PICKED_SUPPORTS.items()}),
    "displacement": _kind(
        "the given dofs held at a nonzero value", "support", _check_dofs, _apply_support,
        lambda bc, on: f"p.add_support({on}, dofs={bc['dofs']!r}, value={bc['value']!r})",
        [("dofs", "dofs", None, "the directions moved"), ("value", "number", "length", "the displacement")],
        "Moved support", "Moves the selection by a set distance in the ticked directions.",
        {"dofs": "Moved in", "value": "Distance"}),
    "force": _kind(
        "a total force [Fx, Fy, Fz], split equally over the nodes", "load", _check_force, _apply_force,
        lambda bc, on: f"p.add_force({on}, total={bc['total']!r})",
        [("total", "vector", "force", "the total force, split equally over the nodes")],
        "Force", "A total force on the selection, shared equally by its nodes.", {"total": "Total force"},
        {"force_xyz": _picked_force, None: _picked_force}),
    "traction": _kind(
        "pressure * area along a direction, split equally over the nodes; area a number or 'bbox'",
        "load", _check_direction, _apply_traction,
        lambda bc, on: f"p.add_traction({on}, pressure={bc['pressure']!r}, "
                       f"direction={bc['direction']!r}, area={bc['area']!r})",
        [("pressure", "number", "stress", "the pressure"), ("direction", "vector", None, "the direction it pushes"),
         ("area", "area", "area", "the area it acts on, or bbox: the selection's extent across the direction")],
        "Pressure", "A pressure on the selection, turned into a force: Pressure times Area, pushing along Direction, "
        "shared equally by its nodes. Tick Area from the selection to use the selection's size across the "
        "direction (Direction must then be along x, y or z).",
        {"pressure": "Pressure", "direction": "Direction", "area": "Area"}),
}

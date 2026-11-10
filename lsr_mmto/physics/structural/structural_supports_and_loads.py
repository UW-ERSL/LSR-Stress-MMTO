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
them into the kinds a project can use: what it does (apply), its name and help in the GUI, and its
values, from which the form, the checks and the setup-script line are made. Every kind here acts on
the displacement u, the physics' one field. picks: how a support or a load picked on the part in the
window (its type there) becomes this kind. A script adds one with p.add_bc, e.g.
p.add_bc("fixed", "clamp", dofs="xyz") or p.add_bc("force", "corner", total=[0, -4, 0]).
A kind's apply(bc, n, mesh, state, where) gets bc, its values by name; n, the selection's nodes; state,
{"values": {dof: the value it is held at}, "force": one number per dof} of the field u, which it writes
into; and where, how error messages name it.
"""
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pyto.core.bc import BC
    from pyto.core.hex_mesher import HexMesher

_AXES = {"x": 0, "y": 1, "z": 2}
_MARK_FIXED, _MARK_LOAD, _MARK_ROLLER = 1, 2, 3
# the directions each support type of the picking tools holds
_PICKED_SUPPORTS = {"Fixed XYZ": "xyz", "Fixed X": "x", "Fixed Y": "y", "Fixed Z": "z"}

def _dofs(dofs: str) -> list[int]:
    """The dof offsets (0, 1, 2) of the letters x, y, z, in that order; ValueError for anything else."""
    if not dofs or any(c not in _AXES for c in dofs) or len(set(dofs)) != len(dofs):
        raise ValueError(f"dofs must be letters from 'xyz', e.g. 'xyz' or 'y', not {dofs!r}")
    return [_AXES[c] for c in sorted(dofs, key="xyz".index)]


def _picked_force(f: dict[str, Any]) -> dict[str, list[float]]:
    """A force picked on the part in the window (force_x, force_y, force_z, each 0 when absent) as this kind's
    values."""
    return {"total": [float(f.get("force_x", 0.0)), float(f.get("force_y", 0.0)), float(f.get("force_z", 0.0))]}


def _check_direction(bc: dict[str, Any]) -> list[str]:
    """An area taken from the selection needs a direction along x, y or z."""
    d = np.asarray(bc["direction"], dtype=float)
    if bc.get("area") == "bbox" and np.count_nonzero(d) != 1:
        return ["area 'bbox' needs a direction along x, y or z; give a number instead"]
    return []


def supports_and_loads_error(mesh: "HexMesher", bcs: "dict[str, BC]") -> str | None:
    """Why the supports let the part move where the loads push, or None (PyTO-AD's check)."""
    from lsr_mmto.pyto_ad_imports import bound_cond
    well_posed, worst = bound_cond.check_rigid_body_constraint(mesh, bcs["u"])
    if well_posed:
        return None
    size = "" if worst is None else f" (relative size {worst:.3g})"
    return (f"the supports leave the part free to move or rotate in a direction the loads push{size}; "
            f"add supports that hold it")


def _apply_support(bc: dict[str, Any], n: np.ndarray, mesh: "HexMesher",
                   state: dict[str, Any], where: str) -> None:
    """Hold the dofs named in bc["dofs"] of the nodes n at the value (0 when absent), in state["values"];
    ValueError when one is already held at another value. Marks the nodes on mesh.node_indices."""
    values, v = state["values"], float(bc.get("value", 0.0))
    for d in (3 * n[:, None] + np.array(_dofs(bc["dofs"]))[None, :]).ravel():
        if values.get(int(d), v) != v:
            raise ValueError(f"{where}: dof {int(d)} is already given the value {values[int(d)]}, here {v}")
        values[int(d)] = v
    mesh.node_indices[n, 3] = _MARK_FIXED if len(bc["dofs"]) == 3 else _MARK_ROLLER


def _add_load(total: np.ndarray, n: np.ndarray, mesh: "HexMesher",
              state: dict[str, Any]) -> None:
    """Add the total force (3,) to state["force"], shared equally by the nodes n; marks them on
    mesh.node_indices."""
    state["force"].reshape(-1, 3)[n] += total / len(n)
    mesh.node_indices[n, 3] = _MARK_LOAD


def _apply_force(bc: dict[str, Any], n: np.ndarray, mesh: "HexMesher",
                 state: dict[str, Any], where: str) -> None:
    """The total force [Fx, Fy, Fz], shared equally by the nodes n (_add_load)."""
    _add_load(np.asarray(bc["total"], dtype=float), n, mesh, state)


def _apply_traction(bc: dict[str, Any], n: np.ndarray, mesh: "HexMesher",
                    state: dict[str, Any], where: str) -> None:
    """pressure * area along the unit direction, shared equally by the nodes n (_add_load); area "bbox" is the
    nodes' extent across the direction, each side at least half the smallest element size."""
    d = np.asarray(bc["direction"], dtype=float)
    d = d / np.linalg.norm(d)
    area = bc["area"]
    if area == "bbox":
        tol = 0.5 * float(np.min(mesh.elem_size))
        p = np.asarray(mesh.node_xyz, dtype=float)[n]
        a, b = [k for k in range(3) if d[k] == 0]
        area = max(np.ptp(p[:, a]), tol) * max(np.ptp(p[:, b]), tol)
    _add_load(bc["pressure"] * area * d, n, mesh, state)


_DOFS = dict(name="dofs", kind="dofs", label="Held in", doc="the directions held")
SUPPORT_AND_LOAD_KINDS = {
    "fixed": dict(
        role="support", apply=_apply_support, shown="Fixed support",
        help="Holds the selection still in the ticked directions.",
        values=[_DOFS],
        picks={t: (lambda c, d=d: {"dofs": d}) for t, d in _PICKED_SUPPORTS.items()}),
    "displacement": dict(
        role="support", apply=_apply_support, shown="Moved support",
        help="Moves the selection by a set distance in the ticked directions.",
        values=[dict(_DOFS, label="Moved in", doc="the directions moved"),
                dict(name="value", kind="number", measures="length", label="Distance", doc="the displacement")]),
    "force": dict(
        role="load", apply=_apply_force, shown="Force",
        help="A total force on the selection, shared equally by its nodes.",
        values=[dict(name="total", kind="vector", measures="force", label="Total force",
                     doc="the total force, split equally over the nodes")],
        picks={"force_xyz": _picked_force, None: _picked_force}),
    "traction": dict(
        role="load", apply=_apply_traction, check=_check_direction, shown="Pressure",
        help="A pressure on the selection, turned into a force: Pressure times Area, pushing along Direction, "
             "shared equally by its nodes. Tick Area from the selection to use the selection's size across the "
             "direction (Direction must then be along x, y or z).",
        values=[dict(name="pressure", kind="number", measures="stress", label="Pressure", doc="the pressure"),
                dict(name="direction", kind="vector", label="Direction", doc="the direction it pushes"),
                dict(name="area", kind="area", measures="area", label="Area",
                     doc="the area it acts on, or bbox: the selection's extent across the direction")]),
}

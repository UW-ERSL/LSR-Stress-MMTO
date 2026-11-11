"""
thermal_supports_and_loads.py
=============================
The thermal kinds of support and load: a fixed temperature and a heat source, what each does to the
equations K T = f, and the check that the temperature is held somewhere.

    problem/boundary_conditions.py   turns each entry of SUPPORT_AND_LOAD_KINDS into a BCKind
    problem/project.py               checks a project's supports and loads (validate), and builds them on the
                                     mesh before the run (supports_and_loads)

A kind's apply(bc, nodes, mesh, state, where) writes into the state of the temperature: state["values"]
{dof: the temperature it is held at} and state["force"] f, one number per dof. Here dof = node.
"""
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pyto.core.bc import BC
    from pyto.core.hex_mesher import HexMesher


# HELPER: the heat source kind's check (SUPPORT_AND_LOAD_KINDS)
def _check_total(bc: dict[str, Any]) -> list[str]:
    """A heat source of zero adds nothing: an error, as a list of short texts (empty when it is fine)."""
    return [] if float(bc["total"]) != 0.0 else ["total heat must not be zero"]


# HELPER: the fixed temperature kind's apply (SUPPORT_AND_LOAD_KINDS)
def _apply_fixed_temperature(bc: dict[str, Any], n: np.ndarray, mesh: "HexMesher",
                             state: dict[str, Any], where: str) -> None:
    """Holds the nodes n at the temperature; a node already held at another temperature is an error naming
    where (the support, "boundary condition 2")."""
    values, v = state["values"], float(bc.get("value", 0.0))
    for d in np.asarray(n, dtype=int):
        if values.get(int(d), v) != v:                       # two supports, two temperatures on one node
            raise ValueError(f"{where}: node {int(d)} is already held at {values[int(d)]}, here {v}")
        values[int(d)] = v


# HELPER: the heat source kind's apply (SUPPORT_AND_LOAD_KINDS)
def _apply_heat_source(bc: dict[str, Any], n: np.ndarray, mesh: "HexMesher",
                       state: dict[str, Any], where: str) -> None:
    """Adds the total heat to f, shared equally by the nodes n."""
    state["force"][np.asarray(n, dtype=int)] += float(bc["total"]) / len(n)


SUPPORT_AND_LOAD_KINDS = {
    "fixed_temperature": dict(
        role="support", apply=_apply_fixed_temperature, shown="Fixed temperature",
        help="Holds the selection at a set temperature.",
        values=[dict(name="value", kind="number", measures="temperature", label="Temperature",
                     doc="the temperature held")]),
    "heat_source": dict(
        role="load", apply=_apply_heat_source, check=_check_total, shown="Heat source",
        help="Puts heat into the selection, shared equally by its nodes.",
        values=[dict(name="total", kind="number", measures="heat", label="Total heat",
                     doc="the heat put in, split equally over the nodes")]),
}


# OPTIONAL: supports_and_loads_error, the check that the temperature is held
def supports_and_loads_error(mesh: "HexMesher", bcs: "dict[str, BC]") -> str | None:
    """Why the supports leave the temperature without a solution, or None: held nowhere, K is singular
    (any constant can be added to T), so at least one Fixed temperature is needed."""
    if len(bcs["T"].fixed_dofs) == 0:
        return "the temperature is held nowhere; add a Fixed temperature on some selection"
    return None

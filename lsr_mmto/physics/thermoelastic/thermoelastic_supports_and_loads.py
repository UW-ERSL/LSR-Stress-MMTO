"""
thermoelastic_supports_and_loads.py
===================================
The thermoelastic physics' kinds of support and load: structural's on the displacement (Fixed support,
Moved support, Force, Pressure) and thermal's on the temperature (Fixed temperature, Heat source). They are
the same entries, so each kind is one, shared by the physics that give it; each names its field, so a
problem of two fields knows which equations a support or load acts on.

supports_and_loads_error checks each field: the temperature held somewhere, and the part held so that it
cannot move.
"""
from typing import TYPE_CHECKING

import numpy as np

from lsr_mmto.physics.structural.structural_supports_and_loads import SUPPORT_AND_LOAD_KINDS as STRUCTURAL_KINDS
from lsr_mmto.physics.structural.structural_supports_and_loads import \
    supports_and_loads_error as structural_error
from lsr_mmto.physics.thermal.thermal_supports_and_loads import SUPPORT_AND_LOAD_KINDS as THERMAL_KINDS
from lsr_mmto.physics.thermal.thermal_supports_and_loads import supports_and_loads_error as thermal_error

if TYPE_CHECKING:
    from pyto.core.bc import BC
    from pyto.core.hex_mesher import HexMesher

# structural's kinds on the displacement and thermal's on the temperature, the same entries (so each kind is one,
# shared by the physics that give it)
SUPPORT_AND_LOAD_KINDS = {**STRUCTURAL_KINDS, **THERMAL_KINDS}


# OPTIONAL: supports_and_loads_error, the check that both fields are held
def supports_and_loads_error(mesh: "HexMesher", bcs: "dict[str, BC]") -> str | None:
    """Why the supports leave either field without a solution, or None: thermal's check on the temperature,
    the part held nowhere, structural's check on the displacement, then any rigid motion left free.
    Structural's check asks only about the directions the applied loads push, and the thermal force is not
    among them, so the part must be held against all six rigid motions here, loads or not: one left free makes
    the displacement's matrix singular, and the solve gives a displacement with an arbitrary drift in it."""
    if len(bcs["u"].fixed_dofs) == 0:
        return "the part is held nowhere; add a Fixed support on some selection"
    error = thermal_error(mesh, bcs) or structural_error(mesh, bcs)
    if error:
        return error
    free = _free_rigid_motions(mesh, bcs["u"].fixed_dofs)
    if free:
        return (f"the supports leave the part free to {' and '.join(free)}; add supports that hold it, for "
                f"example a Fixed support in that direction")
    return None


_MOTIONS = ("move in x", "move in y", "move in z", "rotate about x", "rotate about y", "rotate about z")


# HELPER: of supports_and_loads_error
def _free_rigid_motions(mesh: "HexMesher", fixed_dofs: np.ndarray) -> list[str]:
    """The rigid motions of the part the held dofs do not stop, by name (_MOTIONS); empty when all six are
    held. The six motions (three translations, three rotations about the part's centre, scaled by its size so
    all six are alike) taken at the held dofs only: a combination of them that leaves every held dof still is
    a motion left free, one for each singular value of that (held, 6) matrix that is zero. Each is named by
    the motion that makes up most of it."""
    xyz = np.asarray(mesh.node_xyz, dtype=float)
    r = (xyz - xyz.mean(axis=0)) / max(float(np.ptp(xyz, axis=0).max()), 1e-300)
    R = np.zeros((3 * len(xyz), 6))
    for d in range(3):
        R[d::3, d] = 1.0                                           # translations
    R[1::3, 3], R[2::3, 3] = -r[:, 2], r[:, 1]                     # about x: (0, -z, y)
    R[0::3, 4], R[2::3, 4] = r[:, 2], -r[:, 0]                     # about y: (z, 0, -x)
    R[0::3, 5], R[1::3, 5] = -r[:, 1], r[:, 0]                     # about z: (-y, x, 0)
    held = R[np.asarray(fixed_dofs, dtype=int)]
    _, s, Vt = np.linalg.svd(held, full_matrices=True)
    rank = int(np.sum(s > 1e-8 * s.max())) if s.size else 0
    return [_MOTIONS[int(np.argmax(np.abs(v)))] for v in Vt[rank:]]

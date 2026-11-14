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
    then structural's on the displacement, then the part held nowhere. Structural's check asks only about the
    directions the applied loads push, and the thermal force is not among them, so a part with no support at
    all must be refused here. A support that leaves some rigid motion free is not caught."""
    if len(bcs["u"].fixed_dofs) == 0:
        return "the part is held nowhere; add a Fixed support on some selection"
    return thermal_error(mesh, bcs) or structural_error(mesh, bcs)

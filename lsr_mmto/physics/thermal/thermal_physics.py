"""
thermal_physics.py
==================
The thermal solver, and one iteration's solve inside the torch tree. Steady heat conduction: one
temperature T at each node, and per element a conductivity k from its material:

    K T = f          K_e = SIMP(x_phys_e) * k_e * K_unit

K_unit is the element matrix of a unit conductivity (8 x 8, one row per node of the element), f the heat
put in at each node. PyTO-AD's thermal solver (HexThermalFEA) assembles and solves exactly this, and its
solve is differentiable: autograd gives the gradient of anything computed from T, through the solve, back
to x_phys and to the latent points. This file only gives it a conductivity per element, the way the
structural solver is given E and nu per element.

    make_solver   optimize/optimization_loop.py, once per run, after the mesh and the supports and
                  loads are built
    analyse       optimize/design_model.py (MMTOModelAD.evaluate), once per iteration, after the density
                  filter and projection (m.x_phys) and the decode of the latent points (m.props)
"""
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path, float64 default
from lsr_mmto.pyto_ad_imports import hex_element_stiffness as _hex_ek, mat_lib
from lsr_mmto.pyto_ad_imports import topopt_material_model as ad_mm
from lsr_mmto.pyto_ad_imports import hex_thermal_fea

if TYPE_CHECKING:
    from pyto.autodiff.material_model import MaterialModel
    from pyto.core.bc import BC
    from pyto.core.hex_mesher import HexMesher
    from pyto.core.mat_lib import Material
    from pyto.solve.deflation import DeflationSolver
    from pyto.solve.solvers import Solvers
    from lsr_mmto.optimize.design_model import MMTOModelAD
    from lsr_mmto.optimize.solid_part_analysis import _Analysed

_DTYPE = torch.float64

CONDUCTIVITY = "Conductivity"           # the database column each element's conductivity comes from


# HELPER: the solver class make_solver builds
class ConductionHexFEA(hex_thermal_fea.HexThermalFEA):
    """PyTO-AD's thermal solver with a different conductivity in each element, as a torch tensor in the
    autograd tree. PyTO-AD's solve() multiplies its element matrix elem_stiff_torch by the density scaling
    of each element and assembles K; here elem_stiff_torch is made per element, k_e * K_unit, so the tree
    runs  T -> K -> x_phys, k -> z."""

    K_unit: torch.Tensor                 # the element matrix of a unit conductivity, (8, 8)
    edof: torch.Tensor                   # each element's dofs, (elements, 8), long
    total_force: torch.Tensor            # f, the heat put in at each dof
    k: torch.Tensor                      # set by set_conductivity, (elements,)
    elem_stiff_torch: torch.Tensor       # (elements, 8, 8), what PyTO-AD's solve() assembles

    def __init__(self, mesh: "HexMesher", bc: "BC", solver: "Solvers", mat_prop: "Material | None" = None,
                 **kwargs: Any) -> None:
        if mat_prop is None:                                   # PyTO-AD's constructor wants a material;
                                                               # the real values come in set_conductivity
            mat_prop = mat_lib.create_material_with_defaults(name="unit", thermal_conductivity=1.0)
        # PyTO-AD's thermal solver makes the mesh's one-dof-per-node table (mesh.edofMatThermal) itself,
        # and points mesh.edofMat at it.
        super().__init__(mesh=mesh, mat_prop=mat_prop, bc=bc, solver=solver, **kwargs)
        K_unit = _hex_ek.hex8_stiffness_matrix_thermal(1.0, mesh.elem_size)
        self.K_unit = torch.tensor(K_unit, dtype=_DTYPE)                       # (8, 8)
        self.edof = torch.as_tensor(np.asarray(mesh.edofMatThermal), dtype=torch.long)   # (elements, 8)
        self.total_force = torch.tensor(np.asarray(bc.force, dtype=float), dtype=_DTYPE)  # f, one per dof

    def set_conductivity(self, k: torch.Tensor) -> None:
        """k: the conductivity of each element, torch, shape (elements,). solve() uses it next."""
        self.k = k
        self.elem_stiff_torch = k[:, None, None] * self.K_unit                 # (elements, 8, 8)

    def solve(self, x: torch.Tensor | np.ndarray | None = None, material_model: "MaterialModel | None" = None,
              **kwargs: Any) -> torch.Tensor:
        """PyTO-AD's solve(); also takes a numpy x, as the final check passes one. Returns T, one value per
        node."""
        if x is not None and not torch.is_tensor(x):
            x = torch.as_tensor(np.asarray(x, dtype=float), dtype=_DTYPE)
        return super().solve(x, material_model, **kwargs)


# ESSENTIAL: make_solver, the finite element solver
def make_solver(mesh: "HexMesher", mat_prop: "Material", bcs: "dict[str, BC]", linear_solver: "Solvers",
                dsolver: "DeflationSolver | None", elem_body_force: np.ndarray | None) -> ConductionHexFEA:
    """The thermal solver of the mesh, on the temperature's supports and loads (bcs["T"]). mat_prop is
    Project.build's placeholder material (the conductivities come each iteration); elem_body_force is not
    used. With no deflation space the loop passes PARDISO at every size, so dsolver is None."""
    return ConductionHexFEA(mesh=mesh, mat_prop=mat_prop, bc=bcs["T"], solver=linear_solver, dsolver=dsolver,
                            rtol=1e-8)


# ESSENTIAL: analyse, one iteration's solve
def analyse(m: "MMTOModelAD | _Analysed") -> None:
    """One iteration's solve, inside the torch tree: each element's conductivity from m.props, then T at
    m.x_phys with SIMP (K_e scaled by Kvoid + x_phys^p (1 - Kvoid); the loop raises p during the run, its
    thermal penalty continuation). Sets m.T (nodes,); and the solver's conductivity (set_conductivity)."""
    m.fe.set_conductivity(m.props[CONDUCTIVITY])
    m.T = m.fe.solve(m.x_phys, ad_mm.MaterialModel.SIMP)

"""
template_physics.py
===================
The job of this file: the solver of the physics, and one iteration's solve inside the torch tree.
Thermal's thermal_physics.py is this file filled in.

Who reads it:
    make_solver       optimize/optimization_loop.py, once per run, after the mesh and the supports and
                      loads are built
    analyse           optimize/design_model.py (MMTOModelAD.evaluate), once per iteration, after the
                      density filter and projection (m.x_phys) and the decode of the latent points (m.props)
    deflation_space   optimize/optimization_loop.py, for a problem above the direct solver's limit

How the gradient comes: analyse solves with torch operations only, and PyTO-AD's solvers are
differentiable, so autograd follows anything a quantity computes from the solution back through the
solve to the densities and the latent points. Nothing is differentiated by hand. numpy, .item(),
.detach() or float() inside analyse cut the tree, and the gradient of whatever comes after is
silently zero.

Glossary:
    node      a corner point of the voxel mesh
    dof       one unknown of the solve; a field of 3 unknowns per node has 3 dofs per node
    field     what is solved for, by name (FIELDS on the front page)
    element   one voxel of the mesh, 8 nodes; x_phys has one density per element
"""
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path, float64 default
from lsr_mmto.physics.structural.structural_physics import MultiMaterialHexFEA
from lsr_mmto.physics.thermal.thermal_physics import ConductionHexFEA
from lsr_mmto.pyto_ad_imports import topopt_material_model as ad_mm
if TYPE_CHECKING:
    from pyto.core.bc import BC
    from pyto.core.hex_mesher import HexMesher
    from pyto.core.mat_lib import Material
    from pyto.solve.deflation import DeflationSolver
    from pyto.solve.solvers import Solvers
    from lsr_mmto.optimize.design_model import MMTOModelAD
    from lsr_mmto.optimize.solid_part_analysis import _Analysed
    from lsr_mmto.problem.settings import RunSettings

# ESSENTIAL: make_solver, the finite element solver
def make_solver(mesh: "HexMesher", mat_prop: "Material", bcs: "dict[str, BC]", linear_solver: "Solvers",
                dsolver: "DeflationSolver | None", elem_body_force: np.ndarray | None) -> "ThermoelasticHexFEA":
    """The thermoelastic solver of the mesh: structural's solver on the displacement's supports and loads
    (bcs["u"]), holding thermal's on the temperature's (bcs["T"]). mat_prop is Project.build's placeholder
    material for the structural part (the real values come each iteration); the thermal part makes its own."""
    return ThermoelasticHexFEA(mesh=mesh, bcs=bcs, solver=linear_solver, dsolver=dsolver,
                               elem_body_force=elem_body_force, mat_prop=mat_prop)


# ESSENTIAL: analyse, one iteration's solve
def analyse(m: "MMTOModelAD | _Analysed") -> None:
    """One iteration's solve, inside the torch tree: the temperature m.T from each element's conductivity,
    the thermal force from m.T and each element's expansion, then the displacement m.u under the applied
    loads plus that force, both at the density m.x_phys with SIMP.

    Sets m.T (nodes,), m.f_th (3 * nodes,), m.u (3 * nodes,); and the solvers' material values
    (set_conductivity, set_material_fields)."""
    m.fe.thermal.set_conductivity(m.props["Conductivity"])
    m.T = m.fe.thermal.solve(m.x_phys, ad_mm.MaterialModel.SIMP)
    E, nu = elastic_constants(m.props, m.settings)
    m.f_th = thermal_force(m.fe, m.T, m.x_phys, E, nu, m.props["Thermal_Expansion"],
                           float(m.settings.reference_temperature))
    m.fe.set_material_fields(E, nu)
    m.u = m.fe.solve(m.x_phys, ad_mm.MaterialModel.SIMP, extra_force=m.f_th)

# HELPER: of analyse
def thermal_force(fe: "ThermoelasticHexFEA", T: torch.Tensor, x_phys: torch.Tensor, E: torch.Tensor,
                  nu: torch.Tensor, alpha: torch.Tensor, T_ref: float) -> torch.Tensor:
    """The thermal force on the displacement's dofs, (3 * nodes,): per element
    s(x_phys) * E * alpha / (1 - 2 nu) * H (T_e - T_ref), where s is the stiffness's SIMP scaling. Each
    element's 8 temperatures are taken with thermal's dof table and its 24 forces added in with structural's.
    Inside the torch tree: the gradient reaches T (so the thermal solve), x_phys and the properties."""
    s = ad_mm.get_structural_material_model_scaling_torch(x_phys, ad_mm.MaterialModel.SIMP)   # (N,)
    scale = s * E * alpha / (1.0 - 2.0 * nu)                                   # (N,)
    dT = T[fe.thermal.edof] - T_ref                                            # (N, 8), each element's own
    f_e = scale[:, None] * (dT @ fe.H.T)                                       # (N, 24)
    f = torch.zeros(3 * fe.mesh.num_nodes, dtype=f_e.dtype)
    return f.index_add(0, fe.edof.reshape(-1), f_e.reshape(-1))

# HELPER: of analyse
def elastic_constants(props: dict[str, torch.Tensor], settings: "RunSettings") -> tuple[torch.Tensor, torch.Tensor]:
    """E and nu per element from the decoded properties: the database's Poisson's ratio, or one value for every
    element when the setting asks for it or the database has none. Thermoelastic's own copy of structural's,
    reading its own settings."""
    E = props["Youngs_Modulus"]
    nu = (torch.full_like(E, float(settings.te_constant_poissons_ratio))
          if (settings.te_use_constant_poissons_ratio or "Poissons_Ratio" not in props) else props["Poissons_Ratio"])
    return E, nu
# ----------------------------------------------------------------------------- OPTIONAL entries of this file
# OPTIONAL: deflation_space, for the deflation linear solver
# def deflation_space(dsolver: "DeflationSolver", mesh: "HexMesher") -> None:
#     """Builds the deflation space of this physics' system on dsolver, so problems above the direct
#     solver's limit (DIRECT_SOLVER_DOF_CUTOFF in PyTO-AD) use DPCG. Structural's are the rigid-body
#     motions of groups of nodes. Left out, the direct solver runs at every size."""

# HELPER: the solver class make_solver builds
class ThermoelasticHexFEA(MultiMaterialHexFEA):
    """Structural's solver of the displacement, holding thermal's solver of the temperature. The core reads
    .bc, .mesh and .plot_mesh, so those are the displacement's; analyse solves the temperature with
    self.thermal, then the displacement with the thermal force added to its load."""

    thermal: ConductionHexFEA            # the temperature's solver, on the supports and loads of T
    H: torch.Tensor                      # the element's coupling matrix, (24, 8), from B, Phi and N only  # new

    def __init__(self, mesh: "HexMesher", bcs: "dict[str, BC]", solver: "Solvers",
                 dsolver: "DeflationSolver | None", elem_body_force: np.ndarray | None,
                 mat_prop: "Material | None" = None) -> None:
        # thermal first: each constructor points the shared mesh.edofMat at its own table, and the
        # displacement's must be the one left for code that reads the mesh afterwards
        self.thermal = ConductionHexFEA(mesh=mesh, bc=bcs["T"], solver=solver, dsolver=dsolver, rtol=1e-8)
        super().__init__(mesh=mesh, bc=bcs["u"], solver=solver, mat_prop=mat_prop, dsolver=dsolver,
                         rtol=1e-8, elem_body_force=elem_body_force)
        # PyTO-AD's matrix carries 1/(1 - 2 nu); at nu = 0 it is the same for every element and material,
        # and the thermal force applies each element's 1/(1 - 2 nu) itself
        self.H = torch.tensor(self.thermal.getHMatrix(*mesh.elem_size, 0.0), dtype=torch.float64)  # new
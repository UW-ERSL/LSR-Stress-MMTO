"""
thermoelastic_physics.py
========================
The thermoelastic solvers, and one iteration's solve inside the torch tree. Linear isotropic
thermoelasticity coupled one way: a steady temperature T from conduction, then the displacement u under
the applied loads plus the thermal force the temperature gives:

    K_T T = f_T          K_u u = f + f_th(T)

f_th,e = s(x_phys_e) * E_e * alpha_e / (1 - 2 nu_e) * H (T_e - T_ref), where s is the stiffness's SIMP
scaling, H the element's coupling matrix (24 x 8) and T_e the element's 8 temperatures. Thermal's solver
(ConductionHexFEA) solves the temperature and structural's (MultiMaterialHexFEA) the displacement; both
solves are differentiable, so autograd gives the gradient of anything computed from T and u back to x_phys
and to the latent points. Nothing is differentiated by hand.

Who reads it:
    make_solver       optimize/run_setup.py, once per run, after the mesh and the supports and
                      loads are built
    analyse           optimize/design_model.py (MMTOModelAD.evaluate), once per iteration, after the
                      density filter and projection (m.x_phys) and the decode of the latent points (m.props)
    deflation_space   optimize/run_setup.py, for a problem above the direct solver's limit
    set_simp_penalties  optimize/run_setup.py at the start of the run, optimize/optimization_loop.py at
                      each step of SIMP continuation
"""
from typing import TYPE_CHECKING

import numpy as np
import torch
import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path, float64 default
from lsr_mmto.physics.structural import structural_physics
from lsr_mmto.physics.structural.structural_physics import MultiMaterialHexFEA
from lsr_mmto.physics.thermal import thermal_physics
from lsr_mmto.physics.thermal.thermal_physics import ConductionHexFEA
from lsr_mmto.pyto_ad_imports import topopt_material_model as ad_mm
if TYPE_CHECKING:
    from pyto.core.bc import BC
    from pyto.core.hex_mesher import HexMesher
    from pyto.core.mat_lib import Material
    from pyto.solve.deflation import DeflationSolver
    from pyto.solve.solvers import Solvers
    from lsr_mmto.optimize.design_model import MMTOModelAD
    from lsr_mmto.optimize.iterative_solve_check import CheckedDeflationSolver
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
def deflation_space(dsolver: "CheckedDeflationSolver", mesh: "HexMesher") -> None:
    """Structural's and thermal's: the displacement's system is structural's (its space the rigid-body motions
    of groups of nodes) and the temperature's is thermal's (a constant on each of the same groups). Both solves
    share dsolver; each solver takes the space of its own size (CheckedDeflationSolver.space_for)."""
    structural_physics.deflation_space(dsolver, mesh)
    thermal_physics.deflation_space(dsolver, mesh)


# OPTIONAL: set_simp_penalties, for SIMP_PENALTIES (the front page)
def set_simp_penalties(solver: "ThermoelasticHexFEA", values: dict[str, float]) -> None:
    """Structural's for the displacement (and the thermal force, which scales as the stiffness does) and
    thermal's for the temperature's solver."""
    structural_physics.set_simp_penalties(solver, values)
    thermal_physics.set_simp_penalties(solver.thermal, values)


# HELPER: the solver class make_solver builds
class ThermoelasticHexFEA(MultiMaterialHexFEA):
    """Structural's solver of the displacement, holding thermal's solver of the temperature. The core reads
    .bc, .mesh and .plot_mesh, so those are the displacement's; analyse solves the temperature with
    self.thermal, then the displacement with the thermal force added to its load."""

    thermal: ConductionHexFEA            # the temperature's solver, on the supports and loads of T
    H: torch.Tensor                      # the element's coupling matrix, (24, 8), from B, Phi and N only

    def __init__(self, mesh: "HexMesher", bcs: "dict[str, BC]", solver: "Solvers",
                 dsolver: "DeflationSolver | None", elem_body_force: np.ndarray | None,
                 mat_prop: "Material | None" = None) -> None:
        # each solver keeps its own dof table (.edof); PyTO-AD's constructors also point the mesh's shared
        # mesh.edofMat at theirs, so it is set to the displacement's (the field the core reads, .bc) at the end,
        # whichever is built first
        self.thermal = ConductionHexFEA(mesh=mesh, bc=bcs["T"], solver=solver, dsolver=dsolver, rtol=1e-8)
        super().__init__(mesh=mesh, bc=bcs["u"], solver=solver, mat_prop=mat_prop, dsolver=dsolver,
                         rtol=1e-8, elem_body_force=elem_body_force)
        mesh.edofMat = mesh.edofMatStructural
        # PyTO-AD's matrix carries 1/(1 - 2 nu); at nu = 0 it is the same for every element and material,
        # and the thermal force applies each element's 1/(1 - 2 nu) itself
        self.H = torch.tensor(self.thermal.getHMatrix(*mesh.elem_size, 0.0), dtype=torch.float64)

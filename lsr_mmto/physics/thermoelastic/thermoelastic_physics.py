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
and to the latent points. Nothing is differentiated by hand. Young's modulus, the expansion coefficient and the
yield strength may change with each element's temperature (properties_at_temperature); the conductivity does not.

Who reads it:
    make_solver       optimize/run_setup.py, once per run, after the mesh and the supports and
                      loads are built
    analyse           optimize/design_model.py (MMTOModelAD.evaluate), once per iteration, after the
                      density filter and projection (m.x_phys) and the decode of the latent points (m.props)
    deflation_space   optimize/run_setup.py, for a problem above the direct solver's limit
    set_simp_penalties  optimize/run_setup.py at the start of the run, optimize/optimization_loop.py at
                      each step of SIMP continuation
"""
import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path, float64 default
from lsr_mmto.physics.structural import structural_physics
from lsr_mmto.physics.structural.structural_physics import MultiMaterialHexFEA, von_mises
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

    It takes m.props as decoded (at the reference temperature) and replaces it with the values at each element's
    temperature (properties_at_temperature), which everything after the temperature solve reads. Last, the
    relaxed von Mises stress m.svm of the mechanical strain (thermoelastic_stress), which the stress quantities
    read.

    Sets m.T (nodes,), m.props, m.f_th (3 * nodes,), m.u (3 * nodes,), m.svm (N,); and the solvers' material
    values (set_conductivity, set_material_fields) and stress relaxation exponent (the run's setting)."""
    T_ref = float(m.settings.reference_temperature)
    q = float(m.settings.te_stress_relaxation_exponent)
    if not (q > 0.0 and math.isfinite(q)):
        raise ValueError(f"Stress relaxation exponent must be a number above 0, not {q:g}")
    m.fe.thermal.set_conductivity(m.props["Conductivity"])
    m.T = m.fe.thermal.solve(m.x_phys, ad_mm.MaterialModel.SIMP)
    dT = m.T[m.fe.thermal.edof].mean(dim=1) - T_ref                           # (N,), at each element's centre
    m.props = properties_at_temperature(m.props, dT, T_ref)
    E, nu = elastic_constants(m.props, m.settings)
    m.f_th = thermal_force(m.fe, m.T, m.x_phys, E, nu, m.props["Thermal_Expansion"], T_ref)
    m.fe.set_material_fields(E, nu)
    m.fe.stress_relaxation = q
    m.u = m.fe.solve(m.x_phys, ad_mm.MaterialModel.SIMP, extra_force=m.f_th)
    m.svm = von_mises(thermoelastic_stress(m.fe, m.u, m.x_phys, m.props["Thermal_Expansion"], dT))

# the properties that change with temperature, each with the database column of its slope (thermoelastic_module.py)
TEMPERATURE_SLOPES = {"Youngs_Modulus": "Youngs_Modulus_Temperature_Slope",
                      "Thermal_Expansion": "Thermal_Expansion_Temperature_Slope",
                      "Yield_Strength": "Yield_Strength_Temperature_Slope"}

# HELPER: of analyse
def properties_at_temperature(props: dict[str, torch.Tensor], dT: torch.Tensor,
                              T_ref: float) -> dict[str, torch.Tensor]:
    """A new dict of the properties at each element's temperature: each property P of TEMPERATURE_SLOPES whose
    slope k the database has becomes P (1 + k dT), dT (N,) the element's centre temperature (the mean of its
    nodes' temperatures) minus the reference temperature T_ref; the rest as they are. The thermal expansion
    coefficient is the secant one from T_ref, so the thermal strain is alpha(T) (T - T_ref). Inside the torch
    tree: the gradient reaches T (so the thermal solve) and the slopes. A slope that makes a property zero or
    negative at some element is refused: the straight line does not reach that far."""
    out = dict(props)
    for name, slope in TEMPERATURE_SLOPES.items():
        if name not in props or slope not in props:
            continue
        factor = 1.0 + props[slope] * dT
        if bool(torch.any(factor <= 0.0)):
            worst = int(torch.argmin(factor))
            raise ValueError(f"The {slope} column makes {name} zero or negative at {float(dT[worst]) + T_ref:.6g} "
                             f"degrees (element {worst}). A straight line does not reach that far: use a smaller "
                             f"slope, or keep the temperatures closer to the reference temperature.")
        out[name] = props[name] * factor
    return out

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
def thermoelastic_stress(fe: "ThermoelasticHexFEA", u: torch.Tensor, x_phys: torch.Tensor, alpha: torch.Tensor,
                         dT: torch.Tensor) -> torch.Tensor:
    """The relaxed stress at each element's centre, (N, 6): x_phys^q D (B u_e - alpha dT [1, 1, 1, 0, 0, 0]), the
    mechanical strain (the total strain less the thermal strain) times the element's D (its lam and mu, from
    set_material_fields), q the solver's stress relaxation exponent. alpha and dT (N,) are each element's
    expansion coefficient and centre temperature rise, as the thermal force has them, so a part free to grow
    has no stress."""
    strain = u[fe.edof] @ fe.B.T                                               # (N, 6), total
    thermal = (alpha * dT)[:, None] * torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=strain.dtype)
    D = fe.lam[:, None, None] * fe.D_lam + fe.mu[:, None, None] * fe.D_mu
    stress = torch.einsum("eij,ej->ei", D, strain - thermal)
    return stress * (x_phys ** fe.stress_relaxation)[:, None]

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

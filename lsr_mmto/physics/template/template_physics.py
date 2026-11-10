"""
template_physics.py
===================
What this file is for: the solver of the template physics, and one iteration's solve inside the
torch tree. Start here after template_module.py.

Who reads it:
    make_solver   optimize/optimization_loop.py, once per run, after the mesh and the supports
                  and loads are built
    analyse       optimize/design_model.py (MMTOModelAD.evaluate), once per MMA iteration, after
                  the density filter and projection (m.x_phys) and the decode of the latent points
                  (m.props)

The physics here is steady diffusion: one unknown c at each node (a concentration, or a
temperature: the equation is the same), and per element a diffusion coefficient k:

    K c = f          K_e = SIMP(x_phys_e) * k_e * K_unit

K_unit is the element matrix of a unit coefficient (8 x 8, one row per node of the element).
PyTO-AD's thermal solver (HexThermalFEA) already assembles and solves exactly this, and its solve
is differentiable: torch's autograd gives the gradient of anything computed from c, through the
solve, back to x_phys and to the latent points. The template only gives it a coefficient per
element, the way the structural solver is given E and nu per element.

Glossary:
    node      a corner point of the voxel mesh
    dof       one unknown of the solve; here one per node, so dof number = node number
    field     what is solved for, by name: here "c", one dof per node (FIELDS in template_module.py)
    element   one voxel of the mesh, 8 nodes; x_phys has one density per element

To make your own physics, change the lines marked FILL. The rest can stay as it is.
"""
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  # keep as is: puts PyTO-AD on sys.path, float64 default
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

# FILL: the database column your coefficient comes from. Youngs_Modulus only stands in here, because
# every shipped database has it; a real diffusion or heat physics reads its own attribute (for example
# a conductivity), which must first be added to the material attributes.
COEFFICIENT = "Youngs_Modulus"


class DiffusionHexFEA(hex_thermal_fea.HexThermalFEA):
    """PyTO-AD's thermal solver with a different diffusion coefficient in each element, as a torch
    tensor in the autograd tree.

    PyTO-AD's solve() multiplies its element matrix elem_stiff_torch by the density scaling of each
    element and assembles K. Here elem_stiff_torch is made per element, k_e * K_unit, so the tree
    runs  c -> K -> x_phys, k -> z.

    FILL: for another physics, subclass the PyTO-AD solver of that physics instead, and replace
    set_coefficient by what your elements need (the structural solver takes E and nu per element).
    """

    K_unit: torch.Tensor                 # the element matrix of a unit coefficient, (8, 8)
    edof: torch.Tensor                   # each element's dofs, (elements, 8), long
    total_force: torch.Tensor            # f, one per dof
    k: torch.Tensor                      # set by set_coefficient, (elements,)
    elem_stiff_torch: torch.Tensor       # (elements, 8, 8), what PyTO-AD's solve() assembles

    def __init__(self, mesh: "HexMesher", bc: "BC", solver: "Solvers", mat_prop: "Material | None" = None,
                 **kwargs: Any) -> None:
        if mat_prop is None:                                   # PyTO-AD's constructor wants a material;
                                                               # the real values come in set_coefficient
            mat_prop = mat_lib.create_material_with_defaults(name="unit", thermal_conductivity=1.0)
        # PyTO-AD's thermal solver makes the mesh's one-dof-per-node table (mesh.edofMatThermal) itself,
        # and points mesh.edofMat at it.
        super().__init__(mesh=mesh, mat_prop=mat_prop, bc=bc, solver=solver, **kwargs)
        K_unit = _hex_ek.hex8_stiffness_matrix_thermal(1.0, mesh.elem_size)
        self.K_unit = torch.tensor(K_unit, dtype=_DTYPE)                       # (8, 8)
        self.edof = torch.as_tensor(np.asarray(mesh.edofMatThermal), dtype=torch.long)   # (elements, 8)
        self.total_force = torch.tensor(np.asarray(bc.force, dtype=float), dtype=_DTYPE)  # f, one per dof

    def set_coefficient(self, k: torch.Tensor) -> None:
        """k: the coefficient of each element, torch, shape (elements,). solve() uses it next."""
        self.k = k
        self.elem_stiff_torch = k[:, None, None] * self.K_unit                 # (elements, 8, 8)

    def solve(self, x: torch.Tensor | np.ndarray | None = None, material_model: "MaterialModel | None" = None,
              **kwargs: Any) -> torch.Tensor:
        """PyTO-AD's solve(); also takes a numpy x, as the final check passes one. Returns c, one value per node."""
        if x is not None and not torch.is_tensor(x):
            x = torch.as_tensor(np.asarray(x, dtype=float), dtype=_DTYPE)
        return super().solve(x, material_model, **kwargs)


# ----------------------------------------------------------------------------- the module's entries
def make_solver(mesh: "HexMesher", mat_prop: "Material", bcs: "dict[str, BC]", linear_solver: "Solvers",
                dsolver: "DeflationSolver | None", elem_body_force: np.ndarray | None) -> DiffusionHexFEA:
    """The solver of this physics. Called once per run by optimize/optimization_loop.py.

    mesh            the voxel mesh (PyTO-AD's)
    mat_prop        a placeholder material from Project.build; the real properties come each iteration
    bcs             {field name: PyTO-AD BC}, one per field in FIELDS, built from the supports and loads
    linear_solver   PARDISO, or DPCG for a large problem when the physics has a deflation_space
                    (template_module.py)
    dsolver         the deflation solver, used only with DPCG
    elem_body_force a body force per element, or None (not used by this physics)
    """
    return DiffusionHexFEA(mesh=mesh, mat_prop=mat_prop, bc=bcs["c"],     # FILL: your field's name
                           solver=linear_solver, dsolver=dsolver, rtol=1e-8)


def analyse(m: "MMTOModelAD | _Analysed") -> None:
    """One iteration's solve, inside the torch tree. Called by MMTOModelAD.evaluate.

    m is the iteration's model (optimize/design_model.py). What it already holds:
        m.x_phys     density of each element after the filter and projection, torch (elements,)
        m.props      {database column: value per element}, decoded from the latent points, torch
        m.fe         the solver make_solver returned
        m.settings   every setting of the project, as m.settings.<name>
        m.N          the number of elements
    What this must leave on m: whatever your quantities read (template_quantities.py). Here, m.c.

    Sets m.c (nodes,); and the solver's coefficient (set_coefficient).
    """
    k = m.props[COEFFICIENT] * float(m.settings.template_coefficient_scale)   # FILL: your properties
    m.fe.set_coefficient(k)
    # SIMP: K_e is scaled by Kvoid + x_phys^p (1 - Kvoid), PyTO-AD's thermal material model; the loop
    # raises p during the run (its thermal penalty continuation).
    m.c = m.fe.solve(m.x_phys, ad_mm.MaterialModel.SIMP)                       # FILL: your solve

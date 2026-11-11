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

import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path, float64 default

if TYPE_CHECKING:
    from pyto.core.bc import BC
    from pyto.core.hex_mesher import HexMesher
    from pyto.core.mat_lib import Material
    from pyto.solve.deflation import DeflationSolver
    from pyto.solve.solvers import Solvers
    from lsr_mmto.optimize.design_model import MMTOModelAD
    from lsr_mmto.optimize.solid_part_analysis import _Analysed


# ESSENTIAL: make_solver, the finite element solver
def make_solver(mesh: "HexMesher", mat_prop: "Material", bcs: "dict[str, BC]", linear_solver: "Solvers",
                dsolver: "DeflationSolver | None", elem_body_force: np.ndarray | None) -> Any:
    """Returns the solver of this physics on the mesh; called once per run.

    mesh            the voxel mesh (PyTO-AD's HexMesher)
    mat_prop        a placeholder material from Project.build; the real properties come each iteration
    bcs             {field name: PyTO-AD BC}, one per field in FIELDS, built from the supports and loads
    linear_solver   PARDISO, or DPCG for a large problem when the physics has a deflation_space
    dsolver         the deflation solver, used only with DPCG
    elem_body_force a body force per element, or None

    The solver is usually PyTO-AD's solver of the physics, subclassed (a Helper below) so that each element
    gets its own material values as torch tensors: thermal's ConductionHexFEA, structural's
    MultiMaterialHexFEA. What it returns is m.fe in analyse and in the quantities.
    """
    raise NotImplementedError("FILL: make_solver, the solver of this physics")


# ESSENTIAL: analyse, one iteration's solve
def analyse(m: "MMTOModelAD | _Analysed") -> None:
    """One iteration's solve, inside the torch tree; called once per iteration.

    m is the iteration's model (optimize/design_model.py). What it already holds:
        m.x_phys     the density of each element after the filter and projection, torch (elements,)
        m.props      {database column: value per element}, decoded from the latent points, torch
        m.fe         the solver make_solver returned
        m.settings   every setting of the project, as m.settings.<name>
        m.N          the number of elements
    What it must leave on m: whatever the quantities read, e.g. m.T for thermal, m.u and m.svm for
    structural. Its docstring ends with "Sets m.<each one>."
    """
    raise NotImplementedError("FILL: analyse, one iteration's solve")


# ----------------------------------------------------------------------------- OPTIONAL entries of this file
# OPTIONAL: deflation_space, for the deflation linear solver
# def deflation_space(dsolver: "DeflationSolver", mesh: "HexMesher") -> None:
#     """Builds the deflation space of this physics' system on dsolver, so problems above the direct
#     solver's limit (DIRECT_SOLVER_DOF_CUTOFF in PyTO-AD) use DPCG. Structural's are the rigid-body
#     motions of groups of nodes. Left out, the direct solver runs at every size."""

# ----------------------------------------------------------------------------- Helpers
# Your solver class goes here: PyTO-AD's solver of the physics subclassed so each element gets its own
# material values, and any small function analyse or the solver needs. Mark each one
# "# HELPER: of <what uses it>".

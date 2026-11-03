"""
structural_physics.py
=====================
The structural physics of one iteration: PyTO-AD's structural solver with a different (E, nu) in
each element, as torch tensors in the AD tree, and the element stress, von Mises stress and
p-norm that the structural quantities read. make_solver and analyse are the module's entries
of the same names (structural_module.py).

    u      -> K -> x_phys, E, nu             K_e = SIMP(x_phys) * (lam_e K_lam + mu_e K_mu)

The FE solve is PyTO-AD's differentiable sparse solve, a custom autograd node whose backward is
the adjoint solve, reusing the forward factorization. PyTO-AD is imported and inherited, never
edited. PyTO-AD sets torch's default dtype to float64 on import; everything here is float64.
"""
import numpy as np
import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path
import pyto.physics.hex_element_stiffness as _hex_ek
import pyto.autodiff.material_model as ad_mm
from pyto.physics.structural.hex_structural_fea import HexStructuralFEA as _HexStructuralFEA_AD

from lsr_mmto.elements.hex_element import lame_basis, lame_parameters, structural_B_matrix

_DTYPE = torch.float64


# ----------------------------------------------------------------------------- FE
class MultiMaterialHexFEA(_HexStructuralFEA_AD):
    """PyTO-AD's structural FEA with per-element (E, nu) as torch tensors in the AD tree.

    set_material_fields() is the AD entry point; PyTO-AD's solve() then assembles K from the
    per-element stiffness it is handed, so the tree runs K -> E, nu -> z.
    set_material() and solve(numpy x) make it a drop-in for PyTO's HexStructuralFEA, so the
    driver, diagnostics and post-processing call it exactly as before.
    """

    def __init__(self, mesh, bc, solver, mat_prop=None, **kwargs):
        if mat_prop is None:
            import pyto.core.mat_lib as mat_lib
            mat_prop = mat_lib.create_material_with_defaults(name="unit", youngs_modulus=1.0)
        super().__init__(mesh=mesh, mat_prop=mat_prop, bc=bc, solver=solver, **kwargs)
        es = mesh.elem_size
        K_lam, K_mu = lame_basis(lambda E, nu: _hex_ek.hex8_stiffness_matrix_structural(E, nu, es))
        D_lam, D_mu = lame_basis(_hex_ek.isotropic_constitutive_matrix)
        self.K_lam = torch.tensor(K_lam, dtype=_DTYPE)
        self.K_mu = torch.tensor(K_mu, dtype=_DTYPE)
        self.D_lam = torch.tensor(D_lam, dtype=_DTYPE)
        self.D_mu = torch.tensor(D_mu, dtype=_DTYPE)
        self.B = torch.tensor(structural_B_matrix(es), dtype=_DTYPE)
        self.edof = torch.as_tensor(np.asarray(mesh.edofMatStructural), dtype=torch.long)

    def set_material_fields(self, E, nu):
        """Per-element E, nu (torch, shape (N,)) -> element stiffness used by solve()."""
        self.lam, self.mu = lame_parameters(E, nu)
        self.elem_stiff_torch = (self.lam[:, None, None] * self.K_lam
                                 + self.mu[:, None, None] * self.K_mu)

    def set_material(self, mat_prop):
        """PyTO-style material update: one Material or a per-element list.

        Refreshes the element stiffness that solve() uses (PyTO-AD's own constructor-time copy
        is never refreshed, being single-material by design) and keeps mat_prop for postprocess().
        """
        mats = mat_prop if isinstance(mat_prop, list) else [mat_prop] * self.mesh.num_elems
        E = torch.tensor([m.youngs_modulus for m in mats], dtype=_DTYPE)
        nu = torch.tensor([m.poissons_ratio for m in mats], dtype=_DTYPE)
        self.set_material_fields(E, nu)
        self.mat_prop = mat_prop

    def solve(self, x=None, material_model=None, **kwargs):
        """PyTO-AD's solve(); also accepts a numpy x, as PyTO's solver did."""
        if x is not None and not torch.is_tensor(x):
            x = torch.as_tensor(np.asarray(x, dtype=float), dtype=_DTYPE)
        return super().solve(x, material_model, **kwargs)

    def postprocess(self):
        """Element-center strains, stresses, von Mises, p-norm: same outputs as PyTO-AD's postprocess().

        Replaces it rather than calling it, for two reasons:
          - D comes from the per-element lam, mu already held (set_material_fields), vectorized;
            PyTO-AD's version rebuilds D from a list of Material objects, one call per element.
          - pNormStress uses the CURRENT p-norm exponent: PyTO-AD's FEA module copies
            _PNORM_EXPONENT at import (from ... import), so a per-problem override never reaches it.
        Stress relaxation is xb^q, exactly as element_stress() (the objective) and PyTO's postprocess.
        """
        with torch.no_grad():
            u = self.sol.detach()
            x = self.x.detach()
            strain = u[self.edof] @ self.B.T                                   # (N, 6), total
            self.strainComponents = strain.numpy().copy()
            if self.elem_thermal_strain is not None:                           # elastic part only
                strain = strain - torch.as_tensor(self.elem_thermal_strain, dtype=_DTYPE)
            D = (self.lam[:, None, None] * self.D_lam + self.mu[:, None, None] * self.D_mu)
            stress = torch.einsum("eij,ej->ei", D, strain)
            self.stressComponents = stress.numpy()
            e_stress = stress * (x ** ad_mm.get_stress_relaxation_factor())[:, None]
            self.vonMisesStress = von_mises(e_stress).numpy()
            self.elemStrainEnergy = (0.5 * torch.sum(strain * e_stress, dim=1)).numpy()
        p = ad_mm.get_pNorm_exponent()
        self.pNormStress = (np.sum(self.vonMisesStress ** p)) ** (1.0 / p)

    def element_stress(self, u, xb):
        """Relaxed element-center stress, xb^q * D(E, nu) * B * u_e, shape (N, 6)."""
        strain = u[self.edof] @ self.B.T                                    # (N, 6)
        D = (self.lam[:, None, None] * self.D_lam + self.mu[:, None, None] * self.D_mu)
        stress = torch.einsum("eij,ej->ei", D, strain)
        return stress * (xb ** ad_mm.get_stress_relaxation_factor())[:, None]


def von_mises(s):
    vm2 = 0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2) \
        + 3.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
    return torch.sqrt(torch.clamp(vm2, min=1e-24))


def pnorm(v, p):
    """(sum v^p)^(1/p), as in PyTO's postprocess (no 1/N)."""
    return torch.sum(v ** p) ** (1.0 / p)


# ----------------------------------------------------------------------------- the module's entries
def set_pnorm_exponent(p):
    """The p-norm exponent where PyTO-AD's material model keeps it, read by every p-norm."""
    ad_mm._PNORM_EXPONENT = float(p)


def elastic_constants(props, settings):
    """E and nu per element from the decoded properties: the database's Poisson's ratio, or one value
    for every element when the setting asks for it or the database has none."""
    E = props["Youngs_Modulus"]                 # column names are registry names (databases.ATTRIBUTES)
    nu = (torch.full_like(E, float(settings.constant_poissons_ratio))
          if (settings.use_constant_poissons_ratio or "Poissons_Ratio" not in props) else props["Poissons_Ratio"])
    return E, nu


def make_solver(mesh, mat_prop, bcs, linear_solver, dsolver, elem_body_force):
    """The structural solver of the mesh, with the boundary condition of the displacement field u."""
    return MultiMaterialHexFEA(mesh=mesh, mat_prop=mat_prop, bc=bcs["u"], solver=linear_solver, dsolver=dsolver,
                               rtol=1e-8, elem_body_force=elem_body_force)


def deflation_space(dsolver, mesh):
    """The deflation space of the structural system on dsolver, for DPCG on large problems: the mesh in
    groups of nodes, and the rigid-body motions of each group (three unknowns per node)."""
    nGroups = dsolver.getRecommendedNumberOfGroups(nDOF=3 * mesh.num_nodes)
    dsolver.create_deflation_groups(mesh, nGroups)
    dsolver.create_deflation_matrix(mesh)


def analyse(m):
    """One iteration's solve, inside the torch tree: E and nu from m.props, the displacement m.u at the
    density m.x_phys, and the relaxed von Mises stress m.svm that the stress quantities read."""
    E, nu = elastic_constants(m.props, m.settings)
    m.fe.set_material_fields(E, nu)
    m.u = m.fe.solve(m.x_phys, ad_mm.MaterialModel.SIMP)
    m.svm = von_mises(m.fe.element_stress(m.u, m.x_phys))

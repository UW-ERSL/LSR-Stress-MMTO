"""
Autodiff (PyTorch) model for pure-structural MMTO, built on PyTO-AD.

The forward pass is written once; every gradient comes from torch autograd. The FE solve is
PyTO-AD's differentiable sparse solve, a custom autograd node whose backward is the adjoint
solve, reusing the forward factorization. PyTO-AD is imported and inherited, never edited.

TREE (arrow = "depends on")
    J      -> svm -> s -> x_phys, E, nu, u   objective (to_params.Objective):
                                               PNORM_STRESS                 p-norm of relaxed von Mises
                                               PNORM_STRESS, FAILURE_FACTOR mean p-norm of svm / Y, Y -> z
                                               COMPLIANCE                   F . u
                                               MASS                         sum rho x_phys V, rho -> z
    g_k    -> x_phys, z                      constraints (to_params.Constraints), each g_k <= 0:
                                               VOLUME_FRACTION              global, mean(x_phys)
                                               VOLUME_FRACTION, material_id per material, softmax weights
                                               MASS                         sum rho x_phys V, rho -> z
                                               COST                         sum rho c x_phys V, rho, c -> z
                                               COMPLIANCE                   F . u
                                               STRESS_FAILURE_FACTOR        mean p-norm of svm / Y
    Pz     -> x_phys, z                      latent smoothing penalty
    Dp     -> x_phys (frozen), z             distance to the nearest real material
    E, nu, Y, rho -> z                       VAE decoder (whichever attributes the database has)
    u      -> K -> x_phys, E, nu             K_e = SIMP(x_phys) * (lam_e K_lam + mu_e K_mu)
    x_phys -> x_filt -> x                    Heaviside(density filter(x))
Any other objective or constraint raises NotImplementedError rather than run something different.
The driver (MainPureStructural.py) combines them: OBJ = J/J0 + w*Pz + gamma*Dp/Sd0.

Element stiffness and constitutive matrix are linear in the two Lame parameters:
    D(E, nu)   = lam * D_lam + mu * D_mu
    K_e(E, nu) = lam * K_lam + mu * K_mu
exact for every linear isotropic material, constant nu or not. The basis matrices are
identified once from PyTO-AD's own element routines, so the integration is identical.

PyTO-AD sets torch's default dtype to float64 on import; everything here is float64.
"""
import numpy as np
import torch

import PyTOImports  # noqa: F401  puts PyTO-AD on sys.path
import pyto.physics.hex_element_stiffness as _hex_ek
import pyto.autodiff.material_model as ad_mm
from pyto.physics.structural.hex_structural_fea import HexStructuralFEA as _HexStructuralFEA_AD

_DTYPE = torch.float64


# ----------------------------------------------------------------------------- Lame basis
def lame_parameters(E, nu):
    """(lam, mu) from (E, nu); works on torch tensors and keeps them in the tree."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


def _lame_basis(fn):
    """Basis (A_lam, A_mu) with fn(E, nu) = lam*A_lam + mu*A_mu, for fn linear in D.

    nu = 0 gives lam = 0, which isolates A_mu; a second nu then isolates A_lam.
    """
    lam0, mu0 = lame_parameters(1.0, 0.0)
    A_mu = np.asarray(fn(1.0, 0.0)) / mu0
    lam1, mu1 = lame_parameters(1.0, 0.25)
    A_lam = (np.asarray(fn(1.0, 0.25)) - mu1 * A_mu) / lam1
    return A_lam, A_mu


def structural_B_matrix(elem_size):
    """6x24 strain-displacement matrix at the hex element center (same as PyTO postprocess)."""
    gradN = (1.0 / 8.0) * np.array([
        [-1, 1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1],
    ], dtype=float)
    gradN = gradN * (2.0 / np.asarray(elem_size, dtype=float))[:, None]
    B = np.zeros((6, 24))
    for a in range(8):
        dx, dy, dz = gradN[:, a]
        c = 3 * a
        B[0, c] = dx
        B[1, c + 1] = dy
        B[2, c + 2] = dz
        B[3, c], B[3, c + 1] = dy, dx      # gamma_xy
        B[4, c], B[4, c + 2] = dz, dx      # gamma_xz
        B[5, c + 1], B[5, c + 2] = dz, dy  # gamma_yz
    return B


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
        K_lam, K_mu = _lame_basis(lambda E, nu: _hex_ek.hex8_stiffness_matrix_structural(E, nu, es))
        D_lam, D_mu = _lame_basis(_hex_ek.isotropic_constitutive_matrix)
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


# ----------------------------------------------------------------------------- design maps
class SparseOperator:
    """A constant sparse matrix (a filter) as scipy CSR, with its transpose, both built once."""

    def __init__(self, H):
        self.H = H.tocsr()
        self.Ht = self.H.T.tocsr()


class _SparseMatmul(torch.autograd.Function):
    """H @ x inside the tree: forward H @ x, backward H^T @ g, both scipy CSR products. Same result as
    torch.sparse.mm on a COO tensor, which is about 20 times slower for a large 3D filter."""

    @staticmethod
    def forward(ctx, x, op):
        ctx.op = op
        return torch.from_numpy(np.asarray(op.H @ x.detach().numpy()))

    @staticmethod
    def backward(ctx, g):
        return torch.from_numpy(np.asarray(ctx.op.Ht @ g.detach().numpy())), None


def sparse_matmul(op, x):
    """op.H @ x for x of shape (N,) or (N, k), differentiable w.r.t. x."""
    return _SparseMatmul.apply(x, op)


def density_filter(x, H, Hs):
    return sparse_matmul(H, x) / Hs


def heaviside(x_filt, beta, eta):
    tb = np.tanh(beta * eta)
    return (tb + torch.tanh(beta * (x_filt - eta))) / (tb + np.tanh(beta * (1.0 - eta)))


def decode(enc, z, use_constant_nu, nu_const):
    """E, nu per element, plus every decoded property, from the VAE decoder, inside the tree."""
    props = enc.getMaterialProperties(enc.vaeNet.decoder(z))
    E = props["Youngs_Modulus"]
    nu_key = next((k for k in ("Poissons_Ratio", "Poisson_Ratio", "poissons_ratio",
                               "poisson_ratio", "nu", "Nu") if k in props), None)
    nu = (torch.full_like(E, float(nu_const)) if (use_constant_nu or nu_key is None)
          else props[nu_key])
    return E, nu, props


# ----------------------------------------------------------------------------- penalties, constraints
def distance_penalty(xb, z, z_real, eps=1e-12):
    """Dp = sum_e xb_e * d_min(z_e) / n_active, with xb a frozen weight (no gradient)."""
    w = xb.detach()
    dist = torch.sqrt(torch.clamp(((z[:, None, :] - z_real[None, :, :]) ** 2).sum(-1), min=eps))
    d_min = dist.gather(1, dist.argmin(dim=1, keepdim=True))[:, 0]
    n_active = max(int((w > 0.1).sum()), 1)
    return torch.sum(w * d_min) / n_active


def latent_smoothing(xb, z, Hm, Hsm):
    """Pz = sum_e xb_e * || z_e - (Hm z)_e / Hsm_e ||^2  (latent smoothing, weight 1)."""
    zbar = sparse_matmul(Hm, z) / Hsm[:, None]
    return torch.sum(xb[:, None] * (z - zbar) ** 2)


def material_volume_fraction(xb, z, z_real, material_id, tau):
    """V_k = (1/N) sum_e xb_e * softmax_k(-||z_e - z_k||^2 / tau)."""
    logits = -((z[:, None, :] - z_real[None, :, :]) ** 2).sum(-1) / tau
    p = torch.softmax(logits, dim=1)
    return torch.sum(xb * p[:, material_id]) / xb.shape[0]


# ----------------------------------------------------------------------------- the forward pass
class MMTOModelAD:
    """The optimization callback's differentiable model, one torch tree per MMA iteration.

        x (raw MMA density) -> density filter -> Heaviside -> x_phys
        z (latents)         -> VAE decoder    -> E, nu (and Y, rho when the database has them)
        x_phys, E, nu       -> PyTO-AD solve  -> u -> element stress -> von Mises
        -> objective J, constraints, and the two penalties Pz (latent smoothing), Dp (distance)

    grad_phys(q) returns dq/d(x_phys, z) in the driver's layout [x (N) ; z latent-major (N*latentDim)],
    the same layout the manual sensitivities returned, so the driver can normalize, override
    and add terms exactly as before. chain_to_x(g) carries the x block back through the
    Heaviside projection and the density filter to the raw MMA variable.
    Every derivative comes from autograd; nothing is derived by hand.
    """

    def __init__(self, fe, enc, H_density, Hs_density, H_material, Hs_material, z_real,
                 to_params, use_constant_nu, nu_const, elem_volume):
        self.fe, self.enc, self.to_params = fe, enc, to_params
        self.N = fe.mesh.num_elems
        self.latentDim = enc.vae_params.latentDim
        self.H = SparseOperator(H_density)
        self.Hs = torch.as_tensor(np.asarray(Hs_density, dtype=float).ravel(), dtype=_DTYPE)
        self.Hm = SparseOperator(H_material)
        self.Hsm = torch.as_tensor(np.asarray(Hs_material, dtype=float).ravel(), dtype=_DTYPE)
        self.z_real = torch.as_tensor(np.asarray(z_real, dtype=float), dtype=_DTYPE)
        self.use_constant_nu, self.nu_const = bool(use_constant_nu), float(nu_const)
        self.elem_volume = float(elem_volume)

    def evaluate(self, zeta, beta, eta, use_heaviside=True, apply_density_filter=True):
        """Build the tree at design zeta = [x ; z latent-major]; returns J and the constraint list."""
        zeta = np.asarray(zeta, dtype=float).ravel()
        N, latentDim = self.N, self.latentDim
        self.x = torch.tensor(zeta[:N], dtype=_DTYPE, requires_grad=True)
        self.z = torch.tensor(zeta[N:].reshape(latentDim, N).T.copy(), dtype=_DTYPE, requires_grad=True)

        x_filt = density_filter(self.x, self.H, self.Hs) if apply_density_filter else self.x
        self.x_phys = heaviside(x_filt, beta, eta) if use_heaviside else x_filt

        E, nu, self.props = decode(self.enc, self.z, self.use_constant_nu, self.nu_const)
        self.fe.set_material_fields(E, nu)
        self.u = u = self.fe.solve(self.x_phys, ad_mm.MaterialModel.SIMP)
        self.svm = von_mises(self.fe.element_stress(u, self.x_phys))

        self.J = self._objective()
        self.cons = [self._constraint(qoi, opt, limit) for qoi, opt, limit in self.to_params.Constraints]
        return self.J, self.cons

    def _prop(self, name, qoi):
        """A decoded material property, or a clear error if the database has no such column."""
        if name not in self.props:
            raise KeyError(f"{qoi.name} needs a {name} column in the material database; "
                           f"this one has {', '.join(sorted(self.props))}")
        return self.props[name]

    def _compliance(self):
        """F . u, with F the total force PyTO-AD assembled, as main's total_force . sol."""
        return torch.dot(self.fe.total_force, self.u)

    def _mass(self, qoi):
        return torch.sum(self._prop("Density", qoi) * self.x_phys) * self.elem_volume

    def _ff_pnorm(self, qoi):
        """p-norm of sigma_vm,e / Y_e, mean form (1/N), as the manual code on main."""
        p = ad_mm.get_pNorm_exponent()
        Y = torch.clamp(self._prop("Yield_Strength", qoi), min=1e-12)
        return (torch.sum((self.svm / Y) ** p) / self.N) ** (1.0 / p)

    def _objective(self):
        qoi, opt = self.to_params.Objective[0], self.to_params.Objective[1]
        if qoi.name == "PNORM_STRESS" and opt == "FAILURE_FACTOR":
            return self._ff_pnorm(qoi)
        if qoi.name == "PNORM_STRESS":
            return pnorm(self.svm, ad_mm.get_pNorm_exponent())   # no 1/N, as PyTO's pNormStress
        if qoi.name == "COMPLIANCE":
            return self._compliance()
        if qoi.name == "MASS":
            return self._mass(qoi)
        raise NotImplementedError(f"objective {qoi.name} {opt} not in the AD path yet")

    def _constraint(self, qoi, opt, limit):
        if qoi.name == "VOLUME_FRACTION" and isinstance(opt, dict) and "material_id" in opt:
            V = material_volume_fraction(self.x_phys, self.z, self.z_real,
                                         int(opt["material_id"]), float(opt.get("tau", 1.0)))
            return V / limit - 1.0
        if qoi.name == "VOLUME_FRACTION":
            return torch.mean(self.x_phys) / limit - 1.0
        if qoi.name == "MASS":
            return self._mass(qoi) / limit - 1.0
        if qoi.name == "COST":
            cost = torch.sum(self._prop("Density", qoi) * self._prop("Cost", qoi) * self.x_phys) * self.elem_volume
            return cost / limit - 1.0
        if qoi.name == "COMPLIANCE":
            return self._compliance() / limit - 1.0
        if qoi.name == "STRESS_FAILURE_FACTOR":
            return self._ff_pnorm(qoi) / limit - 1.0
        raise NotImplementedError(f"constraint {qoi.name} {opt} not in the AD path yet")

    def grad_phys(self, q):
        """dq/d(x_phys, z) as one numpy vector [x (N) ; z latent-major (N*latentDim)]."""
        g_x, g_z = torch.autograd.grad(q, [self.x_phys, self.z], retain_graph=True, allow_unused=True)
        g_x = np.zeros(self.N) if g_x is None else g_x.detach().numpy()
        g_z = np.zeros((self.N, self.latentDim)) if g_z is None else g_z.detach().numpy()
        return np.concatenate([g_x, g_z.T.reshape(-1)])

    def chain_to_x(self, g):
        """Replace the x_phys block of g by the gradient w.r.t. the raw x (Heaviside + filter)."""
        g = np.array(g, dtype=float)
        g_xphys = torch.as_tensor(g[:self.N], dtype=_DTYPE)
        (g_x,) = torch.autograd.grad(self.x_phys, self.x, grad_outputs=g_xphys, retain_graph=True)
        g[:self.N] = g_x.detach().numpy()
        return g

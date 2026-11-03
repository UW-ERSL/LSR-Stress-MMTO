"""
design_model.py
===============
The optimization callback's differentiable model, one torch tree per MMA iteration (MMTOModelAD).
It knows no physics: the physics' analyse (its front page) does the solve between the decoded
properties and the quantities. The tree below is the structural one.

TREE (arrow = "depends on")
    J      -> svm -> s -> x_phys, E, nu, u   the objective (to_params.Objective, a quantity's name)
    g_k    -> x_phys, z                      the constraints (to_params.Constraints), each g_k <= 0
                                             (every quantity: docs/project_file_reference.md)
    Pz     -> x_phys, z                      latent smoothing penalty
    Dp     -> x_phys (frozen), z             distance to the nearest real material
    E, nu, Y, rho -> z                       VAE decoder (whichever attributes the database has)
    u      -> K -> x_phys, E, nu             K_e = SIMP(x_phys) * (lam_e K_lam + mu_e K_mu)
    x_phys -> x_filt -> x                    Heaviside(density filter(x))
Each objective and constraint is one entry of QUANTITIES (formulation/quantities.py).
Any other name raises KeyError rather than run something different.
The driver (optimize/optimization_loop.py) combines them: OBJ = J/J0 + w*Pz + gamma*Dp/Sd0.

PyTO-AD sets torch's default dtype to float64 on import; everything here is float64.
"""
import numpy as np
import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path (and its float64 default)

from lsr_mmto.design.density_filter_and_projection import SparseOperator, density_filter, heaviside
from lsr_mmto.design.latent_to_properties import decode
from lsr_mmto.formulation.quantities import find_quantity

_DTYPE = torch.float64


class MMTOModelAD:
    """The optimization callback's differentiable model, one torch tree per MMA iteration.

        x (raw MMA density) -> density filter -> Heaviside -> x_phys
        z (latents)         -> VAE decoder    -> props (every property the database has)
        x_phys, props       -> physics.analyse (structural: E, nu -> solve -> u -> von Mises)
        -> objective J, constraints, and the two penalties Pz (latent smoothing), Dp (distance)

    grad_phys(q) returns dq/d(x_phys, z) in the driver's layout [x (N) ; z latent-major (N*latentDim)],
    the same layout the manual sensitivities returned, so the driver can normalize, override
    and add terms exactly as before. chain_to_x(g) carries the x block back through the
    Heaviside projection and the density filter to the raw MMA variable.
    Every derivative comes from autograd; nothing is derived by hand.
    """

    def __init__(self, fe, physics, enc, H_density, Hs_density, H_material, Hs_material, z_real,
                 to_params, settings, elem_volume):
        self.fe, self.physics, self.enc, self.to_params, self.settings = fe, physics, enc, to_params, settings
        self.N = fe.mesh.num_elems
        self.latentDim = enc.vae_params.latentDim
        self.H = SparseOperator(H_density)
        self.Hs = torch.as_tensor(np.asarray(Hs_density, dtype=float).ravel(), dtype=_DTYPE)
        self.Hm = SparseOperator(H_material)
        self.Hsm = torch.as_tensor(np.asarray(Hs_material, dtype=float).ravel(), dtype=_DTYPE)
        self.z_real = torch.as_tensor(np.asarray(z_real, dtype=float), dtype=_DTYPE)
        self.elem_volume = float(elem_volume)

    def evaluate(self, zeta, beta, eta, use_heaviside=True, apply_density_filter=True):
        """Build the tree at design zeta = [x ; z latent-major]; returns J and the constraint list."""
        zeta = np.asarray(zeta, dtype=float).ravel()
        N, latentDim = self.N, self.latentDim
        self.x = torch.tensor(zeta[:N], dtype=_DTYPE, requires_grad=True)
        self.z = torch.tensor(zeta[N:].reshape(latentDim, N).T.copy(), dtype=_DTYPE, requires_grad=True)

        x_filt = density_filter(self.x, self.H, self.Hs) if apply_density_filter else self.x
        self.x_phys = heaviside(x_filt, beta, eta) if use_heaviside else x_filt

        self.props = decode(self.enc, self.z)
        self.physics.analyse(self)              # the solve, and what the physics' quantities read

        self.J = self._objective()
        self.cons = [self._constraint(name, opt, limit) for name, opt, limit in self.to_params.Constraints]
        return self.J, self.cons

    def physical_density(self, x, beta, eta, use_heaviside=True, apply_density_filter=True):
        """x_phys of a raw density x, numpy in and out: the filter and projection evaluate uses."""
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(x, dtype=float).ravel(), dtype=_DTYPE)
            x_filt = density_filter(x, self.H, self.Hs) if apply_density_filter else x
            return (heaviside(x_filt, beta, eta) if use_heaviside else x_filt).numpy().copy()

    def _value(self, q, opt):
        """q.fn at the current state, after checking the database has the columns it reads."""
        missing = [c for c in q.needs if c not in self.props]
        if missing:
            raise KeyError(f"{q.name} needs a {missing[0]} column in the material database; "
                           f"this one has {', '.join(sorted(self.props))}")
        return q.fn(self, opt)

    def _objective(self):
        name, opt = self.to_params.Objective[0], self.to_params.Objective[1]
        return self._value(find_quantity(name), opt)

    def _constraint(self, name, opt, limit):
        return self._value(find_quantity(name), opt) / limit - 1.0

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

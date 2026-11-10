"""
design_model.py
===============
The optimization callback's differentiable model, one torch tree per MMA iteration (MMTOModelAD).
It knows no physics: the physics' analyse (its front page) does the solve between the decoded
properties and the quantities. The tree below is the structural one.

TREE (arrow = "depends on")
    J      -> svm -> s -> x_phys, E, nu, u   the objective (to_params.Objective, a quantity's name)
    g_k    -> x_phys, z                      the constraints (to_params.Constraints), each g_k <= 0
                                             (every quantity: docs/reference.md)
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
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path (and its float64 default)

from lsr_mmto.design.density_filter_and_projection import SparseOperator, density_filter, heaviside
from lsr_mmto.design.latent_to_properties import decode
from lsr_mmto.formulation.quantities import find_quantity

if TYPE_CHECKING:
    from types import ModuleType

    from scipy.sparse import spmatrix

    from lsr_mmto.formulation.quantities import Quantity
    from lsr_mmto.materials.encoder import MaterialEncoder
    from lsr_mmto.optimize.mmto_to_params import MMTOParams
    from lsr_mmto.problem.settings import RunSettings

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

    Built once per run (optimization_loop.setup); each evaluate() replaces x, z, x_phys, props, J and cons,
    and the physics' analyse adds its own attributes, the ones its quantities read (structural: u, svm;
    template: c). grad_phys and chain_to_x differentiate the tree of the last evaluate().
    """

    fe: Any                                  # the physics' solver, whatever its make_solver returns
    physics: "ModuleType"                    # the physics' front page (physics_module)
    enc: "MaterialEncoder"
    to_params: "MMTOParams"
    settings: "RunSettings"
    N: int                                   # number of elements
    latentDim: int
    H: SparseOperator                        # density filter
    Hs: torch.Tensor                         # its row sums, (N,)
    Hm: SparseOperator                       # material filter
    Hsm: torch.Tensor                        # its row sums, (N,)
    z_real: torch.Tensor                     # the database's materials in latent space, (materials, latentDim)
    elem_volume: float
    # set by evaluate()
    x: torch.Tensor                          # raw density, (N,), a leaf; chain_to_x differentiates to it
    z: torch.Tensor                          # latents, element-major (N, latentDim), a leaf
    x_phys: torch.Tensor                     # filtered and projected density, (N,)
    props: dict[str, torch.Tensor]           # every decoded property, per element, by database column
    J: torch.Tensor                          # objective, a scalar
    cons: list[torch.Tensor]                 # constraints g_k = value / limit - 1, scalars

    def __init__(self, fe: Any, physics: "ModuleType", enc: "MaterialEncoder", H_density: "spmatrix",
                 Hs_density: np.ndarray, H_material: "spmatrix", Hs_material: np.ndarray, z_real: np.ndarray,
                 to_params: "MMTOParams", settings: "RunSettings", elem_volume: float) -> None:
        self.fe, self.physics, self.enc, self.to_params, self.settings = fe, physics, enc, to_params, settings
        self.N = fe.mesh.num_elems
        self.latentDim = enc.vae_params.latentDim
        self.H = SparseOperator(H_density)
        self.Hs = torch.as_tensor(np.asarray(Hs_density, dtype=float).ravel(), dtype=_DTYPE)
        self.Hm = SparseOperator(H_material)
        self.Hsm = torch.as_tensor(np.asarray(Hs_material, dtype=float).ravel(), dtype=_DTYPE)
        self.z_real = torch.as_tensor(np.asarray(z_real, dtype=float), dtype=_DTYPE)
        self.elem_volume = float(elem_volume)

    def evaluate(self, zeta: np.ndarray, beta: float, eta: float, use_heaviside: bool = True,
                 apply_density_filter: bool = True) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Build the tree at design zeta = [x ; z latent-major], N + N*latentDim numbers in any shape; returns J
        and the constraint list. Sets x, z, x_phys, props, J, cons, and runs the physics' analyse."""
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

    def physical_density(self, x: np.ndarray, beta: float, eta: float, use_heaviside: bool = True,
                         apply_density_filter: bool = True) -> np.ndarray:
        """x_phys of a raw density x, numpy in and out, (N,): the filter and projection evaluate uses, outside
        the tree; nothing on the model changes."""
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(x, dtype=float).ravel(), dtype=_DTYPE)
            x_filt = density_filter(x, self.H, self.Hs) if apply_density_filter else x
            return (heaviside(x_filt, beta, eta) if use_heaviside else x_filt).numpy().copy()

    def _value(self, q: "Quantity", opt: dict[str, int | float | str] | None) -> torch.Tensor:
        """q.fn at the current state, after checking the database has the columns it reads."""
        missing = [c for c in q.needs if c not in self.props]
        if missing:
            raise KeyError(f"{q.name} needs a {missing[0]} column in the material database; "
                           f"this one has {', '.join(sorted(self.props))}")
        return q.fn(self, opt)

    def _objective(self) -> torch.Tensor:
        """The objective (to_params.Objective) at the current state, a torch scalar."""
        name, opt = self.to_params.Objective[0], self.to_params.Objective[1]
        return self._value(find_quantity(name), opt)

    def _constraint(self, name: str, opt: dict[str, int | float | str] | None, limit: float) -> torch.Tensor:
        """A constraint at the current state as value / limit - 1, a torch scalar; met when <= 0."""
        return self._value(find_quantity(name), opt) / limit - 1.0

    def grad_phys(self, q: torch.Tensor) -> np.ndarray:
        """dq/d(x_phys, z) as one numpy vector [x (N) ; z latent-major (N*latentDim)], for a scalar q of the last
        evaluate()'s tree; a part q does not depend on is zeros. The tree is kept for further gradients."""
        g_x, g_z = torch.autograd.grad(q, [self.x_phys, self.z], retain_graph=True, allow_unused=True)
        g_x = np.zeros(self.N) if g_x is None else g_x.detach().numpy()
        g_z = np.zeros((self.N, self.latentDim)) if g_z is None else g_z.detach().numpy()
        return np.concatenate([g_x, g_z.T.reshape(-1)])

    def chain_to_x(self, g: np.ndarray) -> np.ndarray:
        """A copy of g, a gradient [x_phys (N) ; z], with its x_phys block replaced by the gradient w.r.t. the raw x
        (Heaviside + filter), through the last evaluate()'s tree; the z block is unchanged."""
        g = np.array(g, dtype=float)
        g_xphys = torch.as_tensor(g[:self.N], dtype=_DTYPE)
        (g_x,) = torch.autograd.grad(self.x_phys, self.x, grad_outputs=g_xphys, retain_graph=True)
        g[:self.N] = g_x.detach().numpy()
        return g

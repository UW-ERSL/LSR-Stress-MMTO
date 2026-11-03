"""
latent_penalties.py
===================
The terms that steer the latent points z, added to the objective by the optimization loop:

    distance_penalty   Dp, each solid element's distance to the nearest real material
    latent_smoothing   Pz, each element's distance to the mean of its neighbours' latent points

and the latent sensitivity filter, which smooths the latent blocks of a gradient with the
material filter (switched off in the loop; kept for other physics).
"""
import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path (and its float64 default)
from lsr_mmto.design.density_filter_and_projection import sparse_matmul


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


def smooth_latent_gradient_inplace(vec, H_material, Hs_material, num_elems, latentDim):
    """Sensitivity-filter (in place) each latent block of an objective gradient.

    Applies (H_material @ g)/Hs_material to each per-latent-dim slice. This is a
    Sigmund-style regularizer on the latent gradient (z itself is not filtered).
    """
    for ell in range(latentDim):
        sl = slice(
            num_elems + ell * num_elems,
            num_elems + (ell + 1) * num_elems,
        )
        vec[sl] = (H_material @ vec[sl]) / Hs_material


def smooth_latent_constraint_gradient_inplace(mat, H_material, Hs_material, num_elems, latentDim):
    """Same latent-block sensitivity filter as above, for the constraint Jacobian."""
    nCons = mat.shape[0]
    for i in range(nCons):
        for ell in range(latentDim):
            sl = slice(
                num_elems + ell * num_elems,
                num_elems + (ell + 1) * num_elems,
            )
            mat[i, sl] = (H_material @ mat[i, sl]) / Hs_material

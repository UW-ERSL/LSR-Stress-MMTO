"""
latent_penalties.py
===================
The terms the optimization loop adds to the objective, each one entry of PENALTIES, in the order
they are added. Two every physics has are written here:

    LatentSmoothing   Pz, each element's distance to the mean of its neighbours' latent points
    DistancePenalty   Dp, each solid element's distance to the nearest real material

and a physics may give its own (PENALTIES on its front page), added after these. Also here: the
latent sensitivity filter, which smooths the latent blocks of a gradient with the material filter
(switched off in the loop; kept for other physics).
"""
import numpy as np
import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path (and its float64 default)
from lsr_mmto.design.density_filter_and_projection import sparse_matmul
from lsr_mmto.physics.available_physics import MODULES


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


# ----------------------------------------------------------------------------- penalties
# Every term added to the objective is one entry of PENALTIES. One that every physics has is written
# here; one of a physics is written in its module (PENALTIES on its front page, as the arguments below
# without physics). Each iteration the loop asks every penalty for its weight; a weight of 0 leaves it
# out of that iteration, otherwise it adds weight * value / scale to the objective, and the same times
# the value's gradient (autograd) to the objective's gradient.
#
#   name         its name, one no other penalty has
#   shown        its name in the app
#   help         what it does, in plain words
#   fn(m)        its value, a torch scalar, from the model's current state (m.x_phys, m.z, m.props, what the
#                physics' analyse left, m.settings); m.x_phys.detach() keeps the density out of its gradient
#   weight(state, it)  its weight this iteration, 0 when it is off; its schedule lives here. state is the
#                run's state (state.s its settings, state.iteration the designs evaluated so far), it the
#                iteration's numbers so far (it.Jn, the objective divided by its first value)
#   scale        what its value is divided by, stated in every entry: "raw" (nothing) or "first value" (its
#                value at the first iteration it is on, at least 1e-12)
#   key          the history key of what it adds to the objective each iteration (0 while it is off)
#   raw_key      optional: the history key of its value before weight and scale (0 while it is off)
#   physics      the physics it applies to; empty: every physics
SCALES = ("raw", "first value")


class Penalty:
    __slots__ = ("name", "shown", "help", "fn", "weight", "scale", "key", "raw_key", "physics")

    def __init__(self, name, shown, help, fn, weight, scale, key, raw_key=None, physics=()):
        if scale not in SCALES:
            raise ValueError(f"penalty {name!r}: scale must be one of {', '.join(map(repr, SCALES))}, "
                             f"not {scale!r}")
        self.name, self.shown, self.help, self.fn, self.weight = name, shown, help, fn, weight
        self.scale, self.key, self.raw_key, self.physics = scale, key, raw_key, frozenset(physics)

    def keys(self):
        """Its history keys: its raw value's (if it has one), then what it adds."""
        return ([self.raw_key] if self.raw_key else []) + [self.key]

    def applies_to(self, physics):
        """True for a penalty of that physics, or one every physics has."""
        return not self.physics or physics in self.physics


def _latent_smoothing_weight(state, it):
    """z_smoothing_weight while Latent smoothing is on; off from z_smoothing_disable_after_iter on.
    Raw scale: the penalty is zero at a spatially uniform start, so it has no first value to be
    divided by; the weight multiplies the raw roughness directly."""
    s = state.s
    if (
        state.use_z_smoothing
        and s.z_smoothing_disable_after_iter is not None
        and state.iteration >= s.z_smoothing_disable_after_iter
    ):
        state.use_z_smoothing = False
        state.during_run["z_smoothing_off_at_iter"] = int(state.iteration)
        if state.verbose:
            print(f"[INFO] z-smoothing penalty disabled at iteration {state.iteration}")
    return float(s.z_smoothing_weight) if state.use_z_smoothing else 0.0


def _distance_penalty_weight(state, it):
    """gamma, a pure ratio: the penalty's share of the objective, its value divided by its first value.
    Off from distance_penalty_disable_after_iter on and before penalty_start_iter (both only with the
    plateau start off); each iteration it is on, gamma grows by gamma_factor up to gamma_max (with the
    plateau start, once the ramp has started). Returns this iteration's gamma; the growth is for the next."""
    s = state.s
    if (
        state.use_distance_penalty
        and not s.use_dynamic_gamma_scheduling
        and s.distance_penalty_disable_after_iter is not None
        and state.iteration >= s.distance_penalty_disable_after_iter
    ):
        state.use_distance_penalty = False
        state.during_run["distance_penalty_off_at_iter"] = int(state.iteration)
        if state.verbose:
            print(f"[INFO] Distance-based penalty disabled at iteration {state.iteration}")
    if not (state.use_distance_penalty and (s.use_dynamic_gamma_scheduling or state.iteration >= int(s.penalty_start_iter))):
        return 0.0
    gamma = float(state.gamma)
    if s.use_dynamic_gamma_scheduling:
        state.objective_history.append(float(it.Jn))
        if not state.gamma_ramp_started:
            # the ramp starts once J changed by less than the plateau tolerance over the last window
            # iterations (not before the earliest start), or in time to reach gamma_max end_margin
            # iterations before the last
            _K, _rel_tol = int(s.gamma_plateau_window), float(s.gamma_plateau_tolerance)
            _min_iter = int(s.gamma_plateau_earliest_iter)
            _n_ramp = int(np.ceil(np.log(max(s.gamma_max / s.gamma_init, 1 + 1e-9))
                                  / np.log(max(s.gamma_factor, 1 + 1e-9))))
            _deadline = int(state.to_params.MaxIterations) - _n_ramp - int(s.gamma_ramp_end_margin)
            _J_hist = state.objective_history
            _settled = (state.iteration >= _min_iter and len(_J_hist) > _K and
                        abs(_J_hist[-1] - _J_hist[-1 - _K])
                        <= _rel_tol * max(abs(_J_hist[-1 - _K]), 1e-12))
            if _settled or state.iteration >= _deadline:
                state.gamma_ramp_started = True
                state.during_run["gamma_ramp_started_at_iter"] = int(state.iteration)
                if state.verbose:
                    print(f"[DYN-GAMMA] ramp started at iter {state.iteration} "
                          f"({'plateau' if _settled else 'deadline'})")
        if state.gamma_ramp_started:
            state.gamma = min(state.gamma * s.gamma_factor, s.gamma_max)
    else:
        state.gamma = min(state.gamma * s.gamma_factor, s.gamma_max)
    return gamma


PENALTIES = [
    Penalty("LatentSmoothing", "Latent smoothing",
            "Discourages neighbouring elements from flickering between materials.",
            lambda m: latent_smoothing(m.x_phys, m.z, m.Hm, m.Hsm), _latent_smoothing_weight, "raw", "Pz_n"),
    # x enters the distance penalty as a frozen weight (detached), so its x block is zero by construction
    Penalty("DistancePenalty", "Distance penalty",
            "Pushes each element's material towards one of the real materials in the database.",
            lambda m: distance_penalty(m.x_phys, m.z, m.z_real), _distance_penalty_weight, "first value", "Pn",
            raw_key="P"),
] + [Penalty(physics=[m.NAME], **p) for m in MODULES for p in m.PENALTIES]


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

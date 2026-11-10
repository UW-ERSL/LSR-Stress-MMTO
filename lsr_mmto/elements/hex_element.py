"""
hex_element.py
==============
The hex element's pieces that depend on its shape, not on the physics: the Lame parameters, the
split of an element matrix into its lam and mu parts, and the strain-displacement matrix at the
element centre.

Element stiffness and constitutive matrix are linear in the two Lame parameters:
    D(E, nu)   = lam * D_lam + mu * D_mu
    K_e(E, nu) = lam * K_lam + mu * K_mu
exact for every linear isotropic material, constant nu or not. The basis matrices are
identified once from PyTO-AD's own element routines, so the integration is identical.
"""
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    import torch


def lame_parameters(E: "float | torch.Tensor",
                    nu: "float | torch.Tensor") -> "tuple[float | torch.Tensor, float | torch.Tensor]":
    """(lam, mu) from (E, nu); works on torch tensors and keeps them in the tree."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


def lame_basis(fn: Callable[[float, float], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Basis (A_lam, A_mu) with fn(E, nu) = lam*A_lam + mu*A_mu, for fn linear in D.

    nu = 0 gives lam = 0, which isolates A_mu; a second nu then isolates A_lam.
    """
    lam0, mu0 = lame_parameters(1.0, 0.0)
    A_mu = np.asarray(fn(1.0, 0.0)) / mu0
    lam1, mu1 = lame_parameters(1.0, 0.25)
    A_lam = (np.asarray(fn(1.0, 0.25)) - mu1 * A_mu) / lam1
    return A_lam, A_mu


def structural_B_matrix(elem_size: np.ndarray | tuple[float, float, float]) -> np.ndarray:
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

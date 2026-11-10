"""
density_filter_and_projection.py
================================
The raw MMA density x to the physical density x_phys: the density filter (a constant sparse
matrix applied inside the torch tree) and the smoothed Heaviside projection.

    x_phys = heaviside(density_filter(x, H, Hs), beta, eta)

The same filter matrix type also smooths the latent points (latent_penalties.latent_smoothing).
"""
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path (and its float64 default)

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix, spmatrix


class SparseOperator:
    """A constant sparse matrix (a filter) as scipy CSR, with its transpose, both built once."""

    H: "csr_matrix"
    Ht: "csr_matrix"

    def __init__(self, H: "spmatrix") -> None:
        self.H = H.tocsr()
        self.Ht = self.H.T.tocsr()


class _SparseMatmul(torch.autograd.Function):
    """H @ x inside the tree: forward H @ x, backward H^T @ g, both scipy CSR products. Same result as
    torch.sparse.mm on a COO tensor, which is about 20 times slower for a large 3D filter."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, op: SparseOperator) -> torch.Tensor:
        ctx.op = op
        return torch.from_numpy(np.asarray(op.H @ x.detach().numpy()))

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> tuple[torch.Tensor, None]:
        return torch.from_numpy(np.asarray(ctx.op.Ht @ g.detach().numpy())), None


def sparse_matmul(op: SparseOperator, x: torch.Tensor) -> torch.Tensor:
    """op.H @ x for x of shape (N,) or (N, k), differentiable w.r.t. x."""
    return _SparseMatmul.apply(x, op)


def density_filter(x: torch.Tensor, H: SparseOperator, Hs: torch.Tensor) -> torch.Tensor:
    """(H @ x) / Hs, differentiable w.r.t. x; x and Hs of shape (N,)."""
    return sparse_matmul(H, x) / Hs


def heaviside(x_filt: torch.Tensor, beta: float, eta: float) -> torch.Tensor:
    """The smoothed Heaviside projection of x_filt, sharpness beta and threshold eta: 0 at x_filt = 0 and 1 at
    x_filt = 1, steeper around eta as beta grows."""
    tb = np.tanh(beta * eta)
    return (tb + torch.tanh(beta * (x_filt - eta))) / (tb + np.tanh(beta * (1.0 - eta)))

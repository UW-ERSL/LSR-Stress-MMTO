"""
density_filter_and_projection.py
================================
The raw MMA density x to the physical density x_phys: the density filter (a constant sparse
matrix applied inside the torch tree) and the smoothed Heaviside projection.

    x_phys = heaviside(density_filter(x, H, Hs), beta, eta)

The same filter matrix type also smooths the latent points (latent_penalties.latent_smoothing).
"""
import numpy as np
import torch

import lsr_mmto.pyto_ad_imports  # noqa: F401  puts PyTO-AD on sys.path (and its float64 default)


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

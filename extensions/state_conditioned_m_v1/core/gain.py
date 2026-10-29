"""Closed-form information/Kalman-style gain and contraction checks."""
from __future__ import annotations
import torch


def information_gain(M: torch.Tensor, H: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Return L = M^{-1} H^T (H M^{-1} H^T + R)^-1.

    M has shape [..., n, n], H [..., m, n], R [..., m, m].
    The result has shape [..., n, m].  This is a gain *construction rule*;
    it is not by itself a proof that the closed-loop error converges.
    """
    if M.shape[-1] != M.shape[-2] or H.shape[-1] != M.shape[-1]:
        raise ValueError("Dimension mismatch in M/H")
    if R.shape[-2:] != (H.shape[-2], H.shape[-2]):
        raise ValueError("R must be m x m")
    M_inv_Ht = torch.linalg.solve(M, H.transpose(-1, -2))
    S = H @ M_inv_Ht + R
    return torch.linalg.solve(S, M_inv_Ht.transpose(-1, -2)).transpose(-1, -2)


def symmetric(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))


def contraction_gram(A: torch.Tensor, H: torch.Tensor, M: torch.Tensor,
                     L: torch.Tensor, alpha: float) -> torch.Tensor:
    """Return (A-LH)^T M + M(A-LH) + 2 alpha M.

    Negative semidefinite (up to numerical tolerance) is the frozen-metric
    linearized contraction condition.
    """
    closed = A - L @ H
    return symmetric(closed.transpose(-1, -2) @ M + M @ closed + 2.0 * alpha * M)


def contraction_margin(A: torch.Tensor, H: torch.Tensor, M: torch.Tensor,
                       L: torch.Tensor, alpha: float) -> torch.Tensor:
    """Positive margin means the frozen-metric contraction certificate holds."""
    gram = contraction_gram(A, H, M, L, alpha)
    return -torch.linalg.eigvalsh(gram).amax(dim=-1)


def trace_eigen_bounds(M: torch.Tensor):
    """Small diagnostics helper for finite SPD matrices."""
    if M.shape[-1] != M.shape[-2]:
        raise ValueError("M must be square")
    sym = symmetric(M)
    eig = torch.linalg.eigvalsh(sym)
    return sym.diagonal(dim1=-2, dim2=-1).sum(-1), eig[..., 0], eig[..., -1]

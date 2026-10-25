"""Forward-only profile of a frozen denoiser along one Brownian path (euler_dgc_derivation.pdf, Section 6).

For ascending variance edges t_0 < ... < t_K, one path per clean sequence:

    X_{t_0} = X0 + sqrt(t_0) Z_0,   X_{t_{k+1}} = X_{t_k} + sqrt(t_{k+1} - t_k) Z_{k+1}   (independent Z)

and, with dD = D(X_{t_{k+1}}) - D(X_{t_k}), dX = X_{t_{k+1}} - X_{t_k}, dt = t_{k+1} - t_k, per sequence and interval
(averaged over the generated coordinates):

    A = mean((dD - 1.5 dX)^2) / dt      B = mean((dD - dX)^2) / dt
    q = mean(dD^2) / dt                 T = mean(dD * dX) / dt          V = mean(dX^2) / dt

plus the clean-target MSE mean((D - X0)^2) at every edge. For a Bayes denoiser A, B -> E||J - c I||^2 / d as the bins
shrink (c = 3/2, 1), and E[T] = h(t_{k+1}) / (t_{k+1} d), E[V] = 1. K + 1 denoiser evaluations per sequence.
Differences and squares are taken in float64; the denoiser output is cast before subtracting.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch

STATS = ("A", "B", "q", "T", "V")


@torch.inference_mode()
def profile_batch(denoise: Callable, x0: torch.Tensor, t_edges: torch.Tensor, *,
                  gen_mask: Optional[torch.Tensor] = None, generator: Optional[torch.Generator] = None) -> dict:
    """Per-sequence interval statistics [B, K] and edge MSE [B, K+1] for one batch of clean sequences.

    denoise(x, sigma) -> D in data coordinates, same shape as x (sigma a python float). gen_mask (bool [B, d]) marks
    generated coordinates: only they are noised, and statistics average over them; other coordinates stay clean.
    """
    x0 = x0.to(torch.float32)
    Bsz, d = x0.shape
    mask = torch.ones_like(x0, dtype=torch.bool) if gen_mask is None else gen_mask.to(torch.bool)
    m = mask.to(torch.float64)
    n_gen = m.sum(dim=1).clamp_min(1.0)
    cmean = lambda v: (v * m).sum(dim=1) / n_gen              # per-sequence mean over generated coordinates
    t = t_edges.to(torch.float64).cpu()
    K = len(t) - 1
    out = {k: torch.empty(Bsz, K, dtype=torch.float64) for k in STATS}
    out["mse"] = torch.empty(Bsz, K + 1, dtype=torch.float64)

    def noise(scale):
        z = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=torch.float32)
        return z * float(scale) * mask

    x = x0 + noise(t[0].sqrt())
    d_prev = denoise(x, float(t[0].sqrt())).double()
    out["mse"][:, 0] = cmean((d_prev - x0.double()) ** 2).cpu()
    for k in range(K):
        dt = float(t[k + 1] - t[k])
        x_next = x + noise(dt ** 0.5)
        d_next = denoise(x_next, float(t[k + 1].sqrt())).double()
        dx = x_next.double() - x.double()
        dd = d_next - d_prev
        out["A"][:, k] = (cmean((dd - 1.5 * dx) ** 2) / dt).cpu()
        out["B"][:, k] = (cmean((dd - dx) ** 2) / dt).cpu()
        out["q"][:, k] = (cmean(dd ** 2) / dt).cpu()
        out["T"][:, k] = (cmean(dd * dx) / dt).cpu()
        out["V"][:, k] = (cmean(dx ** 2) / dt).cpu()
        out["mse"][:, k + 1] = cmean((d_next - x0.double()) ** 2).cpu()
        x, d_prev = x_next, d_next
    return out


def bootstrap_ci(per_seq: torch.Tensor, n_boot: int = 1000, seed: int = 0, level: float = 0.95):
    """Percentile CI of the across-sequence mean, resampling whole sequences (all bins together). [M, K] -> 2 x [K]."""
    g = torch.Generator().manual_seed(int(seed))
    M = per_seq.shape[0]
    idx = torch.randint(0, M, (n_boot, M), generator=g)
    means = per_seq[idx].mean(dim=1)
    a = (1.0 - level) / 2
    return torch.quantile(means, a, dim=0), torch.quantile(means, 1.0 - a, dim=0)

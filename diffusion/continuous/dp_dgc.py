"""Finite-NFE DGC step allocation by dynamic programming (the "dp_dgc" sigma grid).

Follows DGC_five_delta_schedulers.ipynb (Wainwright 2026, Theorems 1-2): K fixed blocks b_0 < ... < b_K uniform in
log-variance r = log sigma^2 between the grid's endpoints, block quantities

    S_k = log(b_{k+1}^2 / b_k^2),      G_k = G(b_k^2, b_{k+1}^2) = 1/2 * int_block h'(v) dr,

and the exact integer program

    min_{n_k}  sum_k G_k * (exp(S_k / n_k) - 1)    s.t.  sum_k n_k = M,  n_k >= 1,

solved by DP; block k is then filled with n_k geometric (log-uniform) steps. M = N - 1 is the interval count of an
N-point grid (the sampler convention: N sigmas, N - 1 transitions).

h' comes from the existing DGC table, not a second definition: the table's density in u = log sigma is
p(u) ∝ sqrt(h'(v)) (scripts/dgc/build_dgc_tables.py; the notebook's continuum grid, uniform in int sqrt(h') dr), so
h'(v) ∝ p(u)^2 and G_k ∝ int_block p(u)^2 du. A global scale on G does not change the argmin.

weight="length_squared" is the earlier variant (G_k = m_k^2 / S_k with m_k the table mass in the block), kept so
its results stay reproducible: its large-N optimum is exactly the continuous DGC block allocation.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

WEIGHTS = ("increment", "length_squared")


def dp_objective(n: Sequence[int], S: Sequence[float], G: Sequence[float]) -> float:
    n, S, G = np.asarray(n, float), np.asarray(S, float), np.asarray(G, float)
    return float(np.sum(G * np.expm1(S / n)))


def dp_allocate(S: Sequence[float], G: Sequence[float], M: int) -> list:
    """Exact integer minimiser of sum_k G_k (exp(S_k/n_k) - 1), sum n_k = M, n_k >= 1. O(K M^2)."""
    S, G = np.asarray(S, float), np.asarray(G, float)
    K, M = len(S), int(M)
    if K < 1:
        raise ValueError("dp_allocate needs at least one block")
    if M < K:
        raise ValueError(f"dp_allocate: {M} steps cannot give each of {K} blocks at least one (need M >= K)")
    if np.any(S <= 0) or np.any(G < 0) or not np.all(np.isfinite(S)) or not np.all(np.isfinite(G)):
        raise ValueError("dp_allocate needs finite S_k > 0 and G_k >= 0")
    n_all = np.arange(1, M + 1)
    # best[k][m]: min cost of blocks k..K-1 using exactly m steps; choice[k][m]: n_k attaining it
    best = np.full((K + 1, M + 1), np.inf)
    best[K, 0] = 0.0
    choice = np.zeros((K, M + 1), dtype=np.int64)
    for k in range(K - 1, -1, -1):
        cost = G[k] * np.expm1(S[k] / n_all)                  # cost[j] = f_k(j + 1)
        for m in range(K - k, M - k + 1):                       # blocks k..K-1 need >= K - k steps
            n = n_all[: m - (K - k - 1)]                         # leave >= 1 step per later block
            tot = cost[: len(n)] + best[k + 1, m - n]
            j = int(np.argmin(tot))
            best[k, m], choice[k, m] = tot[j], n[j]
    out, m = [], M
    for k in range(K):
        out.append(int(choice[k, m]))
        m -= out[-1]
    assert m == 0 and sum(out) == M and min(out) >= 1
    return out


class DGCIncrement:
    """Cumulative DGC increment G(sigma_lo^2, sigma^2) ∝ int p(u)^2 du on [sigma_lo, sigma_hi], from a DGC table.

    The table's per-bin density (pdf at bin-centre sigmas, equal-width bins in log sigma) is interpolated linearly in
    u = log sigma and squared; the integral is a trapezoid on a dense grid, as in the notebook's DGCProfile.
    """

    def __init__(self, sig, pdf, sigma_lo, sigma_hi, grid_points: int = 4097):
        self.u = np.linspace(math.log(sigma_lo), math.log(sigma_hi), int(grid_points))
        p = np.interp(self.u, np.log(np.asarray(sig, float)), np.asarray(pdf, float), left=0.0, right=0.0)
        q = np.maximum(p, 0.0) ** 2
        self.cum = np.r_[0.0, np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(self.u))]

    def __call__(self, s_lo: float, s_hi: float) -> float:
        """G between two standard deviations (any scale; the DP argmin is scale free)."""
        return float(np.interp(math.log(s_hi), self.u, self.cum) - np.interp(math.log(s_lo), self.u, self.cum))

    def finite_step_bound(self, sigmas) -> float:
        """Theorem 1's bound on any decreasing grid: sum over intervals of ((hi/lo)^2 - 1) * G(lo^2, hi^2)."""
        s = np.asarray(sigmas, float)
        if len(s) < 2 or np.any(s[1:] >= s[:-1]) or np.any(s <= 0):
            raise ValueError("finite_step_bound needs strictly decreasing positive sigmas")
        return float(sum(((hi / lo) ** 2 - 1) * self(lo, hi) for hi, lo in zip(s[:-1], s[1:])))


def block_quantities(sig, cdf, sigma_lo: float, sigma_hi: float, K: int, pdf=None, weight: str = "increment"):
    """S_k, G_k, the r-edges of K blocks uniform in r = log sigma^2, and m_k (the continuous grid's step share).

    sig/cdf/pdf are the DGC table (ascending sigma). m_k interpolates the CDF linearly in sigma, as the continuous
    grid's inverse-CDF sampler does.
    """
    if weight not in WEIGHTS:
        raise ValueError(f"weight must be one of {WEIGHTS}, got {weight!r}")
    r_edges = np.linspace(2 * math.log(sigma_lo), 2 * math.log(sigma_hi), K + 1)
    F = np.interp(np.exp(r_edges / 2), sig, cdf)
    total = F[-1] - F[0]
    m = np.diff(F) / total if total > 0 else np.full(K, 1.0 / K)
    S = np.diff(r_edges)
    if weight == "length_squared":
        G = m ** 2 / S
    else:
        if pdf is None:
            raise ValueError("weight='increment' needs the table pdf")
        inc = DGCIncrement(sig, pdf, sigma_lo, sigma_hi)
        b = np.exp(r_edges / 2)
        G = np.array([inc(b[k], b[k + 1]) for k in range(K)])
    return S, G, r_edges, m


def dp_dgc_sigmas(sig, cdf, N: int, sigma_lo: float, sigma_hi: float, K: int = 8, pdf=None,
                  weight: str = "increment"):
    """N sigmas, descending from sigma_hi to sigma_lo, piecewise geometric with DP block allocation.

    Returns (sigmas, allocation high-sigma block first). Exactly N points and N - 1 intervals, like the
    continuous grid it replaces.
    """
    N, K = int(N), int(K)
    if N < 2:
        raise ValueError(f"dp_dgc needs at least 2 sigma points, got N={N}")
    if N - 1 < K:
        raise ValueError(f"dp_dgc: N={N} gives {N - 1} intervals, fewer than K={K} blocks; lower K or raise NFE")
    S, G, r_edges, _ = block_quantities(sig, cdf, sigma_lo, sigma_hi, K, pdf=pdf, weight=weight)
    alloc_lo_first = dp_allocate(S, G, N - 1)
    pts = [np.array([sigma_lo])]
    for k in range(K):                                       # ascending r, then reversed
        r = np.linspace(r_edges[k], r_edges[k + 1], alloc_lo_first[k] + 1)[1:]
        pts.append(np.exp(r / 2))
    sigmas = np.concatenate(pts)[::-1].copy()
    sigmas[0], sigmas[-1] = sigma_hi, sigma_lo
    return sigmas, alloc_lo_first[::-1]

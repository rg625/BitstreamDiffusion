"""Gamma-dependent Euler grids and edge-based grid construction (docs: euler_dgc_derivation.pdf, 23 Oct 2026).

For a profile on K log-sigma bins (u = log sigma, edges u_0 < ... < u_K, widths w_k) with direct-residual statistics
A_k = E||J - 3/2 I||^2 and B_k = E||J - I||^2 (per coordinate, scripts/dgc/profile_forward_increments.py), N actual
intervals and realized per-interval churn gamma, S = N * gamma and

    p_k = sqrt(A_k / (S^2 B_k + eta)),     sum_k w_k p_k = 1,

with eta found by bisection on z = eta + S^2 min(B) > 0 (eta may be negative; every denominator stays positive).
The grid takes the N + 1 quantiles of the piecewise-constant density, inverting the CDF with cumulative masses paired
with bin EDGES and interpolating in log sigma, so the requested endpoints are kept exactly. The same inversion builds
the other arms (log-uniform, corrected entropy-rate table, ordinary DGC sqrt(q_inc)). This minimises a leading-order
Euler discretisation surrogate; it is not a proved optimum for the finite-step EDM sampler.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


def _check_edges(edges_u) -> np.ndarray:
    e = np.asarray(edges_u, np.float64)
    if e.ndim != 1 or len(e) < 2 or not np.all(np.isfinite(e)) or np.any(np.diff(e) <= 0):
        raise ValueError("log-sigma edges must be a finite, strictly increasing 1-D array")
    return e


def solve_gamma_density(A, B, edges_u, S: float, tol: float = 1e-10, max_iter: int = 400):
    """Density p_k (per unit log sigma) minimising the gamma-dependent Euler surrogate, and its multiplier eta."""
    e = _check_edges(edges_u)
    w = np.diff(e)
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
    if A.shape != w.shape or B.shape != w.shape:
        raise ValueError(f"A, B need {len(w)} bins, got {A.shape}, {B.shape}")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))) or np.any(A <= 0) or np.any(B < 0):
        raise ValueError("need finite A > 0 and B >= 0 (no silent clipping of an invalid profile)")
    if not math.isfinite(S) or S < 0:
        raise ValueError(f"S = N*gamma must be finite and >= 0, got {S}")
    offset = S * S * (B - B.min())
    mass = lambda z: float(np.sum(w * np.sqrt(A / (offset + z))))
    lo, hi = 0.0, 1.0
    while mass(hi) > 1.0:
        hi *= 2.0
        if hi > 1e300:
            raise RuntimeError("could not bracket the normalisation multiplier")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if mass(mid) > 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, hi):
            break
    z = 0.5 * (lo + hi)
    p = np.sqrt(A / (offset + z))
    p /= np.sum(w * p)                       # absorb the residual bisection error into the normalisation
    return p, z - S * S * B.min()


def grid_from_density(edges_u, p, N: int) -> np.ndarray:
    """N + 1 descending sigmas: quantiles of a piecewise-constant density in log sigma, exact endpoints."""
    e = _check_edges(edges_u)
    p = np.asarray(p, np.float64)
    if p.shape != (len(e) - 1,) or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("density must be finite, non-negative, one value per bin")
    N = int(N)
    if N < 1:
        raise ValueError(f"need at least one interval, got N={N}")
    m = np.diff(e) * p
    if m.sum() <= 0:
        raise ValueError("density has no mass")
    F = np.r_[0.0, np.cumsum(m)] / m.sum()   # cumulative mass at each EDGE
    q = np.linspace(1.0, 0.0, N + 1)         # descending sigma
    # inverse CDF over the bins with mass, linear in log sigma inside each; zero-mass bins get no points
    pos = np.nonzero(m > 0)[0]
    Fl, Fr, el, er = F[pos], F[pos + 1], e[pos], e[pos + 1]
    k = np.clip(np.searchsorted(Fr, q, side="left"), 0, len(pos) - 1)
    t = np.clip((q - Fl[k]) / (Fr[k] - Fl[k]), 0.0, 1.0)
    u = el[k] + t * (er[k] - el[k])
    u[0], u[-1] = e[-1], e[0]
    s = np.exp(u)
    if not np.all(np.diff(s) < 0):
        raise ValueError("grid is not strictly descending (a quantile fell inside a zero-mass gap)")
    return s


def restrict_density(edges_u, p, u_lo: float, u_hi: float):
    """Truncate a binned density to [u_lo, u_hi]: split the boundary bins, renormalise on the band."""
    e = _check_edges(edges_u)
    p = np.asarray(p, np.float64)
    if not (e[0] - 1e-12 <= u_lo < u_hi <= e[-1] + 1e-12):
        raise ValueError(f"band [{u_lo}, {u_hi}] outside the profile [{e[0]}, {e[-1]}]")
    inner = e[(e > u_lo) & (e < u_hi)]
    ne = np.r_[u_lo, inner, u_hi]
    mids = 0.5 * (ne[:-1] + ne[1:])
    k = np.clip(np.searchsorted(e, mids, side="right") - 1, 0, len(p) - 1)
    np_ = p[k]
    tot = np.sum(np.diff(ne) * np_)
    if tot <= 0:
        raise ValueError("no density mass inside the band")
    return ne, np_ / tot


def log_uniform_grid(sigma_lo: float, sigma_hi: float, N: int) -> np.ndarray:
    return np.exp(np.linspace(math.log(sigma_hi), math.log(sigma_lo), int(N) + 1))


def karras_grid(sigma_lo: float, sigma_hi: float, N: int, rho: float = 7.0) -> np.ndarray:
    """EDM (Karras et al.) grid, as SigmaSchedule._karras_schedule but in float64 with exact endpoints."""
    t = np.linspace(0.0, 1.0, int(N) + 1)
    s = (sigma_hi ** (1 / rho) + t * (sigma_lo ** (1 / rho) - sigma_hi ** (1 / rho))) ** rho
    s[0], s[-1] = sigma_hi, sigma_lo
    return s


def table_density(edges_sigma, pdf) -> tuple:
    """(log-sigma edges, density per unit log sigma) of an entropy-rate table (pdf = mass per bin)."""
    e = np.log(np.asarray(edges_sigma, np.float64))
    pdf = np.asarray(pdf, np.float64)
    return e, pdf / np.diff(e) / pdf.sum()


def gamma_euler_grid(profile: dict, N: int, gamma: float, sigma_lo: Optional[float] = None,
                     sigma_hi: Optional[float] = None, eps: float = 0.0):
    """The gamma-dependent grid for N actual intervals and realized churn gamma; returns (sigmas, info).

    eps > 0 mixes in a uniform density, p_eps = (1 - eps) p + eps / L: a separately labelled regularised arm.
    """
    e = _check_edges(profile["log_sigma_edges"])
    A, B = np.asarray(profile["A"], np.float64), np.asarray(profile["B"], np.float64)
    S = float(N) * float(gamma)
    if sigma_lo is not None or sigma_hi is not None:
        lo = math.log(sigma_lo) if sigma_lo is not None else e[0]
        hi = math.log(sigma_hi) if sigma_hi is not None else e[-1]
        e2, _ = restrict_density(e, np.ones_like(A), lo, hi)
        k = np.clip(np.searchsorted(e, 0.5 * (e2[:-1] + e2[1:]), side="right") - 1, 0, len(A) - 1)
        e, A, B = e2, A[k], B[k]
    p, eta = solve_gamma_density(A, B, e, S)
    if eps > 0:
        p = (1.0 - eps) * p + eps / (e[-1] - e[0])
    s = grid_from_density(e, p, N)
    delta = np.log(s[:-1] / s[1:])
    lam = gamma / (-np.expm1(-delta)) if gamma > 0 else np.zeros_like(delta)
    info = dict(N=int(N), gamma=float(gamma), S=S, eta=float(eta), eps=float(eps), max_log_step=float(delta.max()),
                lambda_min=float(lam.min()), lambda_max=float(lam.max()), max_lambda_delta=float((lam * delta).max()))
    return s, info


def ordinary_dgc_grid(profile: dict, N: int, key: str = "q_increment") -> np.ndarray:
    """Ordinary DGC from the forward profile: density ∝ sqrt(q_inc) (zero-q bins carry no mass, declared)."""
    q = np.asarray(profile[key], np.float64)
    if np.any(q < 0) or not np.all(np.isfinite(q)):
        raise ValueError(f"{key} must be finite and non-negative")
    return grid_from_density(profile["log_sigma_edges"], np.sqrt(q), N)


def finite_check_grid(s: Sequence[float], N: int, sigma_lo: float, sigma_hi: float, rtol: float = 1e-9):
    s = np.asarray(s, np.float64)
    assert len(s) == N + 1, (len(s), N)
    assert np.all(np.diff(s) < 0), "not strictly descending"
    assert abs(s[0] - sigma_hi) <= rtol * sigma_hi and abs(s[-1] - sigma_lo) <= rtol * sigma_lo, "endpoint drift"

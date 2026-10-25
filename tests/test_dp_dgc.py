"""Finite-NFE DP-DGC sigma grid (diffusion/continuous/dp_dgc.py, schedule "dp_dgc").

* the DP returns integer allocations, >= 1 per block, summing exactly to the interval budget, and is optimal
  against brute-force enumeration on small cases;
* on the real DGC tables it works for K = 4/8/12 and refuses budgets below K; its large-N allocation tends to
  n_k ∝ sqrt(S_k G_k) (exp(S/n) - 1 -> S/n), which for weight="length_squared" is the continuous DGC allocation;
* the notebook's own checks (DGC_five_delta_schedulers.ipynb): its brute-force case, the Theorem-1 bound of the DP
  grid equal to the DP objective, and its bad-input cases;
* the "dp_dgc" schedule has the same length, endpoints, direction and NFE accounting as the continuous grid.
"""
import itertools
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from diffusion.continuous.dp_dgc import (DGCIncrement, block_quantities, dp_allocate, dp_dgc_sigmas,
                                        dp_objective)
from diffusion.continuous.samplers import SigmaSchedule
from evaluation.nfe import compute_nfe, steps_for_target_nfe

TABLE = "assets/entropy_tables/owt_medium_dgc"   # the CoBit-M OWT DGC table


def _table(d=TABLE):
    ld = lambda f: torch.load(f"{d}/{f}", map_location="cpu", weights_only=True).double().numpy()
    cdf = ld("entropy_cdf.pt").copy()
    cdf[-1] = 1.0
    return ld("entropy_sigmas.pt"), cdf, ld("entropy_pdf.pt")


def _compositions(M, K):
    for cuts in itertools.combinations(range(1, M), K - 1):
        b = (0,) + cuts + (M,)
        yield [b[i + 1] - b[i] for i in range(K)]


@pytest.mark.parametrize("K,M,seed", [(2, 7, 0), (3, 9, 1), (4, 10, 2), (4, 12, 3), (5, 11, 4)])
def test_dp_matches_brute_force(K, M, seed):
    rng = np.random.default_rng(seed)
    S, G = rng.uniform(0.2, 3.0, K), rng.uniform(0.0, 2.0, K)
    n = dp_allocate(S, G, M)
    assert all(isinstance(x, int) for x in n) and min(n) >= 1 and sum(n) == M
    best = min(dp_objective(c, S, G) for c in _compositions(M, K))
    assert dp_objective(n, S, G) <= best + 1e-12


def test_notebook_bruteforce_case():
    S, G, N = np.array([.2, .5, .7]), np.array([.1, .3, .05]), 9
    alloc = dp_allocate(S, G, N)
    brute = min(sum(G[k] * np.expm1(S[k] / ns[k]) for k in range(3))
                for ns in itertools.product(range(1, N + 1), repeat=3) if sum(ns) == N)
    assert np.isclose(dp_objective(alloc, S, G), brute, rtol=1e-13) and sum(alloc) == N and min(alloc) >= 1


@pytest.mark.parametrize("M,K", [(8, 4), (16, 8), (24, 12), (254, 8)])
def test_notebook_finite_bound_matches_dp(M, K):
    sig, cdf, pdf = _table()
    lo, hi = 0.08, float(sig[-1])
    s, alloc = dp_dgc_sigmas(sig, cdf, M + 1, lo, hi, K, pdf=pdf)
    S, G, _, _ = block_quantities(sig, cdf, lo, hi, K, pdf=pdf)
    bound = DGCIncrement(sig, pdf, lo, hi).finite_step_bound(s)
    assert np.all(np.diff(s) < 0) and len(s) == M + 1 and sum(alloc) == M
    assert np.isclose(bound, dp_objective(alloc[::-1], S, G), rtol=1e-10)


def test_dp_edge_cases():
    assert dp_allocate([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 3) == [1, 1, 1]      # M == K: all ones
    with pytest.raises(ValueError):
        dp_allocate([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 2)                     # M < K
    assert dp_allocate([1.0, 1.0], [0.0, 1.0], 6)[0] == 1                     # zero-weight block gets the minimum
    sig, cdf, pdf = _table()
    for M, K in [(3, 4), (0, 4), (7, 8)]:                                     # notebook's bad inputs + one more
        with pytest.raises(ValueError):
            dp_dgc_sigmas(sig, cdf, M + 1, 0.08, 76.0, K=K, pdf=pdf)
    with pytest.raises(ValueError):
        block_quantities(sig, cdf, 0.08, 76.0, 8)                            # increment weight needs the pdf
    s, alloc = dp_dgc_sigmas(sig, cdf, 9, 0.08, 76.0, K=8, pdf=pdf)          # smallest legal N for K=8
    assert alloc == [1] * 8 and len(s) == 9


@pytest.mark.parametrize("weight", ["increment", "length_squared"])
@pytest.mark.parametrize("K", [4, 8, 12])
@pytest.mark.parametrize("N", [32, 64, 128, 256, 384, 512])
def test_real_table_grid(K, N, weight):
    sig, cdf, pdf = _table()
    lo, hi = 0.08, float(sig[-1])
    s, alloc = dp_dgc_sigmas(sig, cdf, N, lo, hi, K, pdf=pdf, weight=weight)
    assert len(alloc) == K and sum(alloc) == N - 1 and min(alloc) >= 1
    assert all(isinstance(x, int) for x in alloc)
    assert len(s) == N and s[0] == hi and s[-1] == lo
    assert np.all(np.diff(s) < 0)                                            # strictly decreasing, like "entropic"
    S, G, _, m = block_quantities(sig, cdf, lo, hi, K, pdf=pdf, weight=weight)
    # the DP solution is no worse than the rounded continuous allocation (the alternative it replaces)
    cont = np.maximum(1, np.round(m * (N - 1))).astype(int)
    cont[np.argmax(cont)] += (N - 1) - cont.sum()
    if cont.min() >= 1:
        assert dp_objective(alloc[::-1], S, G) <= dp_objective(cont, S, G) + 1e-9


@pytest.mark.parametrize("weight", ["increment", "length_squared"])
def test_large_n_limit(weight):
    sig, cdf, pdf = _table()
    lo, hi, K = 0.08, float(sig[-1]), 8
    S, G, _, m = block_quantities(sig, cdf, lo, hi, K, pdf=pdf, weight=weight)
    target = np.sqrt(S * G) / np.sqrt(S * G).sum()          # the continuum optimum of sum G_k S_k / n_k
    if weight == "length_squared":
        assert np.allclose(target, m / m.sum())             # ... which is the continuous DGC allocation
    gaps = []
    for M in (64, 512, 8192):
        n = np.array(dp_allocate(S, G, M), float)
        live = target > 1e-3
        gaps.append(np.abs(n[live] / M - target[live]).max())
    assert gaps[-1] < 0.01 and gaps[-1] <= gaps[0]


def _schedule():
    cfg = NS(diffusion=NS(continuous=NS(sigma_min=0.002, sigma_max=80.0)),
             evaluation=NS(entropic_blend_alpha=0.0, dp_dgc_blocks=8))
    return SigmaSchedule(None, cfg, torch.device("cpu"))


@pytest.mark.parametrize("N", [31, 255, 511])
def test_schedule_matches_continuous_convention(N):
    sch = _schedule()
    a = sch.prepare(schedule="entropic", num_steps=N, entropy_run_dir=TABLE, sigma_min_override=0.08)
    b = sch.prepare(schedule="dp_dgc", num_steps=N, entropy_run_dir=TABLE, sigma_min_override=0.08)
    assert b.shape == a.shape and b.dtype == a.dtype
    assert torch.allclose(b[[0, -1]], a[[0, -1]])                            # same sigma range and terminal sigma
    assert torch.all(b[1:] < b[:-1])


def test_nfe_accounting_identical():
    for t in (32, 64, 256, 512):
        kw = dict(framework="continuous_score", target_nfe=t, self_condition=True, sc_refresh_mode="carry")
        assert steps_for_target_nfe(sampler_name="ddim_dp_dgc", **kw) == steps_for_target_nfe(sampler_name="ddim_entropic", **kw)
    assert compute_nfe("continuous_score", "ddim_dp_dgc", 255, self_condition=True, sc_refresh_mode="carry") == \
        compute_nfe("continuous_score", "ddim_entropic", 255, self_condition=True, sc_refresh_mode="carry")

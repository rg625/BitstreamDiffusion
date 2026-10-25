"""Gamma-dependent Euler schedule (euler_dgc_derivation.pdf, 23 Oct 2026): Phase-1 acceptance checks on CPU.

* exact local-KL oracles (Appendix A.1 Gaussian table, A.2 binary mixture) approach the predicted coefficient
  (1/4) E||(1 + lambda) J - (lambda + 3/2) I||^2 as delta shrinks;
* the forward-only estimator matches the exact finite-bin identity for a Gaussian denoiser, tends to its small-bin
  limit, and gives A = 9/4, B = 1 (constant denoiser) and A = 1/4, B = 0 (identity);
* the allocator matches a dense two-bin search, gives a uniform density for a constant profile, handles a negative
  multiplier and rejects invalid profiles; grids keep their endpoints and put the right mass in unequal and empty bins;
* the literal Euler step has the stated one-step mean and variance, and model calls are counted, not assumed.
"""
import math

import numpy as np
import pytest
import torch
from numpy.polynomial.hermite_e import hermegauss

from diffusion.continuous.forward_profile import profile_batch
from diffusion.continuous.gamma_grid import (
    finite_check_grid, gamma_euler_grid, grid_from_density, restrict_density, solve_gamma_density,
)

# ------------------------------------------------------------------ local-KL oracles (Appendix A)


def predicted_coeff(J, lam):
    return 0.25 * ((1 + lam) * J - (lam + 1.5)) ** 2


def gaussian_exact_coeff(v, sigma, lam, delta):
    a0 = 1.0 / sigma ** 2
    a1 = a0 * math.exp(2 * delta)
    Phi = lambda a: (a / a0) ** ((1 - lam) / 2) * ((1 + v * a) / (1 + v * a0)) ** ((1 + lam) / 2)
    xs, ws = np.polynomial.legendre.leggauss(64)
    aa = 0.5 * (a1 - a0) * xs + 0.5 * (a1 + a0)
    integral = 0.5 * (a1 - a0) * np.sum(ws * lam / np.array([Phi(a) for a in aa]) ** 2)
    fP, VP = Phi(a1) * a0 / a1, (Phi(a1) / a1) ** 2 * integral
    Delta = sigma * (1 - math.exp(-delta))
    fQ, VQ = 1 - (1 + lam) * sigma * Delta / (v + sigma ** 2), 2 * lam * sigma * Delta
    kl = 0.5 * (math.log(VQ / VP) + VP / VQ - 1 + (fP - fQ) ** 2 * (v + sigma ** 2) / VQ)
    return kl / delta ** 2


@pytest.mark.parametrize("lam,table", [(0.25, 0.31637960), (0.5, 0.39058734), (1.0, 0.56243750),
                                       (2.0, 0.99986498), (4.0, 2.24959711)])
def test_gaussian_local_kl_oracle(lam, table):
    exact = gaussian_exact_coeff(1.0, 1.0, lam, 1e-4)
    assert exact == pytest.approx(table, rel=2e-4)                 # the PDF's Appendix A.1 table
    assert exact == pytest.approx(predicted_coeff(0.5, lam), rel=5e-4)


def binary_exact_coeff(delta, sigma=1.0):
    """Equiprobable +-1 prior, lambda = 1: KL(exact reverse transition || literal Euler), averaged over x ~ p_sigma."""
    t = sigma ** 2
    tp = t * math.exp(-2 * delta)
    r = tp / t
    Delta = sigma * (1 - math.exp(-delta))
    zg, wg = hermegauss(80)
    wg = wg / wg.sum()
    xs = np.concatenate([1 + sigma * zg, -1 + sigma * zg])         # x ~ 0.5 N(1, t) + 0.5 N(-1, t)
    wx = np.concatenate([wg, wg]) * 0.5
    tot = 0.0
    for x, w in zip(xs, wx):
        pplus = 1 / (1 + math.exp(-2 * x / t))                       # P(X0 = 1 | x)
        m = math.tanh(x / t)
        muQ, VQ = x + 2 * sigma * Delta * (m - x) / t, 2 * sigma * Delta
        VP = tp * (1 - r)
        mus, ps = (r * x + (1 - r), r * x - (1 - r)), (pplus, 1 - pplus)
        kl = 0.0
        for mu_c, p_c in zip(mus, ps):
            y = mu_c + math.sqrt(VP) * zg                              # quadrature under each component
            logP = np.log(sum(pk * np.exp(-(y - mk) ** 2 / (2 * VP)) for mk, pk in zip(mus, ps))) - 0.5 * math.log(2 * math.pi * VP)
            logQ = -(y - muQ) ** 2 / (2 * VQ) - 0.5 * math.log(2 * math.pi * VQ)
            kl += p_c * np.sum(wg * (logP - logQ))
        tot += w * kl
    return tot / delta ** 2


def test_binary_local_kl_oracle():
    assert binary_exact_coeff(1e-3) == pytest.approx(0.75452680, rel=2e-3)   # Appendix A.2
    assert abs(binary_exact_coeff(1e-3) - 0.75505657) < abs(binary_exact_coeff(1e-2) - 0.75505657)


# ------------------------------------------------------------------ forward estimator


def _profile(denoise, d=20000, edges=(0.5, 0.55), B=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x0 = (torch.rand(B, d, generator=g) > 0.5).float()
    return profile_batch(denoise, x0, torch.tensor(edges, dtype=torch.float64), generator=g)


def test_constant_and_identity_denoisers():
    st = _profile(lambda x, s: torch.full_like(x, 0.5))
    assert st["A"].mean().item() == pytest.approx(2.25, rel=0.02) and st["B"].mean().item() == pytest.approx(1.0, rel=0.02)
    st = _profile(lambda x, s: x.clone())
    # identity: dD = dX, so A = V / 4 and B = 0 sample by sample (A -> 1/4 in expectation since E V = 1)
    assert torch.allclose(st["A"], 0.25 * st["V"]) and torch.all(st["B"] == 0)
    assert st["A"].mean().item() == pytest.approx(0.25, rel=0.02)


def test_gaussian_denoiser_finite_bin_identity():
    """X0 ~ N(0, v) per coordinate, D = v x / (v + sigma^2): E R_c = v^2/((v+a)(v+b)) - 2 c v/(v+b) + c^2."""
    v, a, b, d = 1.3, 0.4, 0.9, 400_000
    g = torch.Generator().manual_seed(1)
    x0 = torch.randn(1, d, generator=g, dtype=torch.float64).float() * math.sqrt(v)
    st = profile_batch(lambda x, s: v * x / (v + s * s), x0, torch.tensor([a, b], dtype=torch.float64), generator=g)
    for c, key in ((1.5, "A"), (1.0, "B")):
        exact = v ** 2 / ((v + a) * (v + b)) - 2 * c * v / (v + b) + c ** 2
        assert st[key].item() == pytest.approx(exact, rel=0.01)
    # small-bin limit (v/(v+a) - c)^2
    st = profile_batch(lambda x, s: v * x / (v + s * s), x0, torch.tensor([a, a * 1.001], dtype=torch.float64),
                       generator=g)
    assert st["A"].item() == pytest.approx((v / (v + a) - 1.5) ** 2, rel=0.01)
    assert st["V"].item() == pytest.approx(1.0, rel=0.01)


def test_gen_mask_noises_only_generated_coordinates():
    seen = {}
    x0 = torch.zeros(2, 10)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, 5:] = True

    def den(x, s):
        seen.setdefault("x", x.clone())
        return torch.zeros_like(x)
    st = profile_batch(den, x0, torch.tensor([0.1, 0.2], dtype=torch.float64), gen_mask=mask)
    assert torch.all(seen["x"][:, :5] == 0) and torch.all(seen["x"][:, 5:] != 0)
    assert st["V"].shape == (2, 1)


# ------------------------------------------------------------------ allocator and grids


def _objective(p, A, B, w, S):
    return float(np.sum(w * (A / p + S * S * B * p)))


@pytest.mark.parametrize("A,B,S", [((1.0, 3.0), (0.5, 2.0), 2.0), ((0.3, 0.3), (0.0, 5.0), 4.0),
                                   ((2.0, 0.5), (1.0, 0.1), 0.0)])
def test_allocator_matches_two_bin_search(A, B, S):
    A, B, e = np.array(A), np.array(B), np.array([0.0, 0.7, 2.0])
    w = np.diff(e)
    p, _ = solve_gamma_density(A, B, e, S)
    assert np.sum(w * p) == pytest.approx(1.0, abs=1e-9)
    grid = np.linspace(1e-4, 1 / w[0] - 1e-4, 200_001)
    obj = [_objective(np.array([p0, (1 - w[0] * p0) / w[1]]), A, B, w, S) for p0 in grid[::50]]
    assert _objective(p, A, B, w, S) <= min(obj) + 1e-6


def test_allocator_constant_profile_uniform_and_negative_eta():
    e = np.linspace(-2.5, 4.4, 65)
    p, _ = solve_gamma_density(np.full(64, 0.7), np.full(64, 0.3), e, S=3.0)
    assert np.allclose(p, 1 / (e[-1] - e[0]))
    # tiny band (widths sum 0.1) forces a large density, hence a negative multiplier; denominators stay positive
    e2 = np.array([0.0, 0.05, 0.1])
    A2, B2 = np.array([1.0, 2.0]), np.array([1.0, 1.0])
    p2, eta = solve_gamma_density(A2, B2, e2, S=1.0)
    assert eta < 0 and np.all(B2 + eta > 0) and np.sum(np.diff(e2) * p2) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("A,B", [((0.0, 1.0), (1.0, 1.0)), ((1.0, np.nan), (1.0, 1.0)), ((1.0, 1.0), (-0.1, 1.0))])
def test_allocator_rejects_invalid_profiles(A, B):
    with pytest.raises(ValueError):
        solve_gamma_density(np.array(A), np.array(B), np.array([0.0, 1.0, 2.0]), S=1.0)


def test_grid_masses_endpoints_and_empty_bins():
    e = np.array([0.0, 0.2, 1.0, 1.5, 3.0])                      # unequal widths
    p = np.array([1.0, 0.0, 2.0, 0.5])                           # one empty bin
    N = 4000
    s = grid_from_density(e, p, N)
    finite_check_grid(s, N, 1.0, math.exp(3.0))
    m = np.diff(e) * p / np.sum(np.diff(e) * p)
    u = np.log(s)
    counts = np.histogram(0.5 * (u[:-1] + u[1:]), bins=e)[0] / N
    assert np.allclose(counts, m, atol=2e-3)
    assert not np.any((u > e[1] + 1e-12) & (u < e[2] - 1e-12))   # no grid point inside the empty bin (one jump)


def test_restrict_density_splits_boundary_bins():
    e, p = np.array([0.0, 1.0, 2.0]), np.array([1.0, 3.0]) / 4.0
    ne, np_ = restrict_density(e, p, 0.5, 1.5)
    assert np.allclose(ne, [0.5, 1.0, 1.5]) and np.allclose(np_, [0.5, 1.5])


def test_gamma_grid_profile_roundtrip():
    e = np.linspace(math.log(0.08), math.log(80.0), 65)
    u = 0.5 * (e[:-1] + e[1:])
    prof = dict(log_sigma_edges=e, A=0.25 + np.exp(-(u - 0.0) ** 2), B=0.2 * np.exp(-(u - 0.5) ** 2))
    for N, gamma in ((63, 0.2), (255, 0.2), (127, 4 / 127)):
        s, info = gamma_euler_grid(prof, N, gamma)
        finite_check_grid(s, N, 0.08, 80.0)
        assert info["S"] == pytest.approx(N * gamma) and info["lambda_min"] > 0


# ------------------------------------------------------------------ sampler: Euler step and model-call count


def test_euler_lambda_step_moments():
    from diffusion.continuous.samplers import EulerLambdaSampler
    from tests._sampler_harness import TinyBinaryDenoiser, make_cpu_cfg
    from diffusion.continuous.processes import ContinuousForwardProcess
    cfg = make_cpu_cfg(num_steps=8)
    sam = EulerLambdaSampler(TinyBinaryDenoiser(16), ContinuousForwardProcess(cfg), cfg, gamma=0.3)
    torch.manual_seed(0)
    n = 1_000_000
    x = torch.zeros(n, 1)
    d = torch.full((n, 1), 0.4)
    sc, sn = torch.tensor(2.0), torch.tensor(1.7)
    out = sam._integrate_step(x, sn - sc, d, sigma_cur=sc, sigma_next=sn, prefix_mask=None)
    lam = 0.3 * 2.0 / 0.3
    assert out.mean().item() == pytest.approx(float((sn - sc) * (1 + lam) * 0.4), abs=0.006)   # ~4 s.e.
    assert out.var().item() == pytest.approx(2 * 0.3 * 2.0 ** 2, rel=0.01)   # 2 lambda sigma Delta = 2 gamma sigma^2


@pytest.mark.parametrize("mode,expected", [("carry", lambda n: n), ("null", lambda n: n),
                                           ("refined", lambda n: 2 * (n - 1) + 1)])
def test_model_calls_are_counted(mode, expected):
    from diffusion.continuous.processes import ContinuousForwardProcess
    from diffusion.continuous.samplers import DDIMSampler
    from tests._sampler_harness import TinyBinaryDenoiser, make_cpu_cfg
    cfg = make_cpu_cfg(num_steps=12)
    model = TinyBinaryDenoiser(16)
    calls = {"n": 0}
    model.register_forward_hook(lambda *a: calls.__setitem__("n", calls["n"] + 1))
    sam = DDIMSampler(model, ContinuousForwardProcess(cfg), cfg)
    sam.sample(num_samples=2, seq_len=16, num_steps=12, schedule="karras", sc_refresh_mode=mode, ati_eta=0.0,
               return_probs=True, progress=False)
    assert calls["n"] == expected(12)   # num_steps sigma points: one call per interval plus the final decode


def test_euler_lambda_fixed_S_sets_gamma_per_call():
    from diffusion.continuous.processes import ContinuousForwardProcess
    from diffusion.continuous.samplers import EulerLambdaSampler
    from tests._sampler_harness import TinyBinaryDenoiser, make_cpu_cfg
    cfg = make_cpu_cfg(num_steps=12)
    cfg.evaluation.euler_S = 4.0
    sam = EulerLambdaSampler(TinyBinaryDenoiser(16), ContinuousForwardProcess(cfg), cfg)
    for n in (12, 9):
        sam.sample(num_samples=2, seq_len=16, num_steps=n, schedule="karras", sc_refresh_mode="null", ati_eta=0.0,
                   return_probs=True, progress=False)
        assert sam.gamma == pytest.approx(4.0 / (n - 1))
    cfg.evaluation.euler_gamma = 0.1
    with pytest.raises(ValueError):
        EulerLambdaSampler(TinyBinaryDenoiser(16), ContinuousForwardProcess(cfg), cfg)


def test_karras_grid_matches_sampler_schedule():
    from diffusion.continuous.gamma_grid import karras_grid
    from diffusion.continuous.samplers import SigmaSchedule
    from diffusion.continuous.processes import ContinuousForwardProcess
    from tests._sampler_harness import make_cpu_cfg
    cfg = make_cpu_cfg(num_steps=12)
    ref = SigmaSchedule(ContinuousForwardProcess(cfg), cfg, torch.device("cpu"))._karras_schedule(64, sigma_min=0.08, sigma_max=80.0).double().numpy()
    np.testing.assert_allclose(karras_grid(0.08, 80.0, 63), ref, rtol=1e-5)

"""Window ordering: t_j = clip(tau(1+L/k) - r_j/k, 0, 1), sigma_j = F^-1(t_j).

What these pin down, in the order the experiment depends on them:
  * k = inf IS the uniform model: every token at the global sigma, full loss mask;
  * ranks are absolute, so padding after the answer cannot dilute its order;
  * every loss-bearing token has sigma ~ F whatever k and mode are, and arms at
    the same k see an identical multiset of sigmas -- the "identical training-
    noise distribution" claim is tested, not asserted;
  * the draws never touch the global RNG, so arms share data, noise and tau;
  * the sampler spends exactly N evaluations and at k = inf is plain Euler.
"""
import math
from types import SimpleNamespace

import pytest
import torch

from diffusion.continuous.ordering import (
    NoiseCDF,
    sample_window,
    training_window_sigma,
    window_ranks,
    window_time,
)

BPT = 4
F = NoiseCDF.lognormal(n_bins=512)


def _prefix(B, n_tok, plen):
    """[B, n_tok*BPT] prompt mask with per-example prompt lengths `plen`."""
    pm = torch.zeros(B, n_tok * BPT, dtype=torch.bool)
    for b, p in enumerate(plen):
        pm[b, : p * BPT] = True
    return pm


# ------------------------------------------------------------------ NoiseCDF
def test_noise_cdf_roundtrip_and_endpoints():
    t = torch.rand(20000)
    assert torch.allclose(F.cdf_of(F.icdf(t)), t, atol=1e-5)
    assert F.icdf(torch.zeros(1)).item() == pytest.approx(0.002, rel=1e-5)
    assert F.icdf(torch.ones(1)).item() == pytest.approx(F.sigma_hi, rel=1e-6)
    assert bool((F.icdf(torch.linspace(0, 1, 1000)).diff() >= 0).all())


def test_noise_cdf_matches_the_trainer_bin_sampler():
    """icdf(U) must be the distribution sample_entropy_sigma draws from."""
    g = torch.Generator().manual_seed(0)
    edges = torch.logspace(math.log10(0.002), math.log10(80.0), 33)
    mass = torch.rand(32, generator=g) ** 3
    mass[5] = 0.0                                   # a zero-mass bin must be skipped
    cdf = torch.cumsum(mass, 0) / mass.sum()
    Fb = NoiseCDF(edges, cdf)
    n = 200_000
    ours = Fb.icdf(torch.rand(n, generator=g)).double()
    idx = torch.searchsorted(cdf, torch.rand(n, generator=g)).clamp(0, 31)
    lo, hi = edges[idx].log(), edges[idx + 1].log()
    ref = (lo + torch.rand(n, generator=g) * (hi - lo)).exp().double()
    # KS distance on a grid; quantiles are discontinuous across a zero-mass bin
    grid = torch.logspace(math.log10(0.002), math.log10(80.0), 400, dtype=torch.float64)
    ks = ((ours.view(-1, 1) <= grid).double().mean(0)
          - (ref.view(-1, 1) <= grid).double().mean(0)).abs().max()
    assert ks < 0.006, ks
    assert not bool(((ours > edges[5]) & (ours < edges[6])).any())


# ------------------------------------------------------------------ ranks
def test_ranks_are_absolute_and_directional():
    sm = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    r, L = window_ranks("l2r", sm)
    assert L.tolist() == [5, 6]
    assert r[0].tolist() == [0, 0, 4, 3, 2, 1, 0]   # first answer token first
    r, _ = window_ranks("r2l", sm)
    assert r[1].tolist() == [0, 0, 1, 2, 3, 4, 5]


def test_random_ranks_are_a_permutation_of_the_suffix_only():
    g = torch.Generator().manual_seed(1)
    sm = torch.zeros(64, 40, dtype=torch.bool)
    for b in range(64):
        sm[b, b % 17:] = True
    r, L = window_ranks("random", sm, generator=g)
    for b in range(64):
        got = sorted(r[b][sm[b]].long().tolist())
        assert got == list(range(int(L[b])))
        assert bool((r[b][~sm[b]] == 0).all())


def test_random_ranks_never_touch_the_global_rng():
    torch.manual_seed(123)
    before = torch.get_rng_state()
    window_ranks("random", torch.ones(8, 30, dtype=torch.bool),
                 generator=torch.Generator().manual_seed(0))
    assert torch.equal(before, torch.get_rng_state())


# ------------------------------------------------------------------ window_time
def test_k_inf_is_the_uniform_model():
    r, L = window_ranks("l2r", torch.ones(3, 20, dtype=torch.bool))
    tau = torch.tensor([0.1, 0.5, 0.9])
    t = window_time(tau, r, L, float("inf"))
    assert torch.equal(t, tau.view(-1, 1).expand(3, 20))


def test_endpoints_and_active_window_width():
    r, L = window_ranks("l2r", torch.ones(1, 200, dtype=torch.bool))
    assert bool((window_time(torch.ones(1), r, L, 16) == 1).all())
    assert bool((window_time(torch.zeros(1), r, L, 16) == 0).all())
    for k in (4, 16, 64):
        for tau in (0.3, 0.5, 0.7):
            t = window_time(torch.tensor([tau]), r, L, k)
            n_int = int(((t > 0) & (t < 1)).sum())
            assert k - 1 <= n_int <= k + 1, (k, tau, n_int)


def test_interior_time_is_uniform_with_mass_k_over_k_plus_L():
    """Why the arms share one noise distribution: t_j | interior ~ U(0,1)."""
    g = torch.Generator().manual_seed(0)
    Ln, k = 120, 16
    r, L = window_ranks("l2r", torch.ones(1, Ln, dtype=torch.bool))
    tau = torch.rand(200_000, generator=g)
    for j in (0, 60, 119):
        t = window_time(tau, r.expand(len(tau), -1), L.expand(len(tau)), k)[:, j]
        inner = t[(t > 0) & (t < 1)]
        assert inner.numel() / len(tau) == pytest.approx(k / (k + Ln), abs=4e-3)
        qs = torch.linspace(0.05, 0.95, 19)
        assert torch.allclose(inner.quantile(qs), qs, atol=0.01)


# ------------------------------------------------------------------ training
def _train(mode, k, tau, pm, seed=0):
    return training_window_sigma(tau, n_bits=pm.shape[1], bits_per_token=BPT,
                                 mode=mode, k=k, noise=F, prefix_mask=pm,
                                 generator=torch.Generator().manual_seed(seed))


def test_training_k_inf_is_bit_identical_to_global_sigma():
    tau = 1.0 - torch.rand(5)
    pm = _prefix(5, 30, [3, 7, 1, 12, 5])
    sig, keep, sg = _train("none", float("inf"), tau, pm)
    assert torch.equal(sig, sg.view(-1, 1).expand_as(sig))
    assert torch.equal(keep, ~pm)
    assert torch.allclose(sg, F.icdf(tau))


@pytest.mark.parametrize("k", [4, 16, 64])
def test_modes_at_the_same_k_share_the_sigma_multiset(k):
    g = torch.Generator().manual_seed(3)
    tau = 1.0 - torch.rand(16, generator=g)
    pm = _prefix(16, 150, [int(x) for x in torch.randint(5, 40, (16,), generator=g)])
    outs = {m: _train(m, k, tau, pm) for m in ("l2r", "r2l", "random")}
    for b in range(16):
        ref = None
        for m, (sig, keep, sg) in outs.items():
            suf = sig[b][~pm[b]].sort().values
            n_keep = int(keep[b].sum())
            if ref is None:
                ref = (suf, n_keep, sg[b])
            assert torch.equal(suf, ref[0]) and n_keep == ref[1] and sg[b] == ref[2]


def test_prompt_bits_keep_global_sigma_and_bits_share_their_token_sigma():
    tau = torch.tensor([0.6, 0.2])
    pm = _prefix(2, 40, [6, 11])
    sig, keep, sg = _train("random", 8, tau, pm)
    for b in range(2):
        assert bool((sig[b][pm[b]] == sg[b]).all())
        assert not bool(keep[b][pm[b]].any())
    blocks = sig.view(2, 40, BPT)
    assert torch.equal(blocks, blocks[..., :1].expand_as(blocks))


def test_keep_is_exactly_the_window_interior():
    tau = torch.tensor([0.3])
    pm = _prefix(1, 100, [10])
    sig, keep, _ = _train("l2r", 8, tau, pm)
    s_tok, k_tok = sig.view(1, 100, BPT)[0, :, 0], keep.view(1, 100, BPT)[0, :, 0]
    suf = ~pm.view(1, 100, BPT)[0, :, 0]
    lo, hi = F.icdf(torch.zeros(1)), F.icdf(torch.ones(1))
    assert bool(((s_tok == lo) & suf).any()) and bool(((s_tok == hi) & suf).any())
    assert torch.equal(k_tok, suf & (s_tok > lo) & (s_tok < hi))


def test_loss_bearing_share_of_the_answer_is_mode_independent():
    """The defect this mask fixes: with the t=1 atom supervised, l2r and r2l
    put very different fractions of the answer under loss."""
    B, n_tok, plen, ans = 20000, 120, 10, 30
    pm = _prefix(B, n_tok, [plen] * B)
    content = torch.zeros(B, n_tok, dtype=torch.bool)
    content[:, plen:plen + ans] = True
    tau = (torch.arange(B) + 0.5) / B          # a quadrature grid, not draws
    share = {}
    for m in ("l2r", "r2l", "random"):
        _, keep, _ = _train(m, 16, tau, pm)
        share[m] = float(keep.view(B, n_tok, BPT)[..., 0][content].float().mean())
    expect = 16 / (16 + (n_tok - plen))
    for m, v in share.items():
        assert v == pytest.approx(expect, abs=2e-4), (m, share)


def test_l2r_resolves_the_answer_front_first():
    pm = _prefix(1, 300, [40])
    for mode, sign in (("l2r", 1), ("r2l", -1)):
        sig, _, _ = _train(mode, 16, torch.tensor([0.8]), pm)
        s = sig.view(300, BPT)[40:, 0]
        d = sign * s.diff()
        assert bool((d >= 0).all()), mode              # monotone along the answer
        inner = (s > s.min()) & (s < s.max())
        assert 14 <= int(inner.sum()) <= 17, mode      # ~k tokens in the window


# ------------------------------------------------------------------ trainer stream
def test_trainer_draw_matches_across_arms_and_is_resume_exact():
    from trainers.trainer import Trainer

    def fake(mode, k, step):
        cfg = SimpleNamespace(
            train=SimpleNamespace(seed=42, ordering=SimpleNamespace(
                enabled=True, kind="window", mode=mode, k=k,
                noise_table=None)),
            data=SimpleNamespace(bits_per_token=BPT))
        t = SimpleNamespace(cfg=cfg, device=torch.device("cpu"), rank=1,
                            global_step=step, _window_noise=F,
                            _window_gen_tau=torch.Generator(),
                            _window_gen_key=torch.Generator(), _window_micro=0)
        return t

    pm = _prefix(8, 60, [5] * 8)
    torch.manual_seed(0)
    rng = torch.get_rng_state()
    a = Trainer._draw_window_sigma(fake("l2r", 16, 7), 8, 240, pm, False, True, True)
    b = Trainer._draw_window_sigma(fake("random", 16, 7), 8, 240, pm, False, True, True)
    c = Trainer._draw_window_sigma(fake("l2r", 16, 7), 8, 240, pm, False, True, True)
    d = Trainer._draw_window_sigma(fake("l2r", 16, 8), 8, 240, pm, False, True, True)
    assert torch.equal(torch.get_rng_state(), rng)          # global stream untouched
    assert torch.equal(a[2], b[2])                           # same tau across modes
    assert torch.equal(a[0], c[0])                           # same step -> same draw
    assert not torch.equal(a[2], d[2])                       # next step -> new draw


# ------------------------------------------------------------------ sampler
def test_sampler_k_inf_is_euler_on_the_quantile_grid_with_n_minus_1_calls():
    torch.manual_seed(0)
    B, n_tok, N = 2, 6, 9
    x0 = torch.randn(B, n_tok * BPT) * F.sigma_hi
    r, L = window_ranks("none", torch.ones(B, n_tok, dtype=torch.bool))
    calls = []

    def den(x, s):
        calls.append(s.clone())
        return torch.sigmoid(0.5 * x)

    got = sample_window(den, num_steps=N, x_init=x0, noise=F, r=r, L=L,
                        k=float("inf"), bits_per_token=BPT)
    assert len(calls) == N - 1
    x = x0.clone()
    taus = torch.linspace(1.0, 0.0, N)
    for i in range(N - 1):
        sc, sn = float(F.icdf(taus[i:i + 1])), float(F.icdf(taus[i + 1:i + 2]))
        x = x + (sn - sc) * (x - torch.sigmoid(0.5 * x)) / sc
    assert torch.allclose(got, x, atol=1e-4, rtol=1e-4)


def test_sampler_leaves_tokens_outside_the_window_untouched():
    B, n_tok = 1, 50
    x0 = torch.randn(B, n_tok * BPT)
    r, L = window_ranks("l2r", torch.ones(B, n_tok, dtype=torch.bool))
    seen = []

    def den(x, s):
        seen.append(s.clone())
        return torch.zeros_like(x)

    sample_window(den, num_steps=4, x_init=x0, noise=F, r=r, L=L, k=2, bits_per_token=BPT)
    # at the first step the tail of an l2r order has not entered: sigma at the top
    assert torch.allclose(seen[0].view(n_tok, BPT)[-1], torch.full((BPT,), F.sigma_hi))


def test_supervision_matched_control_supervises_like_the_window_arm():
    B, n_tok, plen = 20000, 120, 10
    pm = _prefix(B, n_tok, [plen] * B)
    tau = (torch.arange(B) + 0.5) / B
    _, kw, _ = _train("l2r", 16, tau, pm)
    sig, km, sg = training_window_sigma(tau, n_bits=n_tok * BPT, bits_per_token=BPT,
                                        mode="none", k=float("inf"), noise=F, prefix_mask=pm,
                                        generator=torch.Generator().manual_seed(0),
                                        loss_match_k=16)
    assert torch.equal(sig, sg.view(-1, 1).expand_as(sig))       # still the uniform field
    assert float(km.float().mean()) == pytest.approx(float(kw.float().mean()), rel=0.01)
    assert not bool(km[pm].any())

"""Temporal-ordering schedule: definition, invariants, and the w=0 no-op gate."""
import torch

from diffusion.continuous.losses import _sigma_weight
from diffusion.continuous.ordering import (
    expand_token_sigma_to_bits, ordering_ranks, positional_time,
)


class _Cfg:
    class train:  loss_weighting = "edm"
    class diffusion:
        class continuous: sigma_data = 0.3998
    diffusion.continuous = diffusion.continuous


def test_w0_reproduces_the_current_model_exactly():
    """The whole ordering branch rests on this: w=0 IS today's model."""
    t = torch.rand(4)
    for mode in ("l2r", "r2l", "random", "none"):
        u = ordering_ranks(mode, 12, 4)
        tp = positional_time(t, u, 0.0)
        assert torch.equal(tp, t.view(-1, 1).expand(4, 12)), mode


def test_ranks_cover_the_suffix_only():
    """Prompt tokens must not consume ordering range, or L2R becomes prompt-first."""
    sm = torch.zeros(2, 8, dtype=torch.bool); sm[:, 3:] = True
    u = ordering_ranks("l2r", 8, 2, suffix_mask=sm)
    assert torch.equal(u[0, :3], torch.zeros(3))          # prompt pinned at 0
    assert u[0, 3] == 0.0 and u[0, -1] == 1.0             # suffix spans the full range
    assert torch.all(u[0, 3:].diff() > 0)                 # and is strictly increasing


def test_l2r_and_r2l_are_mirror_images():
    u_l = ordering_ranks("l2r", 10, 1)
    u_r = ordering_ranks("r2l", 10, 1)
    assert torch.allclose(u_l, 1.0 - u_r, atol=1e-6)


def test_lower_rank_resolves_earlier():
    """The stated convention: lower u reaches low sigma first."""
    u = ordering_ranks("l2r", 6, 1)
    tp = positional_time(torch.tensor([0.5]), u, 0.6)
    # t_j is LARGER for small u; with sigma increasing in t that means the
    # low-rank token is further along, i.e. resolves first.
    assert tp[0, 0] > tp[0, -1]


def test_random_ordering_is_resampled_not_fixed():
    """A per-example fixed permutation would be memorisable."""
    g = torch.Generator().manual_seed(0)
    a = ordering_ranks("random", 32, 4, generator=g)
    b = ordering_ranks("random", 32, 4, generator=g)
    assert not torch.equal(a, b), "permutation must differ between calls"
    # and it must be a genuine permutation, not noise
    for row in a:
        assert torch.allclose(row.sort().values, ordering_ranks("l2r", 32, 1)[0].sort().values)


def test_random_ordering_is_reproducible_under_a_seed():
    a = ordering_ranks("random", 16, 3, generator=torch.Generator().manual_seed(7))
    b = ordering_ranks("random", 16, 3, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_sigma_weight_uniform_per_position_equals_scalar():
    """Loss weighting must be elementwise-identical under a uniform schedule."""
    cfg = _Cfg()
    sig = torch.rand(5) * 3 + 0.05
    w_scalar = _sigma_weight(cfg, sig, ndim=3)                 # [B,1,1]
    w_pos = _sigma_weight(cfg, sig[:, None].expand(5, 9), ndim=3)   # [B,9,1]
    assert w_scalar.shape == (5, 1, 1) and w_pos.shape == (5, 9, 1)
    assert torch.allclose(w_scalar.expand_as(w_pos), w_pos, rtol=0, atol=0)


def test_token_sigma_expands_to_bits_blockwise():
    """All bits of a token share its sigma, or the codeword is incoherent."""
    s = torch.tensor([[1.0, 2.0, 3.0]])
    out = expand_token_sigma_to_bits(s, 4)
    assert out.shape == (1, 12)
    assert torch.equal(out[0], torch.tensor([1.]*4 + [2.]*4 + [3.]*4))


def test_ordered_sampler_at_w0_equals_a_plain_euler_loop():
    """The gate for the sampler: at w=0 every token sits at the global sigma,
    so the loop must reduce EXACTLY to ordinary probability-flow Euler."""
    from diffusion.continuous.ordering import sample_ordered

    torch.manual_seed(0)
    B, S, bpt = 2, 64, 16
    sigmas = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.4, 0.1])
    x0 = torch.randn(B, S)
    W = torch.randn(S, S) * 0.01
    denoise = lambda x, s: torch.sigmoid(x @ W)

    got = sample_ordered(denoise, sigmas=sigmas, x_init=x0, w=0.0, bits_per_token=bpt)

    # reference: the same ODE with one scalar sigma per step
    x = x0.clone()
    for i in range(len(sigmas) - 1):
        sc, sn = sigmas[i], sigmas[i + 1]
        D = denoise(x, sc)
        d = -sc * ((D - x) / sc ** 2)
        x = x + (sn - sc) * d
    assert torch.allclose(got, x, rtol=0, atol=1e-5), f"max {float((got-x).abs().max()):.2e}"


def test_ordered_sampler_respects_the_prompt_clamp():
    from diffusion.continuous.ordering import sample_ordered

    torch.manual_seed(0)
    B, S, bpt = 2, 32, 16
    pm = torch.zeros(B, S, dtype=torch.bool); pm[:, :16] = True
    pf = torch.full((B, S), 0.5)
    out = sample_ordered(lambda x, s: torch.rand_like(x), sigmas=torch.tensor([5.0, 1.0, 0.1]),
                         x_init=torch.randn(B, S), w=0.0, bits_per_token=bpt,
                         prefix_full=pf, prefix_mask=pm)
    assert torch.equal(out[pm], pf[pm]), "prompt positions must survive untouched"


def test_ordering_changes_the_trajectory_only_when_w_is_nonzero():
    from diffusion.continuous.ordering import ordering_ranks, sample_ordered

    torch.manual_seed(0)
    B, S, bpt = 2, 64, 16
    sig = torch.tensor([10.0, 2.0, 0.4, 0.1])
    x0 = torch.randn(B, S)
    W = torch.randn(S, S) * 0.01
    f = lambda x, s: torch.sigmoid(x @ W)
    u = ordering_ranks("l2r", S // bpt, B)
    base = sample_ordered(f, sigmas=sig, x_init=x0, u=u, w=0.0, bits_per_token=bpt)
    same = sample_ordered(f, sigmas=sig, x_init=x0, u=None, w=0.0, bits_per_token=bpt)
    diff = sample_ordered(f, sigmas=sig, x_init=x0, u=u, w=0.6, bits_per_token=bpt)
    assert torch.equal(base, same), "at w=0 the ranks must be irrelevant"
    assert not torch.allclose(base, diff), "at w>0 the ordering must matter"

"""OrderedSampler: the eval-facing per-position-sigma path.

The invariant that matters most: at order_w=0 this must reproduce the uniform-
sigma model exactly, so the ordering experiment's control arm is the current
model and not a subtly different one.
"""
import math

import ml_collections
import pytest
import torch

from diffusion.continuous.ordering import (
    expand_token_sigma_to_bits,
    ordering_ranks,
    positional_time,
    sample_ordered,
)
from diffusion.continuous.ordered_sampler import token_prefix_mask_from_bits

BPT = 4


def _sig(n=6, hi=8.0, lo=0.05):
    return torch.tensor([hi * (lo / hi) ** (i / (n - 1)) for i in range(n)])


def _recording_denoise(seen):
    def fn(x, sigma_bits):
        seen.append(sigma_bits.clone())
        return torch.sigmoid(x * 0.5)
    return fn


# --------------------------------------------------------------- w = 0 exactness
def test_w0_reproduces_a_handwritten_scalar_sigma_euler_loop():
    torch.manual_seed(0)
    B, n_tok = 2, 5
    S = n_tok * BPT
    sig = _sig()
    x0 = torch.randn(B, S)
    u = ordering_ranks("l2r", n_tok, B)

    got = sample_ordered(lambda x, s: torch.sigmoid(x * 0.5), sigmas=sig,
                         x_init=x0.clone(), u=u, w=0.0, bits_per_token=BPT)

    # Independent scalar-sigma Euler, written from the ODE directly.
    x = x0.clone()
    s_hi, s_lo = float(sig[0]), float(sig[-1])
    for i in range(len(sig) - 1):
        sc, sn = float(sig[i]), float(sig[i + 1])
        D = torch.sigmoid(x * 0.5)
        d = -sc * (D - x) / (sc ** 2)
        x = x + (sn - sc) * d
    assert torch.allclose(got, x, atol=1e-5), (got - x).abs().max()


def test_w0_gives_every_token_the_same_sigma():
    seen = []
    B, n_tok = 3, 6
    sample_ordered(_recording_denoise(seen), sigmas=_sig(), x_init=torch.zeros(B, n_tok * BPT),
                   u=ordering_ranks("random", n_tok, B, generator=torch.Generator().manual_seed(1)),
                   w=0.0, bits_per_token=BPT)
    for s in seen:
        assert torch.allclose(s, s.flatten()[0].expand_as(s)), "w=0 must be uniform in sigma"


def test_w0_is_invariant_to_the_ranks_themselves():
    """If ranks could leak in at w=0 the control arm would silently depend on
    the ordering mode, and 'ordering off' would not mean what it says."""
    B, n_tok = 2, 5
    x0 = torch.randn(B, n_tok * BPT)
    outs = []
    for mode in ("l2r", "r2l", "random", "none"):
        g = torch.Generator().manual_seed(7)
        u = ordering_ranks(mode, n_tok, B, generator=g)
        outs.append(sample_ordered(lambda x, s: torch.sigmoid(x * 0.5), sigmas=_sig(),
                                   x_init=x0.clone(), u=u, w=0.0, bits_per_token=BPT))
    for o in outs[1:]:
        assert torch.equal(outs[0], o)


# --------------------------------------------------- the intervention is ACTIVE
def test_w_positive_actually_spreads_sigma_across_positions():
    """The experiment is worthless if ordering does not change per-position sigma.
    This is the causal-activity check, asserted rather than eyeballed."""
    seen = []
    B, n_tok = 1, 8
    sample_ordered(_recording_denoise(seen), sigmas=_sig(n=8),
                   x_init=torch.zeros(B, n_tok * BPT),
                   u=ordering_ranks("l2r", n_tok, B), w=1.0, bits_per_token=BPT)
    spreads = [float(s.max() / s.min()) for s in seen]
    assert max(spreads) > 2.0, f"sigma barely varies across positions: {spreads}"


def test_l2r_denoises_earlier_tokens_first():
    """Left-to-right must mean earlier tokens reach LOW sigma sooner."""
    seen = []
    B, n_tok = 1, 8
    sample_ordered(_recording_denoise(seen), sigmas=_sig(n=8),
                   x_init=torch.zeros(B, n_tok * BPT),
                   u=ordering_ranks("l2r", n_tok, B), w=1.0, bits_per_token=BPT)
    mid = seen[len(seen) // 2][0].view(n_tok, BPT)[:, 0]
    assert mid[0] < mid[-1], f"token 0 should be cleaner than token n-1: {mid}"


def test_r2l_is_the_mirror_of_l2r():
    def first_last(mode):
        seen = []
        sample_ordered(_recording_denoise(seen), sigmas=_sig(n=8),
                       x_init=torch.zeros(1, 8 * BPT),
                       u=ordering_ranks(mode, 8, 1), w=1.0, bits_per_token=BPT)
        m = seen[len(seen) // 2][0].view(8, BPT)[:, 0]
        return float(m[0]), float(m[-1])
    a0, a1 = first_last("l2r")
    b0, b1 = first_last("r2l")
    assert (a0 < a1) and (b0 > b1)


def test_larger_w_spreads_sigma_more():
    def spread(w):
        seen = []
        sample_ordered(_recording_denoise(seen), sigmas=_sig(n=8),
                       x_init=torch.zeros(1, 8 * BPT),
                       u=ordering_ranks("l2r", 8, 1), w=w, bits_per_token=BPT)
        return max(float(s.max() / s.min()) for s in seen)
    assert spread(0.25) < spread(1.0) < spread(2.0)


# ------------------------------------------------------------------- ranks
def test_ranks_cover_the_suffix_only_and_prompt_is_excluded():
    B, n_tok = 2, 10
    suffix = torch.zeros(B, n_tok, dtype=torch.bool)
    suffix[:, 4:] = True                       # first 4 tokens are prompt
    u = ordering_ranks("l2r", n_tok, B, suffix_mask=suffix)
    assert torch.all(u[:, :4] == 0)
    su = u[0, 4:]
    assert su.min() == pytest.approx(0.0) and su.max() == pytest.approx(1.0)
    # u is denoising PRIORITY: l2r gives the first suffix token the highest.
    assert su[0] == pytest.approx(1.0) and su[-1] == pytest.approx(0.0)
    assert torch.all(su[1:] <= su[:-1]), "l2r priority must be non-increasing"


def test_prompt_length_does_not_compress_the_suffix_ordering():
    """Ranking over the whole sequence would let a long prompt eat the ordering
    range and turn 'left-to-right' into 'prompt-first'."""
    B, n_tok = 1, 20
    for cut in (2, 10, 18):
        suffix = torch.zeros(B, n_tok, dtype=torch.bool)
        suffix[:, cut:] = True
        u = ordering_ranks("l2r", n_tok, B, suffix_mask=suffix)
        su = u[0, cut:]
        assert su.min() == pytest.approx(0.0)
        assert su.max() == pytest.approx(1.0)


def test_random_ranks_are_per_example_permutations():
    B, n_tok = 4, 12
    u = ordering_ranks("random", n_tok, B, generator=torch.Generator().manual_seed(3))
    for b in range(B):
        assert torch.allclose(torch.sort(u[b]).values, torch.sort(u[0]).values)
    assert not torch.equal(u[0], u[1]), "ranks must be resampled per example"


def test_random_ranks_are_not_accidentally_fixed_across_calls():
    a = ordering_ranks("random", 12, 4, generator=torch.Generator().manual_seed(3))
    b = ordering_ranks("random", 12, 4, generator=torch.Generator().manual_seed(4))
    assert not torch.equal(a, b)


def test_random_ranks_are_seed_reproducible():
    a = ordering_ranks("random", 12, 4, generator=torch.Generator().manual_seed(11))
    b = ordering_ranks("random", 12, 4, generator=torch.Generator().manual_seed(11))
    assert torch.equal(a, b)


# ------------------------------------------------------------------ mechanics
def test_positional_time_matches_the_documented_formula():
    t = torch.tensor([0.4])
    u = torch.tensor([[0.0, 0.5, 1.0]])
    w = 0.8
    got = positional_time(t, u, w)
    want = torch.clamp(0.4 * (1 + w) - w * u, 0.0, 1.0)
    assert torch.allclose(got, want)


def test_positional_time_is_clipped_into_the_unit_interval():
    out = positional_time(torch.tensor([0.9]), torch.tensor([[0.0, 1.0]]), 3.0)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_sigma_expansion_is_blockwise_per_token():
    sig = torch.tensor([[1.0, 2.0, 3.0]])
    out = expand_token_sigma_to_bits(sig, 4)
    assert out.shape == (1, 12)
    assert torch.equal(out[0], torch.tensor([1.] * 4 + [2.] * 4 + [3.] * 4))


def test_token_prefix_mask_requires_all_bits_of_a_token():
    pm = torch.zeros(1, 12, dtype=torch.bool)
    pm[0, :4] = True          # token 0 fully prompt
    pm[0, 4:6] = True         # token 1 only partly
    got = token_prefix_mask_from_bits(pm, 4)
    assert got.tolist() == [[True, False, False]]


def test_prompt_positions_survive_the_whole_trajectory():
    B, n_tok = 2, 6
    S = n_tok * BPT
    pm = torch.zeros(B, S, dtype=torch.bool)
    pm[:, :2 * BPT] = True
    pf = torch.zeros(B, S)
    pf[pm] = 1.0
    out = sample_ordered(lambda x, s: torch.sigmoid(x), sigmas=_sig(),
                         x_init=torch.randn(B, S), u=ordering_ranks("l2r", n_tok, B),
                         w=1.5, bits_per_token=BPT, prefix_full=pf, prefix_mask=pm)
    assert torch.all(out[pm] == 1.0)


def test_prompt_positions_keep_the_global_sigma_under_ordering():
    """The ordering must act on the generated suffix only. If prompt positions
    inherited an ordered sigma, the model would be told its clean, clamped
    prompt is noisy -- a second intervention on top of the one under test, and
    a train/test mismatch the control does not have."""
    seen = []
    B, n_tok = 1, 8
    S = n_tok * BPT
    pm = torch.zeros(B, S, dtype=torch.bool)
    pm[:, :3 * BPT] = True
    pf = torch.zeros(B, S)
    suffix = ~token_prefix_mask_from_bits(pm, BPT)
    sample_ordered(_recording_denoise(seen), sigmas=_sig(n=8), x_init=torch.randn(B, S),
                   u=ordering_ranks("l2r", n_tok, B, suffix_mask=suffix), w=1.5,
                   bits_per_token=BPT, prefix_full=pf, prefix_mask=pm)
    for s in seen:
        prompt_sig = s[0, :3 * BPT]
        assert torch.allclose(prompt_sig, prompt_sig[0].expand_as(prompt_sig)), \
            "prompt sigma must be uniform"
    # and it must equal the w=0 (global) sigma at every step
    seen0 = []
    sample_ordered(_recording_denoise(seen0), sigmas=_sig(n=8), x_init=torch.randn(B, S),
                   u=ordering_ranks("l2r", n_tok, B, suffix_mask=suffix), w=0.0,
                   bits_per_token=BPT, prefix_full=pf, prefix_mask=pm)
    for a, b in zip(seen, seen0):
        assert torch.allclose(a[0, :3 * BPT], b[0, :3 * BPT], atol=1e-6)


def test_suffix_sigma_still_varies_when_a_prompt_is_present():
    """Guard against the prompt pin accidentally flattening the whole sequence."""
    seen = []
    B, n_tok = 1, 10
    S = n_tok * BPT
    pm = torch.zeros(B, S, dtype=torch.bool)
    pm[:, :3 * BPT] = True
    suffix = ~token_prefix_mask_from_bits(pm, BPT)
    sample_ordered(_recording_denoise(seen), sigmas=_sig(n=8), x_init=torch.randn(B, S),
                   u=ordering_ranks("l2r", n_tok, B, suffix_mask=suffix), w=1.0,
                   bits_per_token=BPT, prefix_full=torch.zeros(B, S), prefix_mask=pm)
    spreads = [float(s[0, 3 * BPT:].max() / s[0, 3 * BPT:].min()) for s in seen]
    assert max(spreads) > 2.0, f"suffix sigma should still spread: {spreads}"


# --- the guidance guard must fire on ACTIVE guidance, not on its mere presence
class _FakeG:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _mk_sampler():
    """A sampler whose guard we can exercise without loading a real model."""
    from diffusion.continuous.ordered_sampler import OrderedSampler
    s = OrderedSampler.__new__(OrderedSampler)
    s.order_w, s.order_mode, s.order_seed = 0.0, "l2r", None
    return s


@pytest.mark.parametrize("guidance", [
    None,
    {"cfg_scale": 0.0, "ag_scale": 0.0, "sg_scale": 0.0,
     "sg_variant": None, "sg_delta": None, "sg_mf_mode": None},
    _FakeG(cfg_scale=0.0, ag_scale=0.0, sg_scale=0.0),
])
def test_inactive_guidance_is_accepted(guidance):
    """The task evals ALWAYS pass a guidance config with zero scales. Rejecting
    on its presence made every ordering cell fail with a NotImplementedError."""
    s = _mk_sampler()
    with pytest.raises(AttributeError):   # gets past the guard, dies later on the stub
        s.sample(1, 16, guidance=guidance, guidance_scale=0.0)


@pytest.mark.parametrize("kw", [
    {"guidance_scale": 3.0},
    {"guidance": {"cfg_scale": 2.0}},
    {"guidance": _FakeG(ag_scale=1.5)},
    {"guidance": _FakeG(sg_scale=0.5)},
    {"bad_model": object()},
])
def test_active_guidance_is_refused(kw):
    s = _mk_sampler()
    with pytest.raises(NotImplementedError, match="guidance"):
        s.sample(1, 16, **kw)


def test_returned_probs_come_from_the_FINAL_sigma_not_the_second_to_last():
    """DDIMSampler does an extra denoise at the smallest sigma and returns that
    D. Returning the trajectory's last in-loop D instead decodes bits from a far
    noisier level -- which scored 0.0000 on GSM8K against the baseline's 0.164.
    """
    from diffusion.continuous.ordered_sampler import OrderedSampler
    seen = []

    class _Stub(OrderedSampler):
        def __init__(self):
            self.order_w, self.order_mode, self.order_seed = 0.0, "none", None
            self.bits_per_token, self.data_center = BPT, 0.5
            self.sc_enabled, self.is_cont_tokens = False, False
            self.device = torch.device("cpu")
            self.cfg = None
            self.model = None
            sig = _sig(n=5)

            class _S:
                def prepare(self_inner, **kw):
                    return sig
            self.sigmas = _S()
            self._sig = sig

    s = _Stub()
    import diffusion.continuous.ordered_sampler as mod
    orig = mod._model_logits_continuous
    mod._model_logits_continuous = lambda m, c, x, sg, xh: (seen.append(sg.clone()) or x * 0.0 + sg.mean())
    try:
        _, probs = s.sample(1, 4 * BPT, num_steps=4, return_probs=True)
    finally:
        mod._model_logits_continuous = orig
    assert torch.allclose(seen[-1], torch.full_like(seen[-1], float(s._sig[-1]))), \
        f"final denoise must use sigmas[-1]={float(s._sig[-1])}, got {seen[-1].flatten()[0]}"
    assert len(seen) == 5, f"expected num_steps + 1 = 5 forwards, got {len(seen)}"


def test_random_ranks_accept_a_cpu_generator_regardless_of_target_device():
    """A CPU generator with a CUDA target raised 'Expected a cuda device type
    for generator', which killed both random cells of the first screen."""
    sm = torch.zeros(2, 8, dtype=torch.bool)
    sm[:, 2:] = True
    u = ordering_ranks("random", 8, 2, suffix_mask=sm,
                       generator=torch.Generator().manual_seed(3))
    assert u.shape == (2, 8) and u.device.type == sm.device.type


def test_random_ranks_are_device_independent_for_a_given_seed():
    a = ordering_ranks("random", 16, 3, generator=torch.Generator().manual_seed(9))
    b = ordering_ranks("random", 16, 3, generator=torch.Generator().manual_seed(9),
                       device="cpu")
    assert torch.equal(a, b)

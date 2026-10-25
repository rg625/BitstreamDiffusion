"""Training-time temporal ordering.

The claim this must support is "the model was TRAINED to denoise in an order",
so the tests check what training actually sees, not just that a helper returns
plausible numbers.
"""
import math

import ml_collections
import pytest
import torch

from diffusion.continuous.ordering import (
    sigma_to_time, time_to_sigma, training_position_sigma, ordering_ranks,
)

SMIN, SMAX, BPT = 0.002, 80.0, 16


def _pm(B, n_tok, n_prompt):
    m = torch.zeros(B, n_tok * BPT, dtype=torch.bool)
    m[:, : n_prompt * BPT] = True
    return m


def _tps(sig, mode, w, B=2, n_tok=8, n_prompt=2, gen=None):
    return training_position_sigma(
        sig, n_bits=n_tok * BPT, bits_per_token=BPT, mode=mode, w=w,
        sigma_min=SMIN, sigma_max=SMAX, prefix_mask=_pm(B, n_tok, n_prompt),
        generator=gen)


# --------------------------------------------------------------- equivalence
def test_w0_returns_the_global_sigma_at_every_position():
    sig = torch.tensor([0.05, 3.0])
    out = _tps(sig, "l2r", 0.0)
    assert torch.allclose(out, sig.view(-1, 1).expand_as(out))


def test_sigma_time_roundtrip_is_the_identity():
    s = torch.tensor([0.002, 0.05, 0.4, 3.0, 80.0])
    back = time_to_sigma(sigma_to_time(s, SMIN, SMAX), SMIN, SMAX)
    assert torch.allclose(back, s, rtol=1e-5)


# ------------------------------------------------- the intervention is active
@pytest.mark.parametrize("mode", ["l2r", "r2l", "random"])
def test_training_sees_non_uniform_sigma_for_every_mode(mode):
    """The point of the experiment: TRAINING must see different sigma at
    different suffix positions, for each ordering mode."""
    out = _tps(torch.tensor([0.4, 0.4]), mode, 0.5,
               gen=torch.Generator().manual_seed(0))
    suf = out[0, 2 * BPT:]
    assert float(suf.max() / suf.min()) > 1.5, f"{mode}: sigma barely varies"


def test_the_three_modes_produce_different_sigma_fields():
    """l2r, r2l and random must not collapse to the same schedule -- otherwise
    the three arms would be the same experiment run three times."""
    sig = torch.tensor([0.4, 0.4])
    g = torch.Generator().manual_seed(0)
    a = _tps(sig, "l2r", 0.5)
    b = _tps(sig, "r2l", 0.5)
    c = _tps(sig, "random", 0.5, gen=g)
    assert not torch.allclose(a, b)
    assert not torch.allclose(a, c)
    assert not torch.allclose(b, c)


def test_l2r_and_r2l_are_mirror_images_in_sigma():
    sig = torch.tensor([0.4])
    a = _tps(sig, "l2r", 0.5, B=1)[0, 2 * BPT:].view(-1, BPT)[:, 0]
    b = _tps(sig, "r2l", 0.5, B=1)[0, 2 * BPT:].view(-1, BPT)[:, 0]
    assert a[0] < a[-1], "l2r: first suffix token must be cleanest"
    assert b[0] > b[-1], "r2l: last suffix token must be cleanest"
    assert torch.allclose(a, b.flip(0), rtol=1e-4)


# ------------------------------------------------------------------- suffix
def test_prompt_positions_keep_the_global_sigma():
    sig = torch.tensor([0.4, 1.3])
    out = _tps(sig, "l2r", 0.7, n_prompt=3)
    pr = out[:, : 3 * BPT]
    assert torch.allclose(pr, sig.view(-1, 1).expand_as(pr))


def test_prompt_length_does_not_shift_the_suffix_schedule():
    """Ranking over the whole sequence would let a long prompt eat the ordering
    range and turn l2r into prompt-first."""
    sig = torch.tensor([0.4])
    for np_ in (1, 3, 6):
        out = _tps(sig, "l2r", 0.5, B=1, n_tok=8, n_prompt=np_)
        suf = out[0, np_ * BPT:].view(-1, BPT)[:, 0]
        assert suf[0] < suf[-1]


def test_sigma_is_constant_within_each_token():
    out = _tps(torch.tensor([0.4]), "l2r", 0.6, B=1)
    blocks = out.view(1, -1, BPT)
    assert torch.allclose(blocks, blocks[..., :1].expand_as(blocks))


# ------------------------------------------------------------------ random
def test_random_is_resampled_per_call_and_per_example():
    sig = torch.tensor([0.4, 0.4, 0.4])
    g = torch.Generator().manual_seed(1)
    a = _tps(sig, "random", 0.5, B=3, gen=g)
    b = _tps(sig, "random", 0.5, B=3, gen=g)
    assert not torch.allclose(a, b), "must be redrawn per step"
    assert not torch.allclose(a[0], a[1]), "must be redrawn per example"


def test_random_is_seed_reproducible():
    sig = torch.tensor([0.4, 0.4])
    a = _tps(sig, "random", 0.5, gen=torch.Generator().manual_seed(7))
    b = _tps(sig, "random", 0.5, gen=torch.Generator().manual_seed(7))
    assert torch.allclose(a, b)


def test_sigma_stays_inside_the_trained_range():
    for w in (0.1, 0.25, 0.5):
        for s in (0.002, 0.4, 80.0):
            out = _tps(torch.tensor([s]), "l2r", w, B=1)
            assert float(out.min()) >= SMIN * 0.999
            assert float(out.max()) <= SMAX * 1.001


# --------------------------------------------------- trainer-level behaviour
class _T:
    from trainers.trainer import Trainer as _Tr
    _apply_training_ordering = _Tr._apply_training_ordering

    def __init__(self, enabled, mode="l2r", w=0.5):
        c = ml_collections.ConfigDict()
        c.train = ml_collections.ConfigDict()
        if enabled:
            o = ml_collections.ConfigDict()
            o.enabled, o.mode, o.w = True, mode, w
            c.train.ordering = o
        c.data = ml_collections.ConfigDict(); c.data.bits_per_token = BPT
        c.diffusion = ml_collections.ConfigDict()
        c.diffusion.continuous = ml_collections.ConfigDict()
        c.diffusion.continuous.sigma_min = SMIN
        c.diffusion.continuous.sigma_max = SMAX
        self.cfg = c


def test_trainer_returns_scalar_sigma_untouched_when_disabled():
    sig = torch.tensor([0.4, 1.0])
    out = _T(False)._apply_training_ordering(sig, 8 * BPT, None, False)
    assert out is sig, "disabled ordering must be a no-op, not a reshape"


def test_trainer_returns_scalar_sigma_untouched_at_w0():
    sig = torch.tensor([0.4, 1.0])
    out = _T(True, w=0.0)._apply_training_ordering(sig, 8 * BPT, None, False)
    assert out is sig


def test_trainer_expands_to_per_position_when_enabled():
    sig = torch.tensor([0.4, 1.0])
    out = _T(True, "l2r", 0.5)._apply_training_ordering(
        sig, 8 * BPT, _pm(2, 8, 2), False)
    assert out.shape == (2, 8 * BPT)
    assert float(out[0, 2 * BPT:].max() / out[0, 2 * BPT:].min()) > 1.5


def test_trainer_refuses_token_representation():
    with pytest.raises(NotImplementedError, match="binary bitstreams only"):
        _T(True)._apply_training_ordering(torch.tensor([0.4]), 8 * BPT, None, True)

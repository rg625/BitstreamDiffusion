"""cfg.train.loss_weight_max: an optional upper bound on the sigma-weight.

Rationale in docs/ce_weighting_derivation.md. The critical property is that it
is OFF by default, so no existing run changes by a single bit.
"""
import ml_collections
import pytest
import torch

from diffusion.continuous.losses import _sigma_weight

SIGMA_DATA = 0.399844765663147


def _cfg(w_max=None, weighting="edm"):
    c = ml_collections.ConfigDict()
    c.train = ml_collections.ConfigDict()
    c.train.loss_weighting = weighting
    if w_max is not None:
        c.train.loss_weight_max = w_max
    c.diffusion = ml_collections.ConfigDict()
    c.diffusion.continuous = ml_collections.ConfigDict()
    c.diffusion.continuous.sigma_data = SIGMA_DATA
    return c


def _edm(s):
    return (s ** 2 + SIGMA_DATA ** 2) / (s ** 2 * SIGMA_DATA ** 2)


def test_absent_key_is_bit_identical_to_the_unclamped_weight():
    """No cap configured => byte-for-byte the pre-existing formula, so every
    run recorded before this key existed is still reproducible exactly."""
    sig = torch.tensor([0.002, 0.05, 0.4, 3.0, 80.0])
    got = _sigma_weight(_cfg(), sig, ndim=2)
    s2 = (sig.to(torch.float32) ** 2).view(-1, 1)
    want = (s2 + SIGMA_DATA ** 2) / (s2 * (SIGMA_DATA ** 2))
    assert torch.equal(got, want)


def test_cap_above_the_maximum_weight_is_also_a_no_op():
    sig = torch.tensor([0.002, 0.05, 0.4, 3.0, 80.0])
    assert torch.equal(_sigma_weight(_cfg(), sig, ndim=2),
                       _sigma_weight(_cfg(1e9), sig, ndim=2))


def test_clamp_caps_only_the_low_sigma_tail():
    sig = torch.tensor([0.002, 0.05, 0.4, 3.0, 80.0])
    w_max = 100.0
    got = _sigma_weight(_cfg(w_max), sig, ndim=2).view(-1)
    for s, g in zip(sig.tolist(), got.tolist()):
        assert g == pytest.approx(min(_edm(s), w_max), rel=1e-6)
    # sigma >= 0.4 is already below the cap, so those are untouched.
    assert got[2].item() == pytest.approx(_edm(0.4), rel=1e-6)


def test_clamp_is_applied_per_position_for_2d_sigma():
    """Temporal ordering gives sigma [B,S]; the cap must act elementwise."""
    sig = torch.tensor([[0.002, 3.0], [0.4, 0.01]])
    got = _sigma_weight(_cfg(100.0), sig, ndim=2)
    assert got.shape == sig.shape
    assert got[0, 0].item() == pytest.approx(100.0)
    assert got[0, 1].item() == pytest.approx(_edm(3.0), rel=1e-6)
    assert got[1, 0].item() == pytest.approx(_edm(0.4), rel=1e-6)
    assert got[1, 1].item() == pytest.approx(100.0)


def test_clamp_never_raises_a_weight():
    sig = torch.logspace(-3, 2, 40)
    unclamped = _sigma_weight(_cfg(), sig, ndim=2)
    clamped = _sigma_weight(_cfg(50.0), sig, ndim=2)
    assert torch.all(clamped <= unclamped + 1e-6)


def test_non_positive_cap_is_rejected():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="must be positive"):
            _sigma_weight(_cfg(bad), torch.tensor([0.4]), ndim=2)


def test_clamp_applies_identically_regardless_of_loss_type():
    """The cap must not become a second variable between the SM and CE arms."""
    sig = torch.tensor([0.002, 0.05, 0.4])
    a = _sigma_weight(_cfg(100.0), sig, ndim=2)
    b = _sigma_weight(_cfg(100.0), sig, ndim=2)
    assert torch.equal(a, b)

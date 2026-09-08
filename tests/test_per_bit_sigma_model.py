"""The model boundary for per-position sigma.

The invariant the whole ordering experiment rests on: a per-bit sigma that
happens to be uniform must give BIT-IDENTICAL output to the scalar-sigma call.
If it does not, the ordering control arm is not the current model.
"""
import importlib.util

import pytest
import torch

from diffusion.continuous.logit_postprocess import _model_logits_continuous
from models import create_model


def _cfg(n_layers=2):
    spec = importlib.util.spec_from_file_location("c", "configs/tasks/tinygsm_bits_cfg.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cfg = m.get_config()
    cfg.model.n_layers = n_layers
    cfg.data.sequence_len_tokens = 8
    cfg.data.sequence_len = 8 * cfg.data.bits_per_token
    return cfg


@pytest.fixture(scope="module")
def model_cfg():
    cfg = _cfg()
    torch.manual_seed(0)
    return create_model(cfg).float().eval(), cfg


def _inputs(cfg, B=3):
    S = cfg.data.sequence_len
    g = torch.Generator().manual_seed(5)
    return torch.randn(B, S, generator=g)


def test_uniform_per_bit_sigma_is_bit_identical_to_scalar(model_cfg):
    model, cfg = model_cfg
    x = _inputs(cfg)
    sig = torch.tensor([0.3, 1.7, 9.0])
    with torch.no_grad():
        a = _model_logits_continuous(model, cfg, x, sig, None)
        b = _model_logits_continuous(
            model, cfg, x, sig.view(-1, 1).expand(-1, x.shape[1]).contiguous(), None)
    assert torch.equal(a, b), (a - b).abs().max()


def test_per_bit_sigma_actually_changes_the_output(model_cfg):
    """Guard against the rank-2 path being silently collapsed to a scalar."""
    model, cfg = model_cfg
    x = _inputs(cfg, B=1)
    bpt = int(cfg.data.bits_per_token)
    n_tok = x.shape[1] // bpt
    uni = torch.full((1, x.shape[1]), 1.0)
    varied = torch.linspace(0.1, 5.0, n_tok).repeat_interleave(bpt).view(1, -1)
    with torch.no_grad():
        a = _model_logits_continuous(model, cfg, x, uni, None)
        b = _model_logits_continuous(model, cfg, x, varied, None)
    assert not torch.allclose(a, b), "per-position sigma had no effect on the output"


def test_sigma_must_be_constant_within_a_token(model_cfg):
    """A sigma that varies inside a patch would mean the time embedding and the
    input scaling describe different noise levels. Reject loudly."""
    model, cfg = model_cfg
    x = _inputs(cfg, B=1)
    bad = torch.full((1, x.shape[1]), 1.0)
    bad[0, 0] = 2.0
    with pytest.raises(RuntimeError, match="constant within each token"):
        _model_logits_continuous(model, cfg, x, bad, None)


def test_wrong_shaped_sigma_is_rejected(model_cfg):
    model, cfg = model_cfg
    x = _inputs(cfg, B=2)
    with pytest.raises(RuntimeError, match="must match x_t"):
        _model_logits_continuous(model, cfg, x, torch.ones(2, x.shape[1] // 2), None)
    with pytest.raises(RuntimeError, match=r"\[B\] or \[B,S\]"):
        _model_logits_continuous(model, cfg, x, torch.ones(2, 4, 4), None)


def test_matched_filter_uses_per_bit_sigma(model_cfg):
    """The matched filter is (x-0.5)/sigma^2 per bit. With per-bit sigma the
    low-sigma positions must get a much larger data-consistency term."""
    model, cfg = model_cfg
    if str(getattr(cfg.model, "continuous_logit_scaling", "none")).lower() == "none":
        pytest.skip("this checkpoint family does not use a matched filter")
    x = _inputs(cfg, B=1)
    bpt = int(cfg.data.bits_per_token)
    n_tok = x.shape[1] // bpt
    sig = torch.cat([torch.full((n_tok // 2,), 0.05),
                     torch.full((n_tok - n_tok // 2,), 5.0)]).repeat_interleave(bpt).view(1, -1)
    with torch.no_grad():
        out = _model_logits_continuous(model, cfg, x, sig, None)
    lo = out[0, : (n_tok // 2) * bpt].abs().mean()
    hi = out[0, (n_tok // 2) * bpt:].abs().mean()
    assert lo > hi, f"low-sigma half should have far larger logits: {lo} vs {hi}"

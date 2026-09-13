"""Token-space (V-way softmax) TinyGSM configuration.

This IS a different objective from BCE, unlike a 2-class softmax per bit:
16 independent Bernoullis can only express a product distribution over a
token's bits, while a V-way categorical can express any distribution over
tokens. These tests pin the wiring that makes that true.
"""
import importlib.util
import os

import pytest
import torch


def _cfg(loss="token_ce"):
    os.environ["TOK_LOSS"] = loss
    s = importlib.util.spec_from_file_location("tok", "configs/tasks/tinygsm_tokens.py")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m.get_config()


def test_token_config_is_a_V_way_head_over_token_positions():
    c = _cfg()
    assert c.data.representation == "tokens"
    assert int(c.model.out_dim) == int(c.data.vocab_size) == 49153
    # one transformer position per TOKEN, not per bit
    assert int(c.model.patch_size) == 1
    assert int(c.data.sequence_len) == int(c.data.sequence_len_tokens) == 512


def test_data_centre_is_the_uniform_simplex_not_the_bit_centre():
    """A one-hot token's mean is 1/V. Leaving it at the bits value (0.5) would
    centre the diffusion on a point nowhere near the data manifold."""
    c = _cfg()
    assert c.diffusion.continuous.data_center == pytest.approx(1.0 / 49153)


def test_production_recipe_is_otherwise_preserved():
    c = _cfg()
    assert float(c.cond.p_uncond) == 0.1
    assert str(c.train.loss_weighting) == "edm"
    assert not hasattr(c.train, "loss_weight_max") or c.train.loss_weight_max is None


def test_both_token_losses_are_selectable_and_bad_ones_rejected():
    assert _cfg("token_ce").train.loss_type == "token_ce"
    assert _cfg("token_sm").train.loss_type == "token_sm"
    with pytest.raises(SystemExit):
        _cfg("binary_ce")


def test_dataset_returns_token_ids_and_a_token_level_prefix_mask():
    """The trainer's token path indexes x0 as class indices; handing it bits
    would silently train on the wrong target space."""
    from data.tinygsm import TinyGSMDataset
    c = _cfg()
    ds = TinyGSMDataset(c, split="val")
    r = ds[0]
    assert r["x0"].dtype == torch.int64
    assert r["x0"].shape == (512,)
    assert r["prefix_mask"].shape == (512,)
    assert r["prefix_mask"].dtype == torch.bool
    assert int(r["x0"].max()) < 49153


def test_binary_path_is_unchanged_by_the_token_addition():
    import importlib.util as iu
    s = iu.spec_from_file_location("bits", "configs/tasks/tinygsm_bits_cfg.py")
    m = iu.module_from_spec(s); s.loader.exec_module(m)
    cb = m.get_config()
    from data.tinygsm import TinyGSMDataset
    r = TinyGSMDataset(cb, split="val")[0]
    assert r["x0"].shape == (8192,)          # bits, as before
    assert r["prefix_mask"].shape == (8192,)


def test_expressivity_claim_is_arithmetically_what_we_say_it_is():
    """16 independent Bernoullis: 16 dof. V-way categorical: V-1 dof."""
    bits_dof = 16
    cat_dof = 49153 - 1
    assert cat_dof > bits_dof * 3000


def test_token_prefix_masking_handles_the_V_axis():
    """A [B,S,1] boolean mask cannot index a [B,S,V] tensor -- it raises
    IndexError, which is exactly what killed the first token smoke run."""
    B, S, V = 2, 4, 7
    xt = torch.zeros(B, S, V)
    src = torch.arange(B * S * V, dtype=torch.float32).reshape(B, S, V)
    prefix_mask = torch.zeros(B, S, dtype=torch.bool)
    prefix_mask[:, :2] = True
    pm = prefix_mask.unsqueeze(-1).expand_as(xt)
    xt[pm] = src[pm]
    assert torch.equal(xt[:, :2, :], src[:, :2, :])
    assert torch.all(xt[:, 2:, :] == 0)
    with pytest.raises(IndexError):
        torch.zeros(B, S, V)[prefix_mask.unsqueeze(-1)] = 1.0

"""Gradient-survival probe: the PRIMARY endpoint of the binary_sm/CE pilot.

grad_survival = sum|dL_sm/d_ell| / sum|dL_ce/d_ell| = sum|(D-x0)*D(1-D)| / sum|D-x0|

It is a property of the model's D values, not of the loss being optimised, so it
is computed identically in both arms and the two runs are directly comparable.
"""
import torch

from trainers.trainer import Trainer


class _Probe:
    """Exercise the probe body without constructing a full Trainer."""
    def __init__(self, cfg, calls):
        self.cfg, self.global_step, self.is_master = cfg, 0, True
        self.writer = type("W", (), {"add_scalar": lambda s, k, v, st: calls.append((k, v))})()
    _log_objective_probe = Trainer._log_objective_probe


def _cfg(every=1, enabled=True):
    c = type("C", (), {})()
    c.train = type("T", (), {})()
    c.train.objective_probe = type("P", (), {})()
    c.train.objective_probe.enabled = enabled
    c.train.objective_probe.every_steps = every
    return c


def _run(logits, x0, sigma, mask, **kw):
    calls = []
    p = _Probe(_cfg(**kw), calls)
    p._log_objective_probe(logits, x0, sigma, mask)
    return dict(calls)


def test_saturated_bits_give_near_zero_survival():
    """The whole hypothesis: saturated bits annihilate the SM gradient."""
    logits = torch.full((2, 8), 12.0)          # D ~ 1 -> D(1-D) ~ 6e-6
    x0 = torch.zeros(2, 8)                      # and all of them are WRONG
    out = _run(logits, x0, torch.ones(2), None)
    assert out["objective/grad_survival"] < 1e-4
    assert out["objective/frac_D1mD_lt_0.01"] == 1.0


def test_unsaturated_bits_retain_signal():
    logits = torch.zeros(2, 8)                  # D = 0.5 -> D(1-D) = 0.25, the max
    x0 = torch.zeros(2, 8)
    out = _run(logits, x0, torch.ones(2), None)
    assert abs(out["objective/grad_survival"] - 0.25) < 1e-5
    assert abs(out["objective/median_D1mD"] - 0.25) < 1e-6


def test_probe_honours_the_free_bit_mask():
    """Prompt bits are clamped and carry no gradient; including them would
    dilute every statistic."""
    logits = torch.cat([torch.full((1, 4), 12.0), torch.zeros(1, 4)], dim=1)
    x0 = torch.zeros(1, 8)
    mask = torch.cat([torch.zeros(1, 4), torch.ones(1, 4)], dim=1)   # only the D=0.5 half
    out = _run(logits, x0, torch.ones(1), mask)
    assert abs(out["objective/grad_survival"] - 0.25) < 1e-5


def test_probe_is_disabled_by_default_and_by_cadence():
    logits, x0 = torch.zeros(1, 4), torch.zeros(1, 4)
    assert _run(logits, x0, torch.ones(1), None, enabled=False) == {}
    calls = []
    p = _Probe(_cfg(every=500), calls); p.global_step = 7    # not a multiple of 500
    p._log_objective_probe(logits, x0, torch.ones(1), None)
    assert calls == []


def test_sigma_stratification_is_emitted():
    logits = torch.zeros(4, 8); x0 = torch.zeros(4, 8)
    sigma = torch.tensor([0.1, 1.0, 10.0, 0.2])
    out = _run(logits, x0, sigma, None)
    for k in ("lo", "mid", "hi"):
        assert f"objective/grad_survival_{k}" in out, k

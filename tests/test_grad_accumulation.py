"""Gradient accumulation: does an accumulated step equal the big batch it stands in for?

The token-space V-way model OOMs at 128 examples/GPU, so the first 500k run was
forced down to an effective batch of 128 against a binary control at the same
128 -- matched, but a quarter of production's 512 and scoring 0.0255 against
0.0493. Accumulation restores the 512 on the same four GPUs. That is only worth
anything if an accumulated step really is the big step, so that is what these
tests check, numerically rather than structurally.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from trainers.trainer import Trainer, _micro_batch_size


# ---------------------------------------------------------------- arithmetic
def test_micro_batch_splits_the_effective_batch():
    # The configuration this project will actually run: 512 on 4 GPUs, 4 accum.
    assert _micro_batch_size(512, 4, 4) == 32
    # ... which is exactly the per-GPU load the 128-batch run already proved fits.
    assert _micro_batch_size(128, 4, 1) == 32


def test_micro_batch_is_unchanged_when_accum_is_one():
    for eff, ws in [(512, 4), (128, 4), (256, 8), (64, 1)]:
        assert _micro_batch_size(eff, ws, 1) == eff // ws


@pytest.mark.parametrize("eff,ws,accum", [(512, 4, 3), (100, 8, 1), (512, 5, 4)])
def test_indivisible_batch_raises_rather_than_rounding(eff, ws, accum):
    """A silently rounded batch makes an arm quietly incomparable."""
    with pytest.raises(ValueError, match="divisible"):
        _micro_batch_size(eff, ws, accum)


# ---------------------------------------------------------------- numerics
def _toy():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))


def _grads(model):
    return [p.grad.detach().clone() for p in model.parameters()]


@pytest.mark.parametrize("accum", [2, 4, 8])
def test_accumulated_gradient_equals_the_full_batch_gradient(accum):
    """The claim the experiment rests on: 4 x 32 is 128, not 32.

    This mirrors the trainer's contract exactly -- zero_grad on the first
    micro-batch, loss scaled by 1/accum before every backward, optimiser step on
    the last -- against a single backward over the whole batch.
    """
    torch.manual_seed(1)
    x = torch.randn(accum * 16, 8)
    y = torch.randn(accum * 16, 1)
    lossfn = torch.nn.MSELoss()

    full = _toy()
    full.zero_grad(set_to_none=True)
    lossfn(full(x), y).backward()
    expected = _grads(full)

    acc = _toy()
    acc.zero_grad(set_to_none=True)
    micro = x.shape[0] // accum
    for i in range(accum):
        sl = slice(i * micro, (i + 1) * micro)
        if i == 0:
            acc.zero_grad(set_to_none=True)
        (lossfn(acc(x[sl]), y[sl]) * (1.0 / accum)).backward()
    got = _grads(acc)

    for g, e in zip(got, expected):
        assert torch.allclose(g, e, atol=1e-6), (g - e).abs().max()


def test_unscaled_accumulation_would_be_wrong():
    """Guards the 1/accum factor itself: without it the gradient is accum times too big."""
    torch.manual_seed(1)
    accum = 4
    x, y = torch.randn(64, 8), torch.randn(64, 1)
    lossfn = torch.nn.MSELoss()

    full = _toy(); full.zero_grad(set_to_none=True)
    lossfn(full(x), y).backward()
    expected = _grads(full)

    bad = _toy(); bad.zero_grad(set_to_none=True)
    for i in range(accum):
        sl = slice(i * 16, (i + 1) * 16)
        lossfn(bad(x[sl]), y[sl]).backward()   # no 1/accum
    got = _grads(bad)

    assert not torch.allclose(got[0], expected[0], atol=1e-6)
    assert torch.allclose(got[0], expected[0] * accum, atol=1e-5)


# ---------------------------------------------------------------- wiring
def test_step_defaults_reproduce_the_original_single_batch_path():
    """Every existing run must be untouched: the defaults are a plain step."""
    for fn in (Trainer._step_continuous, Trainer._step_discrete):
        params = inspect.signature(fn).parameters
        assert params["accum_first"].default is True
        assert params["accum_last"].default is True
        assert params["accum_scale"].default == 1.0


def test_discrete_framework_rejects_accumulation():
    """Refuse rather than silently train at 1/accum of the intended batch."""
    with pytest.raises(NotImplementedError, match="grad_accum_steps"):
        Trainer._step_discrete(object(), x0=None, is_train=True, accum_last=False)


def test_optimiser_work_is_gated_on_the_last_micro_batch():
    """zero_grad on the first, step/schedule/EMA on the last -- never per micro-batch."""
    src = inspect.getsource(Trainer._step_continuous)
    body = src.split("Optim step")[1]
    assert "if accum_first:\n                self.opt.zero_grad" in body
    for once_per_step in ("self.opt.step()", "self.lr_sched.step()", "self.ema.update"):
        idx = body.index(once_per_step)
        assert "if accum_last:" in body[:idx], f"{once_per_step} is not gated on accum_last"


def test_global_step_does_not_move_on_a_partial_batch():
    """global_step must stay an OPTIMISER-step count, or 500k means four different
    things across arms and the comparison to the binary control dissolves."""
    src = inspect.getsource(Trainer.train)
    head, tail = src.split("if not is_last_micro:")[0], src.split("if not is_last_micro:")[1]
    assert "continue" in tail.split("\n")[1]
    assert "self.global_step += 1" not in head


# ---------------------------------------------------------------- launcher
import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
CHAIN = REPO / "scripts" / "hpc" / "arch" / "token_chain.sh"


def test_chain_script_sets_its_own_wall_limit():
    """token_train.slurm's header says 8 h; 500k steps at the smoke-measured
    ~0.5 steps/s needs ~275 h, so the chain must ask for the 36 h QOS cap
    explicitly or it silently runs four times as many links as planned."""
    subprocess.run(["bash", "-n", str(CHAIN)], check=True)
    text = CHAIN.read_text()
    assert 'WALL="${WALL:-36:00:00}"' in text
    assert text.count('--time="$WALL"') == 2, "both the first link and the dependents need it"


def test_chain_refuses_to_submit_on_a_full_disk():
    """A disk-full event once killed four runs mid-checkpoint."""
    text = CHAIN.read_text()
    assert "REFUSING" in text and "AVAIL_GB" in text


def test_chain_treats_a_failed_sbatch_as_unknown():
    """rc=124 once meant 'no reply' while the job had in fact queued, and two
    jobs then wrote to the same run directory. Never retry blindly."""
    text = CHAIN.read_text()
    assert "STOPPING" in text
    assert "squeue" in text

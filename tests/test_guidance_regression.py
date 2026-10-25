"""Phase-5 regression: the refactored sampler must reproduce the old one exactly.

The guidance refactor moved all CFG algebra out of `DDIMSampler.sample` and into
`diffusion.continuous.guidance`. That is only safe if, for every setting the old
code supported, the new code returns the *same numbers*. These tests drive the
pinned pre-refactor sampler (see `_legacy_reference`) and the current one over a
matched grid -- same tiny denoiser, seed, sigma schedule, sampler, temperature,
conditioning and self-conditioning mode -- and compare bit-for-bit.

Bit-exactness (not `allclose`) is the right bar here: the refactor is meant to
reorder nothing, so any drift at all indicates a real semantic change.
"""
from __future__ import annotations

import itertools

import pytest
import torch

from diffusion.continuous.processes import ContinuousForwardProcess
from diffusion.continuous.samplers import DDIMSampler
from tests._legacy_reference import load_legacy_samplers
from tests._sampler_harness import TinyBinaryDenoiser, make_conditioning, make_cpu_cfg

B, S, NPROMPT, STEPS = 3, 64, 16, 6


def _build(self_condition: bool):
    cfg = make_cpu_cfg(self_condition=self_condition, num_steps=STEPS)
    model = TinyBinaryDenoiser(S, seed=0).eval()
    proc = ContinuousForwardProcess(cfg)
    return cfg, model, proc


def _run(sampler_cls, cfg, model, proc, *, seed, cond, **kw):
    torch.manual_seed(seed)
    sampler = sampler_cls(model, proc, cfg)
    pf, pm = (cond if cond is not None else (None, None))
    return sampler.sample(
        num_samples=B, seq_len=S,
        conditioning_prefix_full=pf, cond_prefix_mask=pm,
        num_steps=STEPS, schedule="karras",
        return_probs=True, progress=False,
        **kw,
    )


@pytest.mark.parametrize(
    "guidance_scale,self_condition,sc_refresh_mode",
    list(itertools.product([None, 0.0, 1.0, 2.0, 5.0], [False, True], ["refined", "carry"])),
)
def test_ddim_matches_pre_refactor(guidance_scale, self_condition, sc_refresh_mode):
    """Conditional sampling, across the CFG weights and SC modes the old code had."""
    cfg, model, proc = _build(self_condition)
    legacy = load_legacy_samplers()
    cond = make_conditioning(B, S, NPROMPT, seed=1)

    kw = dict(guidance_scale=guidance_scale, sc_refresh_mode=sc_refresh_mode, ati_eta=0.0)
    x_old, p_old = _run(legacy.DDIMSampler, cfg, model, proc, seed=1234, cond=cond, **kw)
    x_new, p_new = _run(DDIMSampler, cfg, model, proc, seed=1234, cond=cond, **kw)

    assert torch.equal(x_old, x_new), (x_old - x_new).abs().max()
    assert torch.equal(p_old, p_new), (p_old - p_new).abs().max()


@pytest.mark.parametrize("self_condition", [False, True])
def test_ddim_unconditional_matches_pre_refactor(self_condition):
    """No prompt at all: the guidance machinery must stay completely inert."""
    cfg, model, proc = _build(self_condition)
    legacy = load_legacy_samplers()

    kw = dict(guidance_scale=None, sc_refresh_mode="refined", ati_eta=0.0)
    x_old, p_old = _run(legacy.DDIMSampler, cfg, model, proc, seed=7, cond=None, **kw)
    x_new, p_new = _run(DDIMSampler, cfg, model, proc, seed=7, cond=None, **kw)

    assert torch.equal(x_old, x_new)
    assert torch.equal(p_old, p_new)


@pytest.mark.parametrize("posterior_temp,target", [(0.5, "learned"), (0.5, "full"), (2.0, "learned")])
def test_ddim_posterior_temp_matches_pre_refactor(posterior_temp, target):
    """Temperature interacts with the matched filter; it must survive the refactor."""
    cfg, model, proc = _build(self_condition=True)
    legacy = load_legacy_samplers()
    cond = make_conditioning(B, S, NPROMPT, seed=2)

    kw = dict(
        guidance_scale=3.0, sc_refresh_mode="refined", ati_eta=0.0,
        posterior_temp=posterior_temp, posterior_temp_target=target,
    )
    x_old, p_old = _run(legacy.DDIMSampler, cfg, model, proc, seed=99, cond=cond, **kw)
    x_new, p_new = _run(DDIMSampler, cfg, model, proc, seed=99, cond=cond, **kw)

    assert torch.equal(x_old, x_new)
    assert torch.equal(p_old, p_new)


def test_ddim_score_temp_matches_pre_refactor():
    """Track-A1 score temperature multiplies the drift after guidance."""
    cfg, model, proc = _build(self_condition=True)
    legacy = load_legacy_samplers()
    cond = make_conditioning(B, S, NPROMPT, seed=3)

    kw = dict(guidance_scale=2.0, sc_refresh_mode="carry", ati_eta=0.0,
              score_temp_tau=0.7, score_temp_clean_var=0.25)
    x_old, p_old = _run(legacy.DDIMSampler, cfg, model, proc, seed=5, cond=cond, **kw)
    x_new, p_new = _run(DDIMSampler, cfg, model, proc, seed=5, cond=cond, **kw)

    assert torch.equal(x_old, x_new)
    assert torch.equal(p_old, p_new)


def test_ddim_stochastic_churn_matches_pre_refactor():
    """EDM churn draws noise inside the loop: identical seeds must stay in lockstep."""
    from evaluation.tasks._task_common import configure_stochastic

    cfg, model, proc = _build(self_condition=True)
    configure_stochastic(cfg, mode="stochastic", gamma=0.3, num_steps=STEPS)
    legacy = load_legacy_samplers()
    cond = make_conditioning(B, S, NPROMPT, seed=4)

    kw = dict(guidance_scale=2.0, sc_refresh_mode="carry", ati_eta=0.0)
    x_old, p_old = _run(legacy.DDIMSampler, cfg, model, proc, seed=11, cond=cond, **kw)
    x_new, p_new = _run(DDIMSampler, cfg, model, proc, seed=11, cond=cond, **kw)

    assert torch.equal(x_old, x_new)
    assert torch.equal(p_old, p_new)

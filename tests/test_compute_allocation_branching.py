"""BranchingSampler (experiments/compute_allocation) against the production DDIM loop.

A tiny randomly initialised SDT on the real LM1B eval config and entropy tables,
on CPU, with the Table 1 stochastic spec (gamma 0.2, full entropy band):

  * GlobalNoise + no sub-steps reproduces DDIMSampler.sample bit for bit;
  * resuming from a saved state reproduces the uninterrupted run;
  * factor 1 is the identity, factor m adds (m-1) NFE per step and changes x;
  * keyed churn noise of a step does not depend on what ran before it, and the
    stream-0 rows of a two-stream batch match a one-stream run.
"""
import pytest
import torch

from diffusion.continuous.processes import ContinuousForwardProcess
from diffusion.continuous.samplers import DDIMSampler
from evaluation.utils import load_config
from experiments.compute_allocation.branch_sampler import (
    BranchingSampler, GlobalNoise, KeyedNoise, SamplerState,
)
from experiments.compute_allocation.run import CONFIG, ENTROPY_DIR, SIGMA_MIN, configure
from models import create_model

STEPS = 17   # 16 integration steps + final denoise


def _cfg():
    cfg = configure(load_config(CONFIG), "cpu")
    cfg.model.use_flash_attn = False
    cfg.train.use_compile = False
    cfg.evaluation.use_compile = False
    cfg.model.embed_dim = 32
    cfg.model.dim_ff = 64
    cfg.model.n_blocks = 1
    cfg.model.n_heads = 2
    cfg.model.head_hidden = 16
    cfg.model.head_embed_dim = 8
    cfg.data.sequence_len_tokens = 8          # short sequence: control flow, not scale
    cfg.data.sequence_len = 8 * int(cfg.data.bits_per_token)
    return cfg


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    cfg = _cfg()
    model = create_model(cfg).eval()
    proc = ContinuousForwardProcess(cfg)
    bs = BranchingSampler(model, proc, cfg, num_steps=STEPS, sigma_min=SIGMA_MIN,
                          entropy_run_dir=ENTROPY_DIR)
    return cfg, model, proc, bs


def test_matches_production_ddim(setup):
    cfg, model, proc, bs = setup
    assert bs.stoch.enabled
    B, S = 3, int(cfg.data.sequence_len)
    torch.manual_seed(123)
    _, p_ref = DDIMSampler(model, proc, cfg).sample(
        B, S, schedule="entropic", num_steps=STEPS, sigma_min_override=SIGMA_MIN,
        entropy_run_dir=ENTROPY_DIR,
        conditioning_prefix_full=torch.zeros(B, S), cond_prefix_mask=torch.zeros(B, S, dtype=torch.bool),
        guidance_scale=0.0, sc_refresh_mode="carry", ati_eta=0.0, return_probs=True, progress=False,
    )
    torch.manual_seed(123)
    noise = GlobalNoise()
    st = bs.initial_state(B, S, noise)
    p, _, nfe = bs.run(st, noise)
    assert torch.equal(p, p_ref)
    assert nfe == bs.n_steps + 1 == STEPS


def test_resume_from_saved_state(setup):
    cfg, model, proc, bs = setup
    B, S = 2, int(cfg.data.sequence_len)
    noise = KeyedNoise(7, [(0, 0, B)])
    st = bs.initial_state(B, S, noise)
    p_full, saved, _ = bs.run(st, noise, save_at=[0, 6, 11])
    for step in (6, 11):
        p_res, _, nfe = bs.run(saved[step], noise)
        assert torch.equal(p_res, p_full)
        assert nfe == bs.n_steps - step + 1


def test_factors(setup):
    cfg, model, proc, bs = setup
    B, S = 2, int(cfg.data.sequence_len)
    noise = KeyedNoise(7, [(0, 0, B)])
    st = bs.initial_state(B, S, noise)
    p0, _, n0 = bs.run(st, noise)
    p1, _, n1 = bs.run(st, noise, factors={i: 1 for i in range(4, 8)})
    assert torch.equal(p0, p1) and n0 == n1
    p3, _, n3 = bs.run(st, noise, factors={i: 3 for i in range(4, 8)})
    assert n3 == n0 + 4 * 2
    assert not torch.equal(p3, p0)
    # Steps before the refined segment are untouched: identical saved state at step 4.
    _, sa, _ = bs.run(st, noise, save_at=[4])
    _, sb, _ = bs.run(st, noise, factors={i: 3 for i in range(4, 8)}, save_at=[4])
    assert torch.equal(sa[4].x, sb[4].x)


def test_keyed_noise_streams(setup):
    cfg, model, proc, bs = setup
    n, S = 2, int(cfg.data.sequence_len)
    one = KeyedNoise(7, [(0, 0, n)])
    two = KeyedNoise(7, [(0, 0, n), (0, 1, n)])
    x = torch.zeros(n, S)
    assert torch.equal(one.churn(5, x), one.churn(5, x))
    assert torch.equal(two.churn(5, torch.zeros(2 * n, S))[:n], one.churn(5, x))
    assert not torch.equal(two.churn(5, torch.zeros(2 * n, S))[n:], one.churn(5, x))
    assert torch.equal(two.initial(2 * n, S, "cpu")[:n], two.initial(2 * n, S, "cpu")[n:])

    st = bs.initial_state(n, S, one)
    _, saved, _ = bs.run(st, one, save_at=[6])
    p1, _, _ = bs.run(saved[6], one)
    p2, _, _ = bs.run(SamplerState.cat([saved[6], saved[6]]), two)
    torch.testing.assert_close(p2[:n], p1, rtol=0, atol=1e-5)
    assert (p2[n:] - p1).abs().max() > 1e-3   # stream 1 is a different continuation

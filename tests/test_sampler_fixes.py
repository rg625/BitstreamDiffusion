"""Score temperature with an unsharpened final step, and the VE ancestral sampler (13 Oct 2026).

* score_temp_final="unsharpened" leaves every step but the last bit-identical to the sharpened run, changes only
  the last one, and is a no-op at tau = 1;
* AncestralSampler's step is the exact VE Gaussian bridge q(x_next | x, x0 = D): mean D + rho^2 (x - D),
  variance sigma_next^2 (1 - rho^2); it refuses EDM churn and accepts score temperature.
"""
import pytest
import torch

from diffusion.continuous.processes import ContinuousForwardProcess
from diffusion.continuous.samplers import AncestralSampler, DDIMSampler
from tests._sampler_harness import TinyBinaryDenoiser, make_conditioning, make_cpu_cfg

B, S, NP, STEPS = 2, 16, 4, 12


def _sampler(cls=DDIMSampler):
    cfg = make_cpu_cfg(num_steps=STEPS)
    return cls(TinyBinaryDenoiser(S, seed=1), ContinuousForwardProcess(cfg), cfg)


def _trace(sampler, **kw):
    pf, pm = make_conditioning(B, S, NP, seed=7)
    states, orig = [], sampler._integrate_step

    def spy(x_state, h, d_cur, **k):
        out = orig(x_state, h, d_cur, **k)
        states.append((x_state.clone(), d_cur.clone(), out.clone()))
        return out

    sampler._integrate_step = spy
    torch.manual_seed(0)
    x, p = sampler.sample(num_samples=B, seq_len=S, conditioning_prefix_full=pf, cond_prefix_mask=pm,
                          num_steps=STEPS, schedule="karras", sc_refresh_mode="carry", ati_eta=0.0,
                          return_probs=True, progress=False, **kw)
    return states, x, p, pm


def test_unsharpened_final_only_changes_the_last_step():
    a, xa, _, _ = _trace(_sampler(), score_temp_tau=0.3, score_temp_mode="global")
    b, xb, _, _ = _trace(_sampler(), score_temp_tau=0.3, score_temp_mode="global", score_temp_final="unsharpened")
    assert len(a) == len(b) == STEPS - 1
    for (xs_a, d_a, o_a), (xs_b, d_b, o_b) in zip(a[:-1], b[:-1]):
        assert torch.equal(xs_a, xs_b) and torch.equal(d_a, d_b) and torch.equal(o_a, o_b)
    # last step: same input state, drift differs by exactly 1/tau (sharpened = unsharpened / tau)
    assert torch.equal(a[-1][0], b[-1][0])
    torch.testing.assert_close(a[-1][1], b[-1][1] / 0.3, rtol=1e-5, atol=1e-6)
    assert not torch.equal(xa, xb)


def test_unsharpened_final_is_a_noop_at_tau_one():
    _, xa, pa, _ = _trace(_sampler())
    _, xb, pb, _ = _trace(_sampler(), score_temp_tau=1.0, score_temp_final="unsharpened")
    assert torch.equal(xa, xb) and torch.equal(pa, pb)


def test_unknown_final_mode_is_rejected():
    with pytest.raises(ValueError, match="score_temp_final"):
        _trace(_sampler(), score_temp_tau=0.5, score_temp_final="sometimes")


def test_ancestral_step_is_the_exact_ve_bridge():
    s = _sampler(AncestralSampler)
    g = torch.Generator().manual_seed(0)
    n = 400_000
    x = torch.randn(n, 1, generator=g) * 2.0
    D = torch.sigmoid(torch.randn(n, 1, generator=g))
    sc, sn = torch.tensor(1.7), torch.tensor(0.6)
    d = (x - D) / sc                                        # the drift the shared loop hands over
    torch.manual_seed(1)
    out = s._integrate_step(x, sn - sc, d, sigma_cur=sc, sigma_next=sn, prefix_mask=None)
    rho2 = float((sn / sc) ** 2)
    resid = out - (D + rho2 * (x - D))
    assert abs(float(resid.mean())) < 5e-3
    assert abs(float(resid.var()) - float(sn ** 2) * (1 - rho2)) < 5e-3


def test_ancestral_runs_with_score_temperature_and_keeps_the_prompt():
    states, x, p, pm = _trace(_sampler(AncestralSampler), score_temp_tau=0.2, score_temp_mode="global",
                              score_temp_final="unsharpened")
    assert torch.isfinite(x).all() and torch.isfinite(p).all()
    _, x1, _, _ = _trace(_sampler(AncestralSampler))
    assert torch.equal(x[pm], x1[pm])                       # clamped prompt never moves


def test_ancestral_refuses_edm_churn():
    from evaluation.tasks._task_common import configure_stochastic
    s = _sampler(AncestralSampler)
    configure_stochastic(s.cfg, mode="stochastic", gamma=0.41, num_steps=STEPS)
    with pytest.raises(RuntimeError, match="churn"):
        _trace(s)

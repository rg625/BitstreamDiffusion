"""Phase-4 unit tests for CFG / AutoGuidance / Self-Guidance.

Organised as gates, matching the repo's existing `tests/test_fkc.py` style:

  A  combination algebra (pure tensor maths, no model)
  B  D-space == score-space equivalence, the claim the whole design rests on
  C  batched vs separate model evaluation
  D  self-guidance: finite differences, normalisation, exact vs previous-step
  E  matched-filter interaction -- guidance must not amplify the analytic term
  F  conditioning and self-conditioning hygiene
  G  numerical stability
"""
from __future__ import annotations

import math

import pytest
import torch

from diffusion.continuous.guidance import (
    BAD_C,
    BAD_U,
    GOOD_C,
    GOOD_U,
    GuidanceConfig,
    GuidedDenoiser,
    apply_sg,
    combine_cfg_ag,
    lerp_guidance,
    score_from_D,
    sg_direction,
)
from diffusion.continuous.samplers import _score_from_probs
from tests._sampler_harness import TinyBinaryDenoiser, make_conditioning, make_cpu_cfg

B, S, NPROMPT = 4, 32, 8


def _rand_branches(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {k: torch.rand(B, S, generator=g) for k in (GOOD_C, GOOD_U, BAD_C, BAD_U)}


# =============================================================================
# A. Combination algebra
# =============================================================================

def test_a1_lerp_is_exact():
    a, b = torch.rand(B, S), torch.rand(B, S)
    for w in (0.0, 0.25, 1.0, 3.0, -1.5):
        assert torch.allclose(lerp_guidance(a, b, w), a + w * (b - a), atol=0, rtol=0)


def test_a2_cfg_zero_and_one_both_give_the_conditional():
    """w=0 disables CFG; w=1 is the algebraic no-op. Both == conditional."""
    br = _rand_branches(1)
    for w in (0.0, 1.0):
        out = combine_cfg_ag(br, cfg_scale=w, ag_scale=0.0, cond_enabled=True)
        assert torch.equal(out, br[GOOD_C])


def test_a3_cfg_matches_closed_form_and_extrapolates():
    br = _rand_branches(2)
    for w in (0.5, 2.0, 7.0):
        out = combine_cfg_ag(br, cfg_scale=w, ag_scale=0.0, cond_enabled=True)
        assert torch.allclose(out, br[GOOD_U] + w * (br[GOOD_C] - br[GOOD_U]))


def test_a4_cfg_inert_without_conditioning():
    """No prompt => nothing to drop => CFG cannot do anything."""
    br = _rand_branches(3)
    out = combine_cfg_ag(br, cfg_scale=5.0, ag_scale=0.0, cond_enabled=False)
    assert torch.equal(out, br[GOOD_C])


def test_a5_ag_zero_and_one_both_give_the_good_model():
    br = _rand_branches(4)
    for w in (0.0, 1.0):
        out = combine_cfg_ag(br, cfg_scale=0.0, ag_scale=w, cond_enabled=True)
        assert torch.equal(out, br[GOOD_C])


def test_a6_ag_matches_closed_form():
    br = _rand_branches(5)
    for w in (0.5, 2.0, 4.0):
        out = combine_cfg_ag(br, cfg_scale=0.0, ag_scale=w, cond_enabled=True)
        assert torch.allclose(out, br[BAD_C] + w * (br[GOOD_C] - br[BAD_C]))


def test_a7_identical_good_and_bad_kills_autoguidance():
    """The AG direction is good - bad, so a bad model equal to the good one
    must leave the prediction untouched at every scale."""
    br = _rand_branches(6)
    br[BAD_C] = br[GOOD_C].clone()
    br[BAD_U] = br[GOOD_U].clone()
    for w_ag in (0.0, 1.0, 3.0, 10.0):
        for w_cfg in (0.0, 2.0):
            out = combine_cfg_ag(br, cfg_scale=w_cfg, ag_scale=w_ag, cond_enabled=True)
            ref = combine_cfg_ag(br, cfg_scale=w_cfg, ag_scale=0.0, cond_enabled=True)
            assert torch.allclose(out, ref, atol=1e-6)


def test_a8_nested_cfg_ag_matches_the_specified_formula():
    br = _rand_branches(7)
    w_cfg, w_ag = 2.5, 1.75
    good = br[GOOD_U] + w_cfg * (br[GOOD_C] - br[GOOD_U])
    bad = br[BAD_U] + w_cfg * (br[BAD_C] - br[BAD_U])
    ref = bad + w_ag * (good - bad)
    out = combine_cfg_ag(br, cfg_scale=w_cfg, ag_scale=w_ag, cond_enabled=True)
    assert torch.allclose(out, ref, atol=0, rtol=0)


def test_a9_shape_dtype_device_preserved():
    br = _rand_branches(8)
    for dt in (torch.float32, torch.float64):
        b2 = {k: v.to(dt) for k, v in br.items()}
        out = combine_cfg_ag(b2, cfg_scale=2.0, ag_scale=2.0, cond_enabled=True)
        assert out.shape == (B, S) and out.dtype == dt and out.device == br[GOOD_C].device


# =============================================================================
# B. D-space == score-space
# =============================================================================

@pytest.mark.parametrize("w_cfg,w_ag", [(0.0, 0.0), (3.0, 0.0), (0.0, 2.0), (3.0, 2.0)])
def test_b1_guiding_D_equals_guiding_the_score(w_cfg, w_ag):
    """The load-bearing claim: because score = (D - x)/sigma^2 is affine in D
    with x and sigma shared by every branch, combining posterior means and
    combining scores are the same operation."""
    br = _rand_branches(9)
    x = torch.rand(B, S)
    sigma = torch.full((B,), 0.7)

    D_guided = combine_cfg_ag(br, cfg_scale=w_cfg, ag_scale=w_ag, cond_enabled=True)
    score_of_guided_D = score_from_D(D_guided, x, sigma)

    scores = {k: score_from_D(v, x, sigma) for k, v in br.items()}
    guided_score = combine_cfg_ag(scores, cfg_scale=w_cfg, ag_scale=w_ag, cond_enabled=True)

    assert torch.allclose(score_of_guided_D, guided_score, atol=1e-5)


def test_b2_score_helper_matches_the_sampler():
    D, x = torch.rand(B, S), torch.rand(B, S)
    sigma = torch.full((B,), 0.35)
    assert torch.allclose(score_from_D(D, x, sigma), _score_from_probs(D, x, sigma))


# =============================================================================
# C. Batched vs separate evaluation
# =============================================================================

class PerRowSigmaDenoiser(torch.nn.Module):
    """Deterministic stand-in that honours a PER-ROW sigma.

    `tests._sampler_harness.TinyBinaryDenoiser` reduces sigma to
    `sigma.reshape(-1)[0]`, i.e. one scalar for the whole batch. That is fine for
    the existing sampler gates (which always pass a uniform sigma) but it cannot
    represent SG-exact, which packs two noise levels into one batch. CoBit's real
    SDT is per-row (`SigmaEmbedding` uses `sigma.log()[:, None]`), so the SG tests
    use this stand-in instead.
    """

    def __init__(self, seq_len: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        self.register_buffer("w", torch.randn(seq_len, generator=g) * 0.7)
        self.register_buffer("b", torch.randn(seq_len, generator=g) * 0.1)
        self.gain = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x_t, sigma, x0_hat=None):
        s = sigma.reshape(-1, 1).to(x_t.dtype)           # [B,1] -- per row
        target = torch.sigmoid(self.w).unsqueeze(0)
        drive = (target - (x_t - 0.5)) / (1.0 + s)
        sc = 0.0 if x0_hat is None else (x0_hat - 0.5) * 0.1
        gmean = (x_t.to(target.dtype).mean(dim=-1, keepdim=True) - 0.5)
        return (drive + self.b.unsqueeze(0) + sc + 0.5 * gmean) * self.gain


def _denoiser(gcfg, *, bad=False, sc=True, diag=False, per_row=False):
    cfg = make_cpu_cfg(self_condition=sc, num_steps=8)
    cls = PerRowSigmaDenoiser if per_row else TinyBinaryDenoiser
    good = cls(S, seed=0).eval()
    bad_model = cls(S, seed=99).eval() if bad else None
    return cfg, GuidedDenoiser(good, cfg, gcfg, bad_model=bad_model,
                              collect_diagnostics=diag), good, bad_model


def _ctx():
    pf, pm = make_conditioning(B, S, NPROMPT, seed=3)
    from diffusion.continuous.samplers import _make_null_full
    cfg = make_cpu_cfg(num_steps=8)
    nf = _make_null_full(pf, pm, cfg)
    return pf, pm, nf


def _manual_branch_D(model, cfg, x_state, sigma, branch_full, pm, sc_tensor):
    """Reference: one un-batched model call for a single branch."""
    from diffusion.continuous.logit_postprocess import _model_logits_continuous
    from diffusion.continuous.samplers import _clamp_mask_
    xin = x_state.clone()
    _clamp_mask_(xin, branch_full, pm)
    scin = sc_tensor.clone()
    _clamp_mask_(scin, branch_full, pm)
    lg = _model_logits_continuous(model, cfg, xin, sigma, scin)
    D = torch.sigmoid(lg.float())
    _clamp_mask_(D, branch_full, pm)
    return D


@pytest.mark.parametrize("w_cfg,w_ag", [(3.0, 0.0), (0.0, 2.0), (3.0, 2.0)])
def test_c1_batched_equals_separate_evaluation(w_cfg, w_ag):
    """Concatenating branches into one forward pass must not change the answer."""
    gcfg = GuidanceConfig(cfg_scale=w_cfg, ag_scale=w_ag)
    cfg, gdn, good, bad = _denoiser(gcfg, bad=w_ag > 0)
    pf, pm, nf = _ctx()
    torch.manual_seed(0)
    x = torch.rand(B, S)
    sigma = torch.full((B,), 0.9)

    sc = gdn.make_self_cond_state(True)
    for b in gdn.branches(True):
        sc.set(b, torch.zeros(B, S))
    pred = gdn.denoise(x, sigma, sc=sc, prefix_full=pf, prefix_mask=pm, null_full=nf,
                       cond_enabled=True)

    ref = {}
    for b in gdn.branches(True):
        model = good if b[0] == "good" else bad
        full = nf if b[1] == "u" else pf
        z = torch.zeros(B, S)
        from diffusion.continuous.samplers import _clamp_mask_
        _clamp_mask_(z, full, pm)
        ref[b] = _manual_branch_D(model, cfg, x, sigma, full, pm, z)

    for b in gdn.branches(True):
        assert torch.allclose(pred.D_branches[b], ref[b], atol=1e-6), b
    expect = combine_cfg_ag(ref, cfg_scale=w_cfg, ag_scale=w_ag, cond_enabled=True)
    from diffusion.continuous.samplers import _clamp_mask_
    _clamp_mask_(expect, pf, pm)
    assert torch.allclose(pred.D, expect, atol=1e-6)


def test_c2_model_evaluation_count_is_reported():
    """One forward pass per distinct network, regardless of branch count."""
    pf, pm, nf = _ctx()
    x, sigma = torch.rand(B, S), torch.full((B,), 0.5)

    # NFE counts rows/B, so it equals the number of branches evaluated -- the
    # quantity Phase-14 compute comparisons need -- not the number of passes.
    for gcfg, bad, expect in [
        (GuidanceConfig(), False, 1),                            # conditional only
        (GuidanceConfig(cfg_scale=3.0), False, 2),               # c+u, batched in 1 pass
        (GuidanceConfig(ag_scale=2.0), True, 2),                 # good_c + bad_c
        (GuidanceConfig(cfg_scale=3.0, ag_scale=2.0), True, 4),  # 4 branches, 2 passes
    ]:
        _, gdn, _, _ = _denoiser(gcfg, bad=bad)
        sc = gdn.make_self_cond_state(True)
        gdn.denoise(x, sigma, sc=sc, prefix_full=pf, prefix_mask=pm, null_full=nf,
                    cond_enabled=True)
        assert gdn.model_evaluations == expect, (gcfg, gdn.model_evaluations)


# =============================================================================
# D. Self-guidance
# =============================================================================

def test_d1_sg_direction_is_a_normalised_finite_difference():
    a, b = torch.rand(B, S), torch.rand(B, S)
    # delta == delta_ref reduces to the plain difference.
    assert torch.allclose(sg_direction(a, b, delta=0.5, delta_ref=0.5), a - b)
    # Halving the realised spacing doubles the normalised direction.
    d1 = sg_direction(a, b, delta=0.5, delta_ref=0.5)
    d2 = sg_direction(a, b, delta=0.25, delta_ref=0.5)
    assert torch.allclose(d2, 2.0 * d1)


def test_d2_sg_normalisation_cancels_grid_spacing():
    """A prediction that is linear in log-sigma must give a spacing-independent
    direction -- this is what makes SG-prev comparable across NFE."""
    slope = torch.randn(B, S)
    ref = 0.5
    out = []
    for delta in (0.05, 0.1, 0.4, 1.0):
        D_cur = torch.zeros(B, S)
        D_bad = D_cur + slope * delta          # linear in log-sigma
        out.append(sg_direction(D_cur, D_bad, delta=delta, delta_ref=ref))
    for o in out[1:]:
        assert torch.allclose(o, out[0], atol=1e-6)
    assert torch.allclose(out[0], -ref * slope, atol=1e-6)


def test_d3_equal_predictions_give_zero_guidance():
    a = torch.rand(B, S)
    d = sg_direction(a, a.clone(), delta=0.3, delta_ref=0.5)
    assert torch.count_nonzero(d) == 0
    assert torch.equal(apply_sg(a, d, 5.0), a)


def test_d4_zero_spacing_is_rejected():
    a, b = torch.rand(B, S), torch.rand(B, S)
    with pytest.raises(ValueError):
        sg_direction(a, b, delta=0.0, delta_ref=0.5)


def test_d5_sg_prev_costs_no_extra_model_evaluations():
    """The whole point of the cheap variant: same NFE as no self-guidance."""
    pf, pm, nf = _ctx()
    x, sigma = torch.rand(B, S), torch.full((B,), 1.0)

    _, plain, _, _ = _denoiser(GuidanceConfig())
    _, sgp, _, _ = _denoiser(GuidanceConfig(sg_scale=1.0, sg_variant="prev"))
    _, sge, _, _ = _denoiser(GuidanceConfig(sg_scale=1.0, sg_variant="exact"), per_row=True)

    for gdn in (plain, sgp, sge):
        sc = gdn.make_self_cond_state(True)
        for s in (1.0, 0.6, 0.3):
            gdn.denoise(x, torch.full((B,), s), sc=sc, prefix_full=pf, prefix_mask=pm,
                        null_full=nf, cond_enabled=True)
    assert sgp.model_evaluations == plain.model_evaluations
    # SG-exact concatenates the shifted level into the same forward pass, so it
    # costs the same number of *passes* but twice the rows -- and NFE counts rows.
    assert sge.model_evaluations == 2 * plain.model_evaluations


def test_d6_sg_prev_is_inert_on_the_first_step():
    """No cached previous prediction yet => no correction, not a crash."""
    pf, pm, nf = _ctx()
    x, sigma = torch.rand(B, S), torch.full((B,), 1.0)
    _, sgp, _, _ = _denoiser(GuidanceConfig(sg_scale=4.0, sg_variant="prev"))
    _, plain, _, _ = _denoiser(GuidanceConfig())
    kw = dict(prefix_full=pf, prefix_mask=pm, null_full=nf, cond_enabled=True)
    a = sgp.denoise(x, sigma, sc=sgp.make_self_cond_state(True), **kw)
    b = plain.denoise(x, sigma, sc=plain.make_self_cond_state(True), **kw)
    assert torch.allclose(a.D, b.D, atol=1e-7)


def test_d7_sg_exact_matches_a_hand_built_two_level_difference():
    gcfg = GuidanceConfig(sg_scale=2.0, sg_variant="exact", sg_delta=0.4, sg_mf_mode="vary")
    cfg, gdn, good, _ = _denoiser(gcfg, per_row=True)
    pf, pm, nf = _ctx()
    x, sigma = torch.rand(B, S), torch.full((B,), 0.8)

    sc = gdn.make_self_cond_state(True)
    z = torch.zeros(B, S)
    from diffusion.continuous.samplers import _clamp_mask_
    _clamp_mask_(z, pf, pm)
    sc.set(GOOD_C, z)
    pred = gdn.denoise(x, sigma, sc=sc, prefix_full=pf, prefix_mask=pm, null_full=nf,
                       cond_enabled=True)

    D_cur = _manual_branch_D(good, cfg, x, sigma, pf, pm, z)
    D_hi = _manual_branch_D(good, cfg, x, sigma * math.exp(0.4), pf, pm, z)
    ref = D_cur + 2.0 * (D_cur - D_hi)       # delta == delta_ref => plain difference
    _clamp_mask_(ref, pf, pm)
    assert torch.allclose(pred.D, ref, atol=1e-6)


def test_d8_sg_scale_zero_is_a_no_op():
    pf, pm, nf = _ctx()
    x = torch.rand(B, S)
    kw = dict(prefix_full=pf, prefix_mask=pm, null_full=nf, cond_enabled=True)
    for variant in ("prev", "exact"):
        _, g0, _, _ = _denoiser(GuidanceConfig(sg_scale=0.0, sg_variant=variant), per_row=True)
        _, gp, _, _ = _denoiser(GuidanceConfig(), per_row=True)
        s0, sp = g0.make_self_cond_state(True), gp.make_self_cond_state(True)
        for s in (1.2, 0.5):
            a = g0.denoise(x, torch.full((B,), s), sc=s0, **kw)
            b = gp.denoise(x, torch.full((B,), s), sc=sp, **kw)
            assert torch.allclose(a.D, b.D, atol=1e-7), variant


# =============================================================================
# E. Matched-filter interaction  (the decisive gate)
# =============================================================================

class _SigmaBlindDenoiser(torch.nn.Module):
    """Learned logit independent of sigma; only the matched filter varies with it.

    With such a model the learned component has *zero* noise-level sensitivity,
    so a correct self-guidance direction must be exactly zero. Anything non-zero
    is the analytic matched-filter term leaking into the guidance.
    """

    def __init__(self, seq_len, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("w", torch.randn(seq_len, generator=g) * 0.5)
        self.gain = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x_t, sigma, x0_hat=None):
        return (x_t * 0.0 + self.w.unsqueeze(0)) * self.gain


def test_e1_sg_hold_does_not_amplify_the_matched_filter():
    """sg_mf_mode='hold' must give zero guidance for a sigma-blind network."""
    cfg = make_cpu_cfg(self_condition=False, num_steps=8)
    assert cfg.model.continuous_logit_scaling == "matched_filter_residual"
    model = _SigmaBlindDenoiser(S).eval()
    pf, pm, nf = _ctx()
    x, sigma = torch.rand(B, S), torch.full((B,), 0.6)
    kw = dict(prefix_full=pf, prefix_mask=pm, null_full=nf, cond_enabled=True)

    hold = GuidedDenoiser(model, cfg, GuidanceConfig(
        sg_scale=5.0, sg_variant="exact", sg_delta=0.5, sg_mf_mode="hold"))
    plain = GuidedDenoiser(model, cfg, GuidanceConfig())
    a = hold.denoise(x, sigma, sc=hold.make_self_cond_state(True), **kw)
    b = plain.denoise(x, sigma, sc=plain.make_self_cond_state(True), **kw)
    assert torch.allclose(a.D, b.D, atol=1e-6), (a.D - b.D).abs().max()


def test_e2_sg_vary_does_amplify_the_matched_filter():
    """The naive variant is measurably wrong on the same model -- which is why
    'hold' is the default."""
    cfg = make_cpu_cfg(self_condition=False, num_steps=8)
    model = _SigmaBlindDenoiser(S).eval()
    pf, pm, nf = _ctx()
    x, sigma = torch.rand(B, S), torch.full((B,), 0.6)
    kw = dict(prefix_full=pf, prefix_mask=pm, null_full=nf, cond_enabled=True)

    vary = GuidedDenoiser(model, cfg, GuidanceConfig(
        sg_scale=5.0, sg_variant="exact", sg_delta=0.5, sg_mf_mode="vary"))
    plain = GuidedDenoiser(model, cfg, GuidanceConfig())
    a = vary.denoise(x, sigma, sc=vary.make_self_cond_state(True), **kw)
    b = plain.denoise(x, sigma, sc=plain.make_self_cond_state(True), **kw)
    free = ~pm
    assert (a.D - b.D)[free].abs().max() > 1e-3


def test_e3_cfg_and_ag_directions_are_already_purely_learned():
    """A sigma-blind *and* input-blind model makes the analytic term the only
    thing that could differ between branches; CFG and AG must still give zero."""
    cfg = make_cpu_cfg(self_condition=False, num_steps=8)
    model = _SigmaBlindDenoiser(S).eval()
    pf, pm, nf = _ctx()
    x, sigma = torch.rand(B, S), torch.full((B,), 0.6)
    kw = dict(prefix_full=pf, prefix_mask=pm, null_full=nf, cond_enabled=True)

    gdn = GuidedDenoiser(model, cfg, GuidanceConfig(cfg_scale=6.0))
    pred = gdn.denoise(x, sigma, sc=gdn.make_self_cond_state(True), **kw)
    free = ~pm
    # x_c and x_u differ only on prompt positions, so on free coordinates the
    # matched filter is identical and the (constant) learned logit cancels.
    assert (pred.D_branches[GOOD_C] - pred.D_branches[GOOD_U])[free].abs().max() < 1e-6


# =============================================================================
# F. Conditioning and self-conditioning hygiene
# =============================================================================

def test_f1_null_prefix_touches_only_prompt_positions():
    from diffusion.continuous.samplers import _make_null_full
    cfg = make_cpu_cfg(num_steps=8)
    pf, pm = make_conditioning(B, S, NPROMPT, seed=5)
    nf = _make_null_full(pf, pm, cfg)
    assert torch.equal(nf[~pm], pf[~pm])                 # suffix untouched
    assert torch.all(nf[pm] == 0.5)                      # null_strategy="half"
    assert not torch.equal(nf[pm], pf[pm]) or torch.all(pf[pm] == 0.5)


def test_f2_branch_inputs_differ_only_on_the_prompt():
    gcfg = GuidanceConfig(cfg_scale=3.0)
    _, gdn, _, _ = _denoiser(gcfg)
    pf, pm, nf = _ctx()
    x = torch.rand(B, S)
    xc = gdn._branch_inputs(GOOD_C, x, pf, pm, nf, True)
    xu = gdn._branch_inputs(GOOD_U, x, pf, pm, nf, True)
    assert torch.equal(xc[~pm], xu[~pm])                 # free coords identical
    assert torch.equal(xc[~pm], x[~pm])                  # and untouched
    assert torch.equal(xc[pm], pf[pm])
    assert torch.equal(xu[pm], nf[pm])


def test_f3_self_conditioning_states_never_leak_between_branches():
    gcfg = GuidanceConfig(cfg_scale=3.0, ag_scale=2.0)
    _, gdn, _, _ = _denoiser(gcfg, bad=True)
    sc = gdn.make_self_cond_state(True)
    marks = {b: torch.full((B, S), float(i)) for i, b in enumerate(gdn.branches(True))}
    for b, v in marks.items():
        sc.set(b, v)
    for b, v in marks.items():
        assert torch.equal(sc.get(b), v), b
    assert set(sc.branches) == set(gdn.branches(True))
    with pytest.raises(KeyError):
        sc.get(("good", "nonsense"))


def test_f4_self_conditioning_state_is_detached():
    _, gdn, _, _ = _denoiser(GuidanceConfig())
    sc = gdn.make_self_cond_state(True)
    t = torch.rand(B, S, requires_grad=True)
    sc.set(GOOD_C, t)
    assert not sc.get(GOOD_C).requires_grad


def test_f5_denoise_does_not_mutate_the_stored_sc_state():
    """The branch input is prompt-clamped; that must not write through to the
    per-branch belief the next step will read."""
    _, gdn, _, _ = _denoiser(GuidanceConfig(cfg_scale=2.0))
    pf, pm, nf = _ctx()
    sc = gdn.make_self_cond_state(True)
    orig = torch.rand(B, S)
    sc.set(GOOD_U, orig.clone())
    sc.set(GOOD_C, torch.rand(B, S))
    gdn.denoise(torch.rand(B, S), torch.full((B,), 0.5), sc=sc, prefix_full=pf,
                prefix_mask=pm, null_full=nf, cond_enabled=True)
    assert torch.equal(sc.get(GOOD_U), orig)


def test_f6_guided_output_keeps_the_prompt_clamped():
    _, gdn, _, _ = _denoiser(GuidanceConfig(cfg_scale=7.0))
    pf, pm, nf = _ctx()
    pred = gdn.denoise(torch.rand(B, S), torch.full((B,), 0.5),
                       sc=gdn.make_self_cond_state(True), prefix_full=pf,
                       prefix_mask=pm, null_full=nf, cond_enabled=True)
    assert torch.equal(pred.D[pm], pf[pm])


# =============================================================================
# G. Numerical stability
# =============================================================================

@pytest.mark.parametrize("sigma", [80.0, 1.0, 0.01, 0.002])
@pytest.mark.parametrize("w_cfg,w_ag,w_sg", [(0.0, 0.0, 0.0), (8.0, 0.0, 0.0),
                                             (0.0, 5.0, 0.0), (4.0, 3.0, 2.0)])
def test_g1_finite_across_sigma_and_scale(sigma, w_cfg, w_ag, w_sg):
    gcfg = GuidanceConfig(cfg_scale=w_cfg, ag_scale=w_ag, sg_scale=w_sg,
                          sg_variant="exact", sg_delta=0.5)
    _, gdn, _, _ = _denoiser(gcfg, bad=w_ag > 0, per_row=True)
    pf, pm, nf = _ctx()
    pred = gdn.denoise(torch.rand(B, S), torch.full((B,), sigma),
                       sc=gdn.make_self_cond_state(True), prefix_full=pf,
                       prefix_mask=pm, null_full=nf, cond_enabled=True)
    assert torch.isfinite(pred.D).all()
    for b, D in pred.D_branches.items():
        assert torch.isfinite(D).all(), b
        # Per-branch posteriors are sigmoids and must stay valid probabilities;
        # the *guided* mixture is an extrapolation and may legitimately leave [0,1].
        assert (D >= 0).all() and (D <= 1).all(), b


def test_g2_extreme_guidance_scale_does_not_produce_nan():
    gcfg = GuidanceConfig(cfg_scale=100.0, ag_scale=50.0)
    _, gdn, _, _ = _denoiser(gcfg, bad=True)
    pf, pm, nf = _ctx()
    pred = gdn.denoise(torch.rand(B, S), torch.full((B,), 0.05),
                       sc=gdn.make_self_cond_state(True), prefix_full=pf,
                       prefix_mask=pm, null_full=nf, cond_enabled=True)
    assert torch.isfinite(pred.D).all()


def test_g3_config_validation():
    with pytest.raises(ValueError):
        GuidanceConfig(sg_variant="bogus")
    with pytest.raises(ValueError):
        GuidanceConfig(sg_mf_mode="bogus")
    with pytest.raises(ValueError):
        GuidanceConfig(sg_scale=1.0, sg_delta=0.0)
    with pytest.raises(ValueError):
        GuidanceConfig(cfg_scale=float("nan"))


def test_g4_autoguidance_without_a_bad_model_is_refused():
    cfg = make_cpu_cfg(num_steps=8)
    with pytest.raises(ValueError, match="bad_model"):
        GuidedDenoiser(TinyBinaryDenoiser(S), cfg, GuidanceConfig(ag_scale=2.0))


def test_g5_diagnostics_are_finite_and_populated():
    gcfg = GuidanceConfig(cfg_scale=3.0, ag_scale=2.0, sg_scale=1.0, sg_variant="exact")
    _, gdn, _, _ = _denoiser(gcfg, bad=True, diag=True, per_row=True)
    pf, pm, nf = _ctx()
    pred = gdn.denoise(torch.rand(B, S), torch.full((B,), 0.4),
                       sc=gdn.make_self_cond_state(True), prefix_full=pf,
                       prefix_mask=pm, null_full=nf, cond_enabled=True)
    for key in ("cfg_dir_rms", "ag_dir_rms", "sg_dir_rms", "score_rms",
                "bit_entropy_mean", "p_mean", "frac_p_lt_0.01", "sigma"):
        assert key in pred.diagnostics, key
        assert math.isfinite(pred.diagnostics[key]), key


def test_d9_sg_exact_requires_per_row_sigma():
    """SG-exact packs two noise levels into one batch.

    Pin the contract it relies on. Under `sg_mf_mode="hold"` the analytic
    matched filter is re-attached at the true sigma, so the ONLY thing that can
    produce a non-zero self-guidance direction is the network actually
    responding to the shifted noise level. A model that collapses sigma to a
    scalar therefore yields exactly zero guidance -- silently, with no error.
    CoBit's SDT is per-row (`SigmaEmbedding` uses `sigma.log()[:, None]`); the
    shared harness stand-in is not, which is what makes this failure mode
    concrete.
    """
    pf, pm, nf = _ctx()
    x, sigma = torch.rand(B, S), torch.full((B,), 0.8)
    gcfg = GuidanceConfig(sg_scale=3.0, sg_variant="exact", sg_delta=0.5, sg_mf_mode="hold")
    kw = dict(prefix_full=pf, prefix_mask=pm, null_full=nf, cond_enabled=True)
    free = ~pm

    def _delta(per_row):
        _, g, _, _ = _denoiser(gcfg, per_row=per_row)
        _, ref, _, _ = _denoiser(GuidanceConfig(), per_row=per_row)
        a = g.denoise(x, sigma, sc=g.make_self_cond_state(True), **kw)
        b = ref.denoise(x, sigma, sc=ref.make_self_cond_state(True), **kw)
        return (a.D - b.D)[free].abs().max()

    assert _delta(per_row=True) > 1e-4     # genuine sigma sensitivity -> guidance
    assert _delta(per_row=False) < 1e-6    # sigma ignored -> guidance vanishes


def test_d10_sg_hold_isolates_the_network_from_the_analytic_term():
    """Companion to d9: with `vary`, the same sigma-blind batching produces a
    large spurious direction that comes entirely from the matched filter."""
    pf, pm, nf = _ctx()
    x, sigma = torch.rand(B, S), torch.full((B,), 0.8)
    kw = dict(prefix_full=pf, prefix_mask=pm, null_full=nf, cond_enabled=True)
    free = ~pm

    gcfg = GuidanceConfig(sg_scale=3.0, sg_variant="exact", sg_delta=0.5, sg_mf_mode="vary")
    _, g, _, _ = _denoiser(gcfg, per_row=False)
    _, ref, _, _ = _denoiser(GuidanceConfig(), per_row=False)
    a = g.denoise(x, sigma, sc=g.make_self_cond_state(True), **kw)
    b = ref.denoise(x, sigma, sc=ref.make_self_cond_state(True), **kw)
    assert (a.D - b.D)[free].abs().max() > 1e-2


def test_d11_sg_exact_shifted_sigma_is_capped_to_the_trained_range():
    """SG-exact must not query the model outside the noise range it was trained on.

    sigma*exp(sg_delta) overshoots the top of the schedule near sigma_max
    (exp(0.5) = 1.65x), and a model asked about an unseen noise level does not
    return a meaningfully "worse prediction of the same thing" -- it
    extrapolates. The cap keeps the shifted call in-distribution, and the
    spacing normalisation absorbs the smaller, varying offset that results.
    """
    pf, pm, nf = _ctx()
    kw = dict(prefix_full=pf, prefix_mask=pm, null_full=nf, cond_enabled=True)
    gcfg = GuidanceConfig(sg_scale=1.0, sg_variant="exact", sg_delta=0.5, sg_mf_mode="vary")
    cap = 10.0
    cfg = make_cpu_cfg(self_condition=False, num_steps=8)
    model = PerRowSigmaDenoiser(S, seed=0).eval()

    seen = []

    class Spy(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x, sigma, x0_hat=None):
            seen.append(float(sigma.max()))
            return self.inner(x, sigma, x0_hat)

    gdn = GuidedDenoiser(Spy(model), cfg, gcfg, sigma_hi_cap=cap)

    # Well below the cap: the full nominal offset is available.
    seen.clear()
    gdn.denoise(torch.rand(B, S), torch.full((B,), 1.0),
                sc=gdn.make_self_cond_state(True), **kw)
    assert max(seen) == pytest.approx(1.0 * math.exp(0.5), rel=1e-5)

    # At the cap: the shifted evaluation must not exceed it.
    seen.clear()
    gdn.denoise(torch.rand(B, S), torch.full((B,), cap),
                sc=gdn.make_self_cond_state(True), **kw)
    assert max(seen) <= cap * (1 + 1e-6), max(seen)

    # Just under the cap: shifted level clamped, and still a real correction.
    seen.clear()
    p_capped = gdn.denoise(torch.rand(B, S), torch.full((B,), cap * 0.9),
                           sc=gdn.make_self_cond_state(True), **kw)
    assert max(seen) <= cap * (1 + 1e-6)
    assert torch.isfinite(p_capped.D).all()

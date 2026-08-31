"""Phase-6 integration: every guidance combination, end to end through DDIM.

These drive the real `DDIMSampler.sample` loop (sigma schedule, EDM churn,
self-conditioning carry, prompt clamping, final denoise) rather than the
combinator in isolation, and check the properties that must hold for *any*
configuration before an experiment is worth launching: finite output, valid
probabilities, an intact prompt, determinism under a fixed seed, and a model
evaluation count that matches what the policy should cost.
"""
from __future__ import annotations

import pytest
import torch

from diffusion.continuous.guidance import GuidanceConfig
from diffusion.continuous.processes import ContinuousForwardProcess
from diffusion.continuous.samplers import DDIMSampler
from tests._sampler_harness import make_conditioning, make_cpu_cfg
from tests.test_guidance import PerRowSigmaDenoiser

B, S, NPROMPT, STEPS = 3, 48, 12, 6

# The 12 cells of the factorial design, named as in the experiment plan.
COMBINATIONS = {
    "baseline":        GuidanceConfig(),
    "cfg":             GuidanceConfig(cfg_scale=3.0),
    "ag":              GuidanceConfig(ag_scale=2.0),
    "sg_prev":         GuidanceConfig(sg_scale=1.0, sg_variant="prev"),
    "sg_exact":        GuidanceConfig(sg_scale=1.0, sg_variant="exact"),
    "cfg_ag":          GuidanceConfig(cfg_scale=3.0, ag_scale=2.0),
    "cfg_sg_prev":     GuidanceConfig(cfg_scale=3.0, sg_scale=1.0, sg_variant="prev"),
    "ag_sg_prev":      GuidanceConfig(ag_scale=2.0, sg_scale=1.0, sg_variant="prev"),
    "cfg_ag_sg_prev":  GuidanceConfig(cfg_scale=3.0, ag_scale=2.0, sg_scale=1.0, sg_variant="prev"),
    "cfg_sg_exact":    GuidanceConfig(cfg_scale=3.0, sg_scale=1.0, sg_variant="exact"),
    "ag_sg_exact":     GuidanceConfig(ag_scale=2.0, sg_scale=1.0, sg_variant="exact"),
    "cfg_ag_sg_exact": GuidanceConfig(cfg_scale=3.0, ag_scale=2.0, sg_scale=1.0, sg_variant="exact"),
}

# Branches evaluated per denoise call, per policy (== NFE in units of B rows).
EXPECTED_BRANCHES = {
    "baseline": 1, "cfg": 2, "ag": 2, "sg_prev": 1, "sg_exact": 2,
    "cfg_ag": 4, "cfg_sg_prev": 2, "ag_sg_prev": 2, "cfg_ag_sg_prev": 4,
    "cfg_sg_exact": 4, "ag_sg_exact": 4, "cfg_ag_sg_exact": 8,
}


def _setup(*, self_condition=True, stochastic=False):
    cfg = make_cpu_cfg(self_condition=self_condition, num_steps=STEPS)
    if stochastic:
        from evaluation.tasks._task_common import configure_stochastic
        configure_stochastic(cfg, mode="stochastic", gamma=0.3, num_steps=STEPS)
    good = PerRowSigmaDenoiser(S, seed=0).eval()
    bad = PerRowSigmaDenoiser(S, seed=99).eval()
    return cfg, good, bad, ContinuousForwardProcess(cfg)


def _sample(cfg, good, bad, proc, gcfg, *, seed=0, cond=True, **kw):
    torch.manual_seed(seed)
    sampler = DDIMSampler(good, proc, cfg)
    pf, pm = make_conditioning(B, S, NPROMPT, seed=1) if cond else (None, None)
    x, probs = sampler.sample(
        num_samples=B, seq_len=S,
        conditioning_prefix_full=pf, cond_prefix_mask=pm,
        num_steps=STEPS, schedule="karras",
        guidance=gcfg, bad_model=bad if gcfg.ag_enabled else None,
        return_probs=True, progress=False, ati_eta=0.0,
        **kw,
    )
    return x, probs, pf, pm


@pytest.mark.parametrize("name", list(COMBINATIONS))
def test_combination_runs_and_is_well_formed(name):
    cfg, good, bad, proc = _setup()
    x, probs, pf, pm = _sample(cfg, good, bad, proc, COMBINATIONS[name])

    assert x.shape == (B, S) and probs.shape == (B, S)
    assert torch.isfinite(x).all(), f"{name}: non-finite state"
    assert torch.isfinite(probs).all(), f"{name}: non-finite probabilities"
    # The prompt is clamped, so it must survive the trajectory untouched.
    assert torch.equal(x[pm], pf[pm]), f"{name}: prompt corrupted in x"
    assert torch.equal(probs[pm], pf[pm]), f"{name}: prompt corrupted in probs"


@pytest.mark.parametrize("name", list(COMBINATIONS))
def test_combination_is_deterministic(name):
    cfg, good, bad, proc = _setup()
    a = _sample(cfg, good, bad, proc, COMBINATIONS[name], seed=7)[0]
    b = _sample(cfg, good, bad, proc, COMBINATIONS[name], seed=7)[0]
    assert torch.equal(a, b), f"{name}: not reproducible under a fixed seed"


@pytest.mark.parametrize("name", list(COMBINATIONS))
def test_combination_runs_under_stochastic_churn(name):
    """The headline CoBit eval path uses EDM churn; guidance must survive it."""
    cfg, good, bad, proc = _setup(stochastic=True)
    x, probs, pf, pm = _sample(cfg, good, bad, proc, COMBINATIONS[name], seed=3)
    assert torch.isfinite(x).all() and torch.isfinite(probs).all()
    assert torch.equal(x[pm], pf[pm])


@pytest.mark.parametrize("name", list(COMBINATIONS))
def test_combination_without_self_conditioning(name):
    cfg, good, bad, proc = _setup(self_condition=False)
    x, probs, _, _ = _sample(cfg, good, bad, proc, COMBINATIONS[name], seed=5)
    assert torch.isfinite(x).all() and torch.isfinite(probs).all()


@pytest.mark.parametrize("name", list(COMBINATIONS))
def test_guidance_actually_changes_the_output(name):
    """A guided run that matched the baseline exactly would mean the policy was
    silently dropped -- the failure mode most likely to survive to an HPC sweep."""
    cfg, good, bad, proc = _setup()
    base = _sample(cfg, good, bad, proc, COMBINATIONS["baseline"], seed=2)[0]
    out = _sample(cfg, good, bad, proc, COMBINATIONS[name], seed=2)[0]
    if name == "baseline":
        assert torch.equal(base, out)
    else:
        assert not torch.allclose(base, out, atol=1e-6), f"{name}: had no effect"


@pytest.mark.parametrize("name", list(COMBINATIONS))
def test_model_evaluation_count_matches_the_policy(name):
    """NFE bookkeeping: SG-prev must be free, SG-exact must double the branches."""
    from diffusion.continuous.guidance import GuidedDenoiser

    cfg, good, bad, proc = _setup()
    gcfg = COMBINATIONS[name]
    seen = {}
    orig = GuidedDenoiser.denoise

    seen["n"] = 0

    def spy(self, *a, **k):
        seen["gdn"] = self
        seen["n"] += 1
        return orig(self, *a, **k)

    GuidedDenoiser.denoise = spy
    try:
        _sample(cfg, good, bad, proc, gcfg, seed=1)
    finally:
        GuidedDenoiser.denoise = orig

    gdn = seen["gdn"]
    # The invariant that matters is branches-per-call; how many calls the loop
    # makes is a property of the sigma schedule, so measure it rather than
    # hard-coding a step count.
    n_calls = seen["n"]
    assert n_calls > 0

    # SG-exact is capped to the trained noise range, so at the very top of the
    # schedule there is no headroom for a shifted evaluation and that call
    # costs the un-shifted branch count. Account for those explicitly rather
    # than loosening the assertion.
    branches = EXPECTED_BRANCHES[name]
    gcfg_ = COMBINATIONS[name]
    skipped = gdn.sg_stats["skipped"] if gcfg_.sg_variant == "exact" and gcfg_.sg_enabled else 0
    expected = n_calls * branches - skipped * (branches // 2)
    assert gdn.model_evaluations == expected, (
        name, gdn.model_evaluations, expected, n_calls, branches, skipped)


def test_sg_prev_costs_exactly_what_the_unguided_sampler_costs():
    """The headline efficiency claim, measured through the real sampler."""
    from diffusion.continuous.guidance import GuidedDenoiser

    counts = {}
    orig = GuidedDenoiser.denoise

    for name in ("baseline", "sg_prev", "sg_exact"):
        cfg, good, bad, proc = _setup()
        seen = {}

        def spy(self, *a, **k):
            seen["gdn"] = self
            return orig(self, *a, **k)

        GuidedDenoiser.denoise = spy
        try:
            _sample(cfg, good, bad, proc, COMBINATIONS[name], seed=1)
        finally:
            GuidedDenoiser.denoise = orig
        counts[name] = seen["gdn"].model_evaluations

    # SG-prev is exactly free.
    assert counts["sg_prev"] == counts["baseline"]
    # SG-exact roughly doubles it -- strictly less than 2x because the call at
    # the top of the sigma schedule has no headroom for a shifted evaluation.
    assert counts["baseline"] < counts["sg_exact"] <= 2 * counts["baseline"]


@pytest.mark.parametrize("name", list(COMBINATIONS))
def test_diagnostics_trace_is_emitted(name):
    cfg, good, bad, proc = _setup()
    gcfg = COMBINATIONS[name]
    torch.manual_seed(0)
    sampler = DDIMSampler(good, proc, cfg)
    pf, pm = make_conditioning(B, S, NPROMPT, seed=1)
    x, probs, trace = sampler.sample(
        num_samples=B, seq_len=S,
        conditioning_prefix_full=pf, cond_prefix_mask=pm,
        num_steps=STEPS, schedule="karras",
        guidance=gcfg, bad_model=bad if gcfg.ag_enabled else None,
        return_probs=True, progress=False, ati_eta=0.0,
        collect_diagnostics=True,
    )
    # One record per loop step plus one for the final denoise. The loop length
    # follows the sigma schedule (num_steps sigmas => num_steps-1 steps).
    assert len(trace) == STEPS
    assert all("sigma" in r and "bit_entropy_mean" in r for r in trace)
    if gcfg.cfg_enabled:
        assert all("cfg_dir_rms" in r for r in trace)
    if gcfg.ag_enabled:
        assert all("ag_dir_rms" in r for r in trace)


def test_autoguidance_with_an_identical_bad_model_reproduces_the_baseline():
    """End-to-end version of the algebraic gate: a bad model that IS the good
    model must leave the sampler's trajectory unchanged at any AG scale."""
    cfg, good, _, proc = _setup()
    base = _sample(cfg, good, good, proc, GuidanceConfig(), seed=4)[0]
    for w in (0.5, 2.0, 5.0):
        out = _sample(cfg, good, good, proc, GuidanceConfig(ag_scale=w), seed=4)[0]
        assert torch.allclose(base, out, atol=1e-5), w


def test_unsupported_samplers_refuse_rather_than_ignore():
    """Heun and PC still run the legacy inline CFG block."""
    from diffusion.continuous.samplers import HeunSampler, PredictorCorrectorSampler

    cfg, good, bad, proc = _setup()
    pf, pm = make_conditioning(B, S, NPROMPT, seed=1)
    kw = dict(num_samples=B, seq_len=S, conditioning_prefix_full=pf,
              cond_prefix_mask=pm, num_steps=STEPS, schedule="karras", progress=False)

    for cls in (HeunSampler, PredictorCorrectorSampler):
        with pytest.raises(NotImplementedError):
            cls(good, proc, cfg).sample(guidance=GuidanceConfig(ag_scale=2.0), **kw)
        with pytest.raises(NotImplementedError):
            cls(good, proc, cfg).sample(guidance=GuidanceConfig(sg_scale=1.0), **kw)


def test_sg_prev_reports_when_churn_makes_it_inapplicable():
    """SG-prev needs the cached evaluation to sit at a strictly HIGHER sigma.

    EDM churn can push sigma back up between steps, which leaves no valid
    spacing and silently disables the correction. Without this bookkeeping a
    null self-guidance result under churn would be indistinguishable from
    'self-guidance does not help', so the denoiser counts both outcomes.
    """
    from diffusion.continuous.guidance import GuidedDenoiser

    stats = {}
    orig = GuidedDenoiser.denoise

    def spy(self, *a, **k):
        stats["gdn"] = self
        return orig(self, *a, **k)

    for stochastic in (False, True):
        cfg, good, bad, proc = _setup(stochastic=stochastic)
        GuidedDenoiser.denoise = spy
        try:
            _sample(cfg, good, bad, proc, COMBINATIONS["sg_prev"], seed=1)
        finally:
            GuidedDenoiser.denoise = orig
        s = stats["gdn"].sg_stats
        assert s["applied"] + s["skipped"] > 0, "no SG-prev decisions recorded"
        if not stochastic:
            # Deterministic sampling: sigma decreases monotonically, so every
            # step after the first must have a usable spacing.
            assert s["applied"] > 0
            assert s["skipped"] == 0, s


def test_factorial_confirm_refuses_a_default_operating_point(monkeypatch):
    """An unset GUID_* must not silently build the placeholder grid.

    factorial_confirm is ~18 GPU-hours. If a variable is missing from --export,
    falling back to the defaults (w=3/1.5/0.5, bad=first checkpoint on disk)
    would spend all of it on the wrong operating point and leave result JSONs
    that look entirely normal.
    """
    import pytest
    from experiments.guidance import grids

    for k in ("GUID_CFG_W", "GUID_AG_W", "GUID_SG_W", "GUID_BAD_STEP"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit) as e:
        grids.build("factorial_confirm")
    msg = str(e.value)
    assert "GUID_CFG_W" in msg and "operating point" in msg
    # The message must carry the values the confirmations actually chose.
    assert "GUID_BAD_STEP=350000" in msg


def test_factorial_confirm_uses_the_exported_operating_point(monkeypatch):
    from experiments.guidance import grids

    monkeypatch.setenv("GUID_CFG_W", "12")
    monkeypatch.setenv("GUID_AG_W", "15")
    monkeypatch.setenv("GUID_SG_W", "2")
    monkeypatch.setenv("GUID_BAD_STEP", "350000")
    try:
        cells = grids.build("factorial_confirm")
    except SystemExit:
        import pytest
        pytest.skip("step=000350000.pt not present in this checkout")
    assert len(cells) == 36, "12 conditions x 3 seeds"
    assert {c.limit for c in cells} == {1319}
    assert {c.steps for c in cells} == {512}
    assert {c.seed for c in cells} == {42, 43, 44}
    # Every axis is either off or at exactly the exported operating point.
    assert {c.guidance_scale for c in cells} == {0.0, 12.0}
    assert {c.ag_scale for c in cells} == {0.0, 15.0}
    assert {c.sg_scale for c in cells} == {0.0, 2.0}


def test_solver_control_arms_are_nfe_matched():
    """The Heun rival must be compared at equal model evaluations, not equal steps.

    Heun is 2 evaluations per step, DDIM is 1. If the grid compared both at 512
    steps the Heun arm would silently get twice the compute and the control
    would prove nothing.
    """
    from experiments.guidance import grids

    cells = {c.name.rsplit("_seed", 1)[0]: c for c in grids.build("solver_control")}
    def nfe(c):
        return c.steps * (2 if c.sampler_kind == "heun" else 1)

    assert nfe(cells["solv_ddim512"]) == 512
    assert nfe(cells["solv_sgprev512"]) == 512
    assert nfe(cells["solv_heun256"]) == 512, "the discriminating comparison"
    assert nfe(cells["solv_ddim1024"]) == 1024
    assert nfe(cells["solv_heun512"]) == 1024
    # SG-prev must be the ONLY arm carrying guidance; the rest are pure solvers.
    for name, c in cells.items():
        expected = 2.0 if name == "solv_sgprev512" else 0.0
        assert c.sg_scale == expected, name
        assert c.guidance_scale == 0.0 and c.ag_scale == 0.0, name
    # Heun has no GuidedDenoiser path and cannot emit the diagnostics trace.
    assert cells["solv_heun256"].collect_diagnostics is False
    assert cells["solv_ddim512"].collect_diagnostics is True


def test_solver_and_compute_controls_run_at_confirmation_scale():
    from experiments.guidance import grids

    for name, n_cells in (("solver_control", 15), ("compute_control", 4)):
        cells = grids.build(name)
        assert len(cells) == n_cells, name
        assert {c.limit for c in cells} == {1319}, name
    # compute_control must use seeds DISJOINT from the confirmation grids, or
    # maj@k would vote over samples already counted in the baseline it beats.
    seeds = {c.seed for c in grids.build("compute_control")}
    assert seeds.isdisjoint({42, 43, 44}), seeds


def test_null_ablation_emits_the_override_flag():
    from experiments.guidance import grids

    cells = grids.build("null_ablation")
    strategies = set()
    for c in cells:
        argv = c.cli("/tmp/out")
        assert c.guidance_scale == 12.0, "the null only matters when CFG is on"
        i = argv.index("--null_strategy")
        strategies.add(argv[i + 1])
    assert strategies == {"half", "data_center", "zeros"}


def test_ag_ema_ablation_varies_only_the_bad_model_weights():
    from experiments.guidance import grids

    cells = grids.build("ag_ema")
    assert len(cells) == 4
    assert {c.bad_ema for c in cells} == {0, 1}
    assert {c.ag_scale for c in cells} == {15.0}
    for c in cells:
        argv = c.cli("/tmp/out")
        assert argv[argv.index("--bad_ema") + 1] == str(c.bad_ema)

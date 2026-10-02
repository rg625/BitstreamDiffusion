"""
CPU tests for the compute-allocation diagnostic hooks in DDIMSampler (PLAN §10).

Run from the repo root:
    python -m pytest -q experiments/compute_allocation/tests

Uses a tiny SDT (85k params, 8 tokens x 15 bits) with the production LM1B
config otherwise, the real LM1B entropy tables, 33 sigma points (32 steps),
and segments of 8 steps (4 segments).
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from evaluation.utils import load_config  # noqa: E402
from models import create_model  # noqa: E402
from diffusion.continuous.samplers import DDIMSampler, _churn_step_seed  # noqa: E402

CONFIG = REPO / "configs/lm1b/continuous/eval/rate_eval_seeds.py"
ENTROPY_DIR = str(REPO / "assets/entropy_tables/lm1b")
NUM_STEPS = 33  # sigma points -> 32 Euler steps
SEG_LEN = 8
B = 6


def _tiny_cfg(stochastic: bool = True):
    cfg = load_config(str(CONFIG))
    cfg.device = "cpu"
    m = cfg.model
    m.use_flash_attn = False
    m.embed_dim, m.dim_ff, m.n_blocks, m.n_heads = 32, 64, 1, 2
    m.head_hidden, m.head_embed_dim = 16, 16
    m.content_dim_discrete = m.content_dim_continuous = 16
    m.dropout = 0.0
    cfg.data.sequence_len_tokens = 8
    cfg.data.sequence_len = 8 * 15
    st = cfg.evaluation.stochastic
    st.enabled = bool(stochastic)
    st.s_churn = 0.2 * NUM_STEPS if stochastic else 0.0
    st.s_noise = 1.003
    st.window_mode = "entropy_cdf"
    st.entropy_quantile_lo = 0.0
    st.entropy_quantile_hi = 1.0
    st.s_tmin = None
    st.s_tmax = None
    st.entropy_fallback = "deterministic"
    return cfg


class CountingModel(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.calls = 0

    def forward(self, *a, **k):
        self.calls += 1
        return self.inner(*a, **k)


@pytest.fixture(scope="module")
def tiny():
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = create_model(cfg).eval()
    # zero-initialised layers make a fresh SDT ignore its self-conditioning input;
    # perturb every weight so all input paths are live
    with torch.no_grad():
        for prm in model.parameters():
            prm.add_(0.05 * torch.randn_like(prm))
    return cfg, model


@pytest.fixture(scope="module")
def orig_samplers_module(tmp_path_factory):
    """The sampler module exactly as on `main` (before this branch's changes)."""
    src = subprocess.check_output(
        ["git", "show", "main:diffusion/continuous/samplers.py"], cwd=REPO
    )
    p = tmp_path_factory.mktemp("orig") / "orig_samplers.py"
    p.write_bytes(src)
    spec = importlib.util.spec_from_file_location("orig_samplers", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["orig_samplers"] = mod  # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


def _kw(**extra):
    kw = dict(
        schedule="entropic",
        num_steps=NUM_STEPS,
        entropy_run_dir=ENTROPY_DIR,
        sigma_min_override=0.08,
        sigma_max_override=None,
        sc_refresh_mode="carry",
        ati_eta=0.0,
        return_probs=True,
        progress=False,
    )
    kw.update(extra)
    return kw


def _run(sampler, seed, S, **extra):
    torch.manual_seed(seed)
    return sampler.sample(B, S, **_kw(**extra))


def _eq(a, b):
    return torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


# ---------------------------------------------------------------------------
# §10 test 1: no overrides -> identical to the current (main) sampler
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stochastic", [True, False])
@pytest.mark.parametrize("sc_mode", ["carry", "refined"])
def test_default_bit_identical_to_main(tiny, orig_samplers_module, stochastic, sc_mode):
    _, model = tiny
    cfg = _tiny_cfg(stochastic)
    S = cfg.data.sequence_len
    new = DDIMSampler(model, None, cfg)
    old = orig_samplers_module.DDIMSampler(model, None, cfg)
    a = _run(new, 123, S, sc_refresh_mode=sc_mode)
    b = _run(old, 123, S, sc_refresh_mode=sc_mode)
    assert _eq(a, b)


# ---------------------------------------------------------------------------
# §10 test 2: null replay from a saved state matches the uninterrupted run
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("churn_seed", [7, None])  # seeded per-step noise, and legacy global stream
def test_null_replay_from_saved_state(tiny, churn_seed):
    cfg, model = tiny
    S = cfg.data.sequence_len
    smp = DDIMSampler(model, None, cfg)
    states = {}
    ref = _run(
        smp, 5, S,
        churn_noise_seed=churn_seed,
        save_states_at=[0, 8, 16, 24, NUM_STEPS - 1],
        state_hook=lambda st: states.__setitem__(st["i"], st),
        segment_len=SEG_LEN,
    )
    assert sorted(states) == [0, 8, 16, 24, NUM_STEPS - 1]
    for i, st in states.items():
        torch.manual_seed(999)  # global RNG must not matter (restored / unused)
        rep = smp.sample(B, S, **_kw(churn_noise_seed=churn_seed, init_state=st, segment_len=SEG_LEN))
        assert _eq(rep, ref), f"replay from step {i} diverged"


# ---------------------------------------------------------------------------
# §10 test 3: refine with 1 sub-step is identical to the reference
# ---------------------------------------------------------------------------
def test_refine_one_substep_identical(tiny):
    cfg, model = tiny
    S = cfg.data.sequence_len
    smp = DDIMSampler(model, None, cfg)
    ref = _run(smp, 11, S, churn_noise_seed=3, segment_len=SEG_LEN)
    for s in range(4):
        br = _run(
            smp, 11, S, churn_noise_seed=3, segment_len=SEG_LEN,
            segment_overrides={s: {"substeps": 1, "gamma": None}},
        )
        assert _eq(br, ref)


# ---------------------------------------------------------------------------
# §10 test 4: churn noise for step i does not depend on earlier sub-steps
# ---------------------------------------------------------------------------
def _collect_eps(smp, S, **extra):
    eps = {}

    def cb(**info):
        if info["final"] or info["eps"] is None:
            return
        assert info["i"] not in eps, "churn applied twice in one original step"
        assert info["sub"] == 0, "churn must be applied at the first sub-step only"
        eps[info["i"]] = info["eps"].clone()

    out = _run(smp, 21, S, step_callback=cb, segment_len=SEG_LEN, **extra)
    return eps, out


def test_churn_noise_independent_of_substeps(tiny):
    cfg, model = tiny
    S = cfg.data.sequence_len
    smp = DDIMSampler(model, None, cfg)
    ref_eps, _ = _collect_eps(smp, S, churn_noise_seed=4)
    assert sorted(ref_eps) == list(range(NUM_STEPS - 1))
    for s in range(4):
        br_eps, _ = _collect_eps(
            smp, S, churn_noise_seed=4, segment_overrides={s: {"substeps": 4}}
        )
        assert sorted(br_eps) == sorted(ref_eps)
        for i in ref_eps:
            assert torch.equal(br_eps[i], ref_eps[i]), f"eps at step {i} shifted by refine[{s}]"
    # also independent of the batch-global RNG state
    torch.manual_seed(12345)
    assert torch.equal(
        torch.randn(3, generator=torch.Generator().manual_seed(_churn_step_seed(4, 10))),
        torch.randn(3, generator=torch.Generator().manual_seed(_churn_step_seed(4, 10))),
    )
    assert _churn_step_seed(4, 10) != _churn_step_seed(4, 11) != _churn_step_seed(5, 10)


# ---------------------------------------------------------------------------
# Additional tests
# ---------------------------------------------------------------------------
def test_substep_sigmas_even_in_entropy_quantile(tiny):
    cfg, model = tiny
    smp = DDIMSampler(model, None, cfg)
    sch = smp.sigmas
    sigmas = sch.prepare(schedule="entropic", num_steps=NUM_STEPS, entropy_run_dir=ENTROPY_DIR,
                         sigma_min_override=0.08)
    _, cdf, sig_tab = sch._load_entropy_tables(entropy_run_dir=ENTROPY_DIR)
    F = cdf.float().clone()
    F[-1] = 1.0
    for i in range(NUM_STEPS - 1):
        sub = sch.entropic_substep_sigmas(sigmas, i, 4, entropy_run_dir=ENTROPY_DIR,
                                          sigma_min_override=0.08)
        assert sub.numel() == 5
        assert sub[0].item() == sigmas[i].item() and sub[-1].item() == sigmas[i + 1].item()
        assert torch.all(sub[1:] < sub[:-1]), f"sub-sigmas not decreasing at step {i}"
        # quantiles of the interior points are evenly spaced between the endpoints'
        q = sch._interp1d_monotone(sig_tab.float(), F, sub)
        if 0 < i < NUM_STEPS - 2:  # endpoints 0 / N-1 are pinned to table bounds
            d = q[1:] - q[:-1]
            assert torch.allclose(d, d.mean().expand_as(d), rtol=1e-3, atol=1e-6), (i, d)
    # one sub-step returns the original step
    assert torch.equal(sch.entropic_substep_sigmas(sigmas, 3, 1), sigmas[3:5])


def test_refine_nfe_and_nochurn_scope(tiny):
    cfg, model = tiny
    S = cfg.data.sequence_len
    cm = CountingModel(model)
    smp = DDIMSampler(cm, None, cfg)
    _run(smp, 1, S, churn_noise_seed=2, segment_len=SEG_LEN)
    assert cm.calls == (NUM_STEPS - 1) + 1  # carry mode: 1 call/step + final denoise
    cm.calls = 0
    _run(smp, 1, S, churn_noise_seed=2, segment_len=SEG_LEN, segment_overrides={1: {"substeps": 4}})
    assert cm.calls == (NUM_STEPS - 1) + 1 + 3 * SEG_LEN

    ref_eps, _ = _collect_eps(smp, S, churn_noise_seed=2)
    nc_eps, _ = _collect_eps(smp, S, churn_noise_seed=2, segment_overrides={2: {"gamma": 0.0}})
    assert sorted(nc_eps) == [i for i in range(NUM_STEPS - 1) if not (16 <= i < 24)]
    for i in nc_eps:
        assert torch.equal(nc_eps[i], ref_eps[i])


@pytest.mark.parametrize("kind", ["refine", "nochurn"])
def test_branch_from_saved_state_equals_branch_from_scratch(tiny, kind):
    """A probe started at segment s's saved state equals the same override run from step 0."""
    cfg, model = tiny
    S = cfg.data.sequence_len
    smp = DDIMSampler(model, None, cfg)
    states = {}
    _run(smp, 8, S, churn_noise_seed=9, segment_len=SEG_LEN, save_states_at=[0, 8, 16, 24],
         state_hook=lambda st: states.__setitem__(st["i"], st))
    ov = {"substeps": 4} if kind == "refine" else {"gamma": 0.0}
    for s in range(4):
        full = _run(smp, 8, S, churn_noise_seed=9, segment_len=SEG_LEN, segment_overrides={s: ov})
        torch.manual_seed(0)
        part = smp.sample(B, S, **_kw(churn_noise_seed=9, segment_len=SEG_LEN,
                                      segment_overrides={s: ov}, init_state=states[SEG_LEN * s]))
        assert _eq(full, part), f"{kind}[{s}] from saved state differs"
        ref = _run(smp, 8, S, churn_noise_seed=9, segment_len=SEG_LEN)
        assert not torch.equal(full[1], ref[1]), f"{kind}[{s}] had no effect at all"


# ---------------------------------------------------------------------------
# sc_refresh override (experiments/adaptive_compute): extra self-conditioning passes
# ---------------------------------------------------------------------------
def test_sc_refresh_one_everywhere_equals_refined_mode(tiny):
    cfg, model = tiny
    S = cfg.data.sequence_len
    smp = DDIMSampler(model, None, cfg)
    allseg = {s: {"sc_refresh": 1} for s in range(4)}
    a = _run(smp, 31, S, churn_noise_seed=6, segment_len=SEG_LEN, segment_overrides=allseg)
    b = _run(smp, 31, S, churn_noise_seed=6, segment_len=SEG_LEN, sc_refresh_mode="refined")
    assert _eq(a, b)


def test_sc_refresh_zero_is_noop_and_counts_nfe(tiny):
    cfg, model = tiny
    S = cfg.data.sequence_len
    cm = CountingModel(model)
    smp = DDIMSampler(cm, None, cfg)
    ref = _run(smp, 2, S, churn_noise_seed=1, segment_len=SEG_LEN)
    z = _run(smp, 2, S, churn_noise_seed=1, segment_len=SEG_LEN, segment_overrides={0: {"sc_refresh": 0}})
    assert _eq(ref, z)
    for c in (1, 2, 4):
        cm.calls = 0
        out = _run(smp, 2, S, churn_noise_seed=1, segment_len=SEG_LEN, segment_overrides={0: {"sc_refresh": c}})
        assert cm.calls == (NUM_STEPS - 1) + 1 + c * SEG_LEN
        assert not torch.equal(out[0], ref[0])  # tiny model: D saturates at low sigma, so test at high sigma (segment 0)


def test_sc_refresh_keeps_churn_noise_and_prefix(tiny):
    cfg, model = tiny
    S = cfg.data.sequence_len
    smp = DDIMSampler(model, None, cfg)
    ref_eps, _ = _collect_eps(smp, S, churn_noise_seed=8)
    br_eps, _ = _collect_eps(smp, S, churn_noise_seed=8, segment_overrides={1: {"sc_refresh": 4}})
    assert sorted(br_eps) == sorted(ref_eps)
    for i in ref_eps:
        assert torch.equal(br_eps[i], ref_eps[i])
    # trajectory before segment 1 is untouched: identical saved state at step 8
    sa, sb = {}, {}
    _run(smp, 3, S, churn_noise_seed=8, segment_len=SEG_LEN, save_states_at=[8],
         state_hook=lambda st: sa.__setitem__(st["i"], st))
    _run(smp, 3, S, churn_noise_seed=8, segment_len=SEG_LEN, save_states_at=[8],
         segment_overrides={1: {"sc_refresh": 4}}, state_hook=lambda st: sb.__setitem__(st["i"], st))
    assert torch.equal(sa[8]["x"], sb[8]["x"]) and torch.equal(sa[8]["x0_hat"], sb[8]["x0_hat"])

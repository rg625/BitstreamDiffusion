"""
CPU checks for the global score temperature (port of Isambard cpu-reanalysis 02ef262) in DDIMSampler and in
diag_sampler's stemp<tau>[u] variants.

Run from the repo root:
    python -m pytest -q experiments/denoising_mse/tinygsm/test_score_temp.py

ST_SAMPLERS / ST_DIAG point the tests at staged copies of samplers.py / diag_sampler.py (default: the repo files).
The reference for "unchanged at tau = 1" is samplers.py as committed at HEAD. Uses a tiny SDT with the TinyGSM
CoBit-S config otherwise (8 tokens x 16 bits, prompt of 3 tokens), the run's real entropy tables and its entropic
grid (whose last step is the oversized one), EDM churn gamma 0.31 as diagnostics.py.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import common as C  # noqa: E402
from gsm8k_eval import configure_stochastic  # noqa: E402
from models import create_model  # noqa: E402
from diffusion.continuous.processes import ContinuousForwardProcess  # noqa: E402

RUN_DIR = str(REPO / C.MODELS["S"]["run_dir"])
T_TOK, P_TOK = 8, 3
S_BITS = T_TOK * C.BPT
B = 4
STEPS = 24
GAMMA = 0.31
CHURN = 12345

torch.set_num_threads(4)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def new():
    return _load("st_new_samplers", os.environ.get("ST_SAMPLERS", REPO / "diffusion/continuous/samplers.py"))


@pytest.fixture(scope="module")
def orig(tmp_path_factory):
    src = subprocess.check_output(["git", "show", "HEAD:diffusion/continuous/samplers.py"], cwd=REPO)
    p = tmp_path_factory.mktemp("orig") / "orig_samplers.py"
    p.write_bytes(src)
    return _load("st_orig_samplers", p)


@pytest.fixture(scope="module")
def DS():
    return _load("st_diag_sampler", os.environ.get("ST_DIAG", HERE / "diag_sampler.py"))


@pytest.fixture(scope="module")
def tiny():
    cfg = C.load_cfg("S", rope_fix=False)
    cfg.device = "cpu"
    m = cfg.model
    m.use_flash_attn = False
    m.embed_dim, m.dim_ff, m.n_blocks, m.n_heads = 32, 64, 1, 2
    m.head_hidden, m.head_embed_dim = 16, 16
    m.content_dim_discrete = m.content_dim_continuous = 16
    m.dropout = 0.0
    cfg.data.sequence_len_tokens = T_TOK
    cfg.data.sequence_len = S_BITS
    configure_stochastic(cfg, GAMMA, STEPS)
    torch.manual_seed(0)
    model = create_model(cfg).eval()
    # zero-initialised layers make a fresh SDT ignore its self-conditioning input; perturb every weight
    with torch.no_grad():
        for prm in model.parameters():
            prm.add_(0.05 * torch.randn_like(prm))
    torch.manual_seed(1)
    ids = torch.randint(0, 4000, (B, T_TOK))
    x0 = C.ids_to_bits(ids).float()
    pm = C.tok_mask_to_bits(torch.arange(T_TOK)[None].expand(B, T_TOK) < P_TOK)
    return cfg, model, x0, pm


def _sigmas(mod, cfg, model):
    probe = mod.DDIMSampler(model, ContinuousForwardProcess(cfg), cfg)
    sig = probe.sigmas.prepare(schedule="entropic", num_steps=STEPS, entropy_run_dir=RUN_DIR)
    return sig, probe.sigmas.resolve_stochastic_cfg(entropy_run_dir=RUN_DIR)


def _x_init(x0, pm, sig0):
    g = torch.Generator().manual_seed(7)
    return torch.where(pm, x0, torch.randn(x0.shape, generator=g) * sig0 + 0.5)


def _ddim(mod, tiny, steps_out=None, **kw):
    cfg, model, x0, pm = tiny
    sampler = mod.DDIMSampler(model, ContinuousForwardProcess(cfg), cfg)
    sig, _ = _sigmas(mod, cfg, model)
    x0_hat = torch.where(pm, x0, torch.zeros_like(x0))
    cb = None
    if steps_out is not None:
        def cb(**a):
            if a["final"]:  # the readout D(x, sigma_min) after the loop, not an integration step
                return
            steps_out.append({k: (v.clone() if torch.is_tensor(v) else v) for k, v in a.items()})
    return sampler.sample(
        B, S_BITS, conditioning_prefix_full=x0, cond_prefix_mask=pm, guidance_scale=0.0,
        schedule="entropic", num_steps=STEPS, entropy_run_dir=RUN_DIR, sc_refresh_mode="carry", ati_eta=0.0,
        return_probs=True, progress=False, churn_noise_seed=CHURN,
        init_state=dict(i=0, x=_x_init(x0, pm, sig[0]), x0_hat=x0_hat, sigmas=sig), step_callback=cb,
        **{"score_temp_mode": "global", **kw})


# ---------------------------------------------------------------------------
# DDIMSampler
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kw", [{}, dict(score_temp_tau=1.0), dict(score_temp_tau=1.0, score_temp_final="unsharpened")])
def test_ddim_tau1_bit_identical_to_head(new, orig, tiny, kw):
    a = _ddim(orig, tiny)
    b = _ddim(new, tiny, **kw)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@pytest.mark.parametrize("final", ["sharpened", "unsharpened"])
def test_ddim_step_is_drift_over_tau(new, tiny, final):
    """Every step: x_next = x_state + h (x_state - D) / (sigma_state tau); the last step without tau if unsharpened.
    Deterministic run (churn off) so that x_state is the previous step's x."""
    cfg = tiny[0]
    st = cfg.evaluation.stochastic
    saved = st.enabled
    st.enabled = False
    try:
        tau, steps = 0.3, []
        _ddim(new, tiny, steps, score_temp_tau=tau, score_temp_final=final)
    finally:
        st.enabled = saved
    pm = tiny[3]
    assert len(steps) == STEPS - 1  # num_steps counts sigma points
    for j in range(1, len(steps)):
        s, prev = steps[j], steps[j - 1]["x"]
        last = j == len(steps) - 1
        t = 1.0 if (last and final == "unsharpened") else tau
        d = (prev - s["D"]) / s["sigma"]
        d[pm] = 0.0
        want = torch.where(pm, prev, prev + (s["sigma_next"] - s["sigma"]) * d / t)
        torch.testing.assert_close(s["x"], want, rtol=1e-5, atol=1e-5)


def test_ddim_unsharpened_differs_only_at_last_step(new, tiny):
    a, b = [], []
    _ddim(new, tiny, a, score_temp_tau=0.3, score_temp_final="sharpened")
    _ddim(new, tiny, b, score_temp_tau=0.3, score_temp_final="unsharpened")
    for sa, sb in zip(a[:-1], b[:-1]):
        assert torch.equal(sa["x"], sb["x"])
    assert not torch.equal(a[-1]["x"], b[-1]["x"])


def test_ddim_unsharpened_last_substep_only(new, tiny):
    """With refine substeps in the last segment, only the final substep of the final step is untempered."""
    ov = {(STEPS - 2) // 8: dict(substeps=2)}  # segment holding the last step, i = STEPS - 2
    a, b = [], []
    _ddim(new, tiny, a, score_temp_tau=0.3, segment_overrides=ov, segment_len=8)
    _ddim(new, tiny, b, score_temp_tau=0.3, score_temp_final="unsharpened", segment_overrides=ov, segment_len=8)
    assert a[-1]["n_sub"] == 2 and a[-1]["sub"] == 1
    for sa, sb in zip(a[:-1], b[:-1]):
        assert torch.equal(sa["x"], sb["x"])
    assert not torch.equal(a[-1]["x"], b[-1]["x"])


def test_ddim_tau_changes_output(new, tiny):
    assert not torch.equal(_ddim(new, tiny)[0], _ddim(new, tiny, score_temp_tau=0.5)[0])


@pytest.mark.parametrize("kw", [dict(score_temp_tau=0.0), dict(score_temp_tau=-1.0), dict(score_temp_final="last")])
def test_ddim_rejects_bad_args(new, tiny, kw):
    with pytest.raises(ValueError):
        _ddim(new, tiny, **kw)


# ---------------------------------------------------------------------------
# diag_sampler
# ---------------------------------------------------------------------------
def test_parse_variant(DS):
    v = DS.parse_variant("stemp0.3u")
    assert (v.st_tau, v.st_final) == (0.3, "unsharpened")
    v = DS.parse_variant("stemp0.5")
    assert (v.st_tau, v.st_final) == (0.5, "sharpened")
    assert DS.parse_variant("stagger0.5").stagger == 0.5 and DS.parse_variant("stagger0.5").st_tau is None
    assert DS.parse_variant("base").st_tau is None
    with pytest.raises(ValueError):
        DS.parse_variant("stemp0")


def _diag(DS, new, tiny, name):
    cfg, model, x0, pm = tiny
    sig, stc = _sigmas(new, cfg, model)
    return DS.sample(model, cfg, stc, sig, _x_init(x0, pm, sig[0]), x0, pm, [P_TOK] * B, CHURN,
                     DS.parse_variant(name))[0]


@pytest.mark.parametrize("name,kw", [
    ("base", {}),
    ("stemp1.0", dict(score_temp_tau=1.0)),
    ("stemp0.5", dict(score_temp_tau=0.5)),
    ("stemp0.3u", dict(score_temp_tau=0.3, score_temp_final="unsharpened")),
    ("stemp0.2u", dict(score_temp_tau=0.2, score_temp_final="unsharpened")),
])
def test_diag_matches_ddim(DS, new, tiny, name, kw):
    """diag_sampler's stemp variants reproduce DDIMSampler(sc_refresh_mode="carry", score_temp_*) exactly."""
    want = _ddim(new, tiny, **kw)[1]
    got = _diag(DS, new, tiny, name)
    assert torch.equal(got, want)


def test_diag_stagger_tau1_is_noop(DS, new, tiny):
    import dataclasses

    v0 = DS.parse_variant("stagger0.5")
    v1 = dataclasses.replace(v0, st_tau=1.0)
    cfg, model, x0, pm = tiny
    sig, stc = _sigmas(new, cfg, model)
    xi = _x_init(x0, pm, sig[0])
    a = DS.sample(model, cfg, stc, sig, xi, x0, pm, [P_TOK] * B, CHURN, v0)[0]
    b = DS.sample(model, cfg, stc, sig, xi, x0, pm, [P_TOK] * B, CHURN, v1)[0]
    c = DS.sample(model, cfg, stc, sig, xi, x0, pm, [P_TOK] * B, CHURN, dataclasses.replace(v0, st_tau=0.5))[0]
    assert torch.equal(a, b) and not torch.equal(a, c)

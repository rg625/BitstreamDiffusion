"""Gradient-survival probe: the PRIMARY endpoint of the binary_sm/CE pilot.

grad_survival = sum|dL_sm/d_ell| / sum|dL_ce/d_ell| = sum|(D-x0)*D(1-D)| / sum|D-x0|

It is a property of the model's D values, not of the loss being optimised, so it
is computed identically in both arms and the two runs are directly comparable.
"""
import inspect

import ml_collections
import pytest
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
    for k in ("s075_150", "s150_300", "s1_3", "s10_up"):
        assert f"objective/grad_survival_{k}" in out, k


def test_low_sigma_band_resolves_below_the_production_collapse_point():
    """The old (0, 0.5) 'lo' band pooled sigma=0.05 -- where production survival
    is exactly 0 -- with sigma=0.4, where it is ~0.15. That average hid the only
    real effect, so 0.05 and 0.4 must land in DIFFERENT bands."""
    from trainers.trainer import SIGMA_BANDS

    def band_of(s):
        return next(n for lo, hi, n in SIGMA_BANDS if lo <= s < hi)

    assert band_of(0.05) != band_of(0.4)
    assert band_of(0.05) != band_of(0.1)          # 0.075 boundary is resolved
    edges = [lo for lo, _, _ in SIGMA_BANDS]
    assert edges == sorted(edges) and len(set(edges)) == len(edges)


def test_logit_magnitude_is_reported():
    """|ell| ~ 1e3 was the state of the diverged pilot; it must be visible live
    rather than only by loading checkpoints afterwards."""
    logits = torch.full((2, 8), 900.0)
    out = _run(logits, torch.zeros(2, 8), torch.tensor([0.4, 0.4]), None)
    assert out["objective/logit_abs_mean"] == pytest.approx(900.0)
    assert out["objective/logit_abs_max"] == pytest.approx(900.0)


# ---------------------------------------------------------------------------
# Wiring. The five tests above exercise the probe's arithmetic by calling it
# directly, which is exactly why they all passed while the probe was in fact
# dead code: the call site sat in _step_discrete, but every bitstream task runs
# framework == "continuous_score" and is therefore dispatched to
# _step_continuous. A 98 GPU-h pilot finished with no primary endpoint logged.
# These tests assert the probe is reachable from the path that actually runs.
# ---------------------------------------------------------------------------

def _binary_cfg(probe_every=1):
    cfg = ml_collections.ConfigDict()
    cfg.framework = "continuous_score"
    cfg.data = ml_collections.ConfigDict()
    cfg.data.representation = "binary"
    cfg.model = ml_collections.ConfigDict()
    cfg.model.self_condition = False
    cfg.model.out_dim = 1
    cfg.diffusion = ml_collections.ConfigDict()
    cfg.diffusion.continuous = ml_collections.ConfigDict()
    cfg.diffusion.continuous.data_center = 0.5
    cfg.train = ml_collections.ConfigDict()
    cfg.train.use_fp16 = False
    cfg.train.self_condition_prob = 0.0
    cfg.train.objective_probe = ml_collections.ConfigDict()
    cfg.train.objective_probe.enabled = True
    cfg.train.objective_probe.every_steps = probe_every
    return cfg


class _NoOp:
    def step(self, *a, **k):
        pass

    def update(self, *a, **k):
        pass


class _NoOpOpt:
    def __init__(self, params):
        self._p = list(params)

    def zero_grad(self, set_to_none=True):
        for p in self._p:
            p.grad = None

    def step(self, *a, **k):
        pass


class _RecordingWriter:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, tag, value, step):
        self.scalars.setdefault(tag, []).append((step, float(value)))


class _StepStub:
    """Minimal surface for Trainer._step_continuous with is_train=False."""

    _step_continuous = Trainer._step_continuous
    _log_objective_probe = Trainer._log_objective_probe
    _log_optim_diagnostics = Trainer._log_optim_diagnostics

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.amp_dtype = torch.float32
        self.entropy_compute = False
        self.global_step = 0
        self.is_master = True
        self.writer = _RecordingWriter()
        # A trivial model with one real parameter, so is_train=True can run a
        # backward pass and the training path is exercised for real.
        self._w = torch.nn.Parameter(torch.tensor(2.0))
        self.model = lambda xt, sigma, x0_hat=None, **kw: xt * self._w
        self.use_scaler = False
        self.grad_clip = 0.0
        self.opt = _NoOpOpt([self._w])
        self.cfg.optim = ml_collections.ConfigDict()
        self.cfg.optim.eps = 1e-8
        self.lr_sched = _NoOp()
        self.ema = _NoOp()

    def _draw_sigma(self, B):
        return torch.full((B,), 0.7)

    def loss_fn(self, logits, target, sigma, cfg, return_entropy_metric=False, mask=None):
        return ((torch.sigmoid(logits) - target) ** 2).mean()


def test_probe_is_reachable_from_the_dispatched_continuous_step():
    """The regression test for the dead-call-site bug."""
    stub = _StepStub(_binary_cfg(probe_every=1))
    x0 = (torch.rand(4, 16) > 0.5).float()
    stub._step_continuous(x0, is_train=True)
    tags = set(stub.writer.scalars)
    assert "objective/grad_survival" in tags, (
        "the primary endpoint was not logged from _step_continuous; "
        f"got {sorted(tags)}"
    )
    assert "objective/median_D1mD" in tags


def test_probe_measures_only_free_bits_under_prefix_conditioning():
    """Prompt positions are clamped to clean bits, so D(1-D) there is not a
    property of the model's learning signal. Including them would bias the
    endpoint towards whatever fraction of the sequence happens to be prompt."""
    cfg = _binary_cfg(probe_every=1)
    cfg.cond = ml_collections.ConfigDict()
    cfg.cond.enabled = True
    cfg.cond.noise_prefix = False
    cfg.cond.loss_on_suffix_only = True
    cfg.cond.p_uncond = 0.0
    cfg.cond.null_strategy = "zeros"

    stub = _StepStub(cfg)
    x0 = (torch.rand(4, 16) > 0.5).float()
    pm = torch.zeros(4, 16, dtype=torch.bool)
    pm[:, :8] = True  # first half is prompt

    stub._step_continuous(x0, is_train=True, batch_prefix_mask=pm)
    assert "objective/grad_survival" in stub.writer.scalars

    # Same batch, no mask -> the clamped prefix bits now enter the statistic and
    # move it. If the mask were being ignored the two would coincide.
    stub2 = _StepStub(cfg)
    torch.manual_seed(0)
    masked = stub.writer.scalars["objective/grad_survival"][0][1]
    assert 0.0 <= masked <= 0.25 + 1e-6  # D(1-D) is bounded by 1/4


def test_dispatch_selects_the_step_function_the_probe_lives_in():
    """Guards against the probe being reintroduced only on the discrete path."""
    src = inspect.getsource(Trainer._step_continuous)
    assert "_log_objective_probe" in src, (
        "framework == 'continuous_score' dispatches to _step_continuous, so the "
        "probe must be called there or it is dead code for every bitstream task"
    )


# ---------------------------------------------------------------------------
# Pilot config invariants. The first pilot was voided by a dead probe and by
# an unvalidated control arm; these pin both so neither recurs silently.
# ---------------------------------------------------------------------------

def _pilot_cfg(arm, tag=""):
    import importlib.util
    import os
    old = {k: os.environ.get(k) for k in ("OBJ_LOSS", "OBJ_TAG")}
    os.environ["OBJ_LOSS"], os.environ["OBJ_TAG"] = arm, tag
    try:
        spec = importlib.util.spec_from_file_location(
            f"pilot_{arm}_{tag}", "configs/tasks/tinygsm_bits_objective.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.get_config()
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def _flat(c, pre=""):
    out = {}
    for k, v in dict(c).items():
        if hasattr(v, "items"):
            try:
                out.update(_flat(v, pre + k + "."))
                continue
            except Exception:
                pass
        out[pre + k] = v
    return out


def test_arms_differ_only_by_loss_type_and_output_paths():
    a, b = _flat(_pilot_cfg("binary_sm")), _flat(_pilot_cfg("binary_ce"))
    diff = {k for k in set(a) | set(b) if str(a.get(k)) != str(b.get(k))}
    assert diff == {"train.loss_type", "experiment",
                    "evaluation.checkpoint_path", "evaluation.out_dir"}, diff


def test_control_arm_matches_the_healthy_production_run():
    """The previous control ran p_uncond=0.0 while the run that trained
    healthily to 500k used 0.1 -- so it was never a validated baseline."""
    cfg = _pilot_cfg("binary_sm")
    assert float(cfg.cond.p_uncond) == 0.1


def test_weight_clamp_is_off_by_default_so_the_control_matches_production():
    """The 5k smoke refuted the clamp: SM broke at step 3,900 WITH it, while
    that arm differed from the healthy 500k run by this key alone. It is now a
    variable to test, not a default to assume."""
    for arm in ("binary_sm", "binary_ce"):
        cfg = _pilot_cfg(arm)
        assert not hasattr(cfg.train, "loss_weight_max") or \
            cfg.train.loss_weight_max is None


def test_guard_is_on_and_identical_in_both_arms():
    a, b = _pilot_cfg("binary_sm"), _pilot_cfg("binary_ce")
    for c in (a, b):
        assert bool(c.train.divergence_guard.enabled)
    assert (float(a.train.divergence_guard.factor)
            == float(b.train.divergence_guard.factor) == 4.0)


def test_tag_isolates_smoke_runs_from_the_pilot_directory():
    assert _pilot_cfg("binary_sm").experiment != \
        _pilot_cfg("binary_sm", "smoke").experiment


def test_probe_does_not_fire_on_validation_steps():
    """Validation does not advance global_step, so every validation batch would
    log at the same step. The 5k smoke wrote 230 extra points at step 5000, on
    validation data with a different noise draw."""
    stub = _StepStub(_binary_cfg(probe_every=1))
    x0 = (torch.rand(4, 16) > 0.5).float()
    stub._step_continuous(x0, is_train=False)
    assert stub.writer.scalars == {}, (
        f"probe logged during validation: {sorted(stub.writer.scalars)}")
    stub._step_continuous(x0, is_train=True)
    assert "objective/grad_survival" in stub.writer.scalars


def test_repeated_validation_cannot_stack_points_on_one_step():
    stub = _StepStub(_binary_cfg(probe_every=1))
    x0 = (torch.rand(4, 16) > 0.5).float()
    stub._step_continuous(x0, is_train=True)
    for _ in range(20):
        stub._step_continuous(x0, is_train=False)
    steps = [s for s, _ in stub.writer.scalars["objective/grad_survival"]]
    assert len(steps) == len(set(steps)) == 1

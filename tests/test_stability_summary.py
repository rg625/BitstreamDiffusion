"""time_to_divergence is the PRIMARY endpoint of the stability study, so it must
agree with the live divergence guard rather than being a second, looser rule."""
import importlib.util

import pytest

spec = importlib.util.spec_from_file_location(
    "stability_summary", "scripts/analysis/stability_summary.py")
ss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ss)


def _series(values, stride=20):
    return [(i * stride, v) for i, v in enumerate(values)]


def test_flat_healthy_run_is_not_flagged():
    brk, _, peak = ss.time_to_divergence(_series([0.05] * 500))
    assert brk is None and peak == pytest.approx(1.0, abs=0.05)


def test_decreasing_loss_is_never_flagged():
    brk, _, _ = ss.time_to_divergence(_series([1.0 / (i + 1) for i in range(500)]))
    assert brk is None


def test_sustained_break_is_flagged():
    brk, best, peak = ss.time_to_divergence(_series([0.03] * 300 + [0.5] * 300))
    assert brk is not None and peak > 4.0


def test_transient_spike_is_not_flagged():
    """One bad batch is not a divergence. This is not academic: each logged
    point stands for `stride` steps in the replay, so without the rolling median
    a single 167x sample counts as 20 consecutive bad steps and fakes a break."""
    brk, _, _ = ss.time_to_divergence(
        _series([0.03] * 300 + [5.0] * 2 + [0.03] * 300))
    assert brk is None


def test_two_isolated_spikes_far_apart_are_not_flagged():
    vals = [0.03] * 200 + [8.0] + [0.03] * 200 + [8.0] + [0.03] * 200
    assert ss.time_to_divergence(_series(vals))[0] is None


def test_stride_is_accounted_for():
    """The logged series is subsampled. If the replay applied the per-step decay
    once per LOGGED point it would smooth ~20x harder than the live guard and
    miss real breaks -- which is exactly how factor=20 hid the first one. The
    same break must be detected whether it is logged every step or every 20."""
    vals = [0.03] * 600 + [0.5] * 600
    fine = ss.time_to_divergence(_series(vals, stride=1), min_steps=0)[0]
    coarse = ss.time_to_divergence(_series(vals, stride=20), min_steps=0)[0]
    assert fine is not None and coarse is not None


def test_reproduces_the_measured_smoke_matrix():
    """Both SM smoke arms broke, neither CE arm did. Pinned so a change to the
    detector cannot silently rewrite the result it is meant to measure."""
    sm = _series([0.033] * 200 + [0.033 * 5.0] * 200)      # unclamped SM: ~4.9x
    ce = _series([0.102] * 200 + [0.102 * 1.9] * 200)      # healthy CE peak 1.9x
    assert ss.time_to_divergence(sm)[0] is not None
    assert ss.time_to_divergence(ce)[0] is None

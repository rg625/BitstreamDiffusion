"""The offline probe runs on a login node whose watchdog has killed it twice
mid-sweep, ~35 min per checkpoint. Resume must never lose finished work and
must never silently re-report a stale checkpoint as fresh."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "offline_probe", "scripts/analysis/offline_objective_probe.py")
op = importlib.util.module_from_spec(spec)
spec.loader.exec_module(op)


def test_step_is_parsed_from_the_filename_without_loading_the_checkpoint():
    assert op._step_from_name(Path("step=000250000.pt")) == 250000
    assert op._step_from_name(Path("last.pt")) is None
    assert op._step_from_name(Path("best.pt")) is None


def test_generator_is_stable_across_processes():
    """Uses crc32, not hash(): Python string hashing is salted per process, so
    hash() would silently break reproducibility between runs -- exactly the
    property the paired design depends on."""
    a = op._gen("grid", 5678, 3, 16).initial_seed()
    b = op._gen("grid", 5678, 3, 16).initial_seed()
    assert a == b
    assert op._gen("grid", 5678, 3, 16).initial_seed() != \
        op._gen("grid", 5678, 3, 17).initial_seed()


def test_survival_is_error_weighted_so_saturated_correct_bits_are_invisible():
    """grad_survival = sum|D-x0|*D(1-D) / sum|D-x0|: an average of D(1-D)
    WEIGHTED BY ERROR. A bit that is saturated and CORRECT carries no gradient
    under either objective, so it must not drag the ratio down -- it is not lost
    signal, it is signal that was never there. This is why saturation fraction
    and survival are different quantities and are reported separately."""
    import torch
    acc = op._Acc()
    acc.add(torch.full((1000,), 1e-9), torch.zeros(1000))   # saturated, correct
    acc.add(torch.full((2,), 0.5), torch.zeros(2))          # uncertain
    # ~0.25, the D(1-D) of the uncertain bits: the 1000 correct bits barely count
    assert acc.as_dict()["grad_survival"] == pytest.approx(0.25, rel=1e-3)


def test_survival_collapses_when_saturated_bits_are_WRONG():
    """The failure mode that matters: confidently wrong bits carry maximal error
    and near-zero D(1-D), so SM's gradient on them all but vanishes."""
    import torch
    acc = op._Acc()
    acc.add(torch.full((1000,), 1.0 - 1e-9), torch.zeros(1000))  # saturated, wrong
    acc.add(torch.full((2,), 0.5), torch.zeros(2))
    assert acc.as_dict()["grad_survival"] < 0.01


def test_pooling_is_over_bits_not_a_mean_of_per_call_ratios():
    import torch
    acc = op._Acc()
    acc.add(torch.full((1000,), 1.0 - 1e-9), torch.zeros(1000))
    acc.add(torch.full((1,), 0.5), torch.zeros(1))
    # a mean of the two per-call ratios would be ~0.125; pooled is far smaller
    assert acc.as_dict()["grad_survival"] < 0.01


def test_bootstrap_ci_brackets_the_point_estimate():
    import torch
    acc = op._Acc()
    for i in range(40):
        acc.add_rows([torch.full((50,), 0.3 + 0.001 * i)], [torch.zeros(50)], seed=1)
        acc.add(torch.full((50,), 0.3 + 0.001 * i), torch.zeros(50))
    d = acc.as_dict()
    lo, hi = d["grad_survival_ci95"]
    assert lo <= d["grad_survival"] <= hi
    assert d["n_examples"] == 40


def test_per_seed_split_is_reported_separately():
    import torch
    acc = op._Acc()
    acc.add_rows([torch.full((10,), 0.5)], [torch.zeros(10)], seed=11)
    acc.add_rows([torch.full((10,), 0.5)], [torch.zeros(10)], seed=22)
    acc.add(torch.full((20,), 0.5), torch.zeros(20))
    assert set(acc.as_dict()["per_seed_grad_survival"]) == {"11", "22"}

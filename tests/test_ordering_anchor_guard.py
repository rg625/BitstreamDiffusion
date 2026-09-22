"""The analysis must never present a clean table it cannot vouch for.

A sampler bug once returned probabilities from sigma_{N-1} instead of a final
denoise and scored 0.0000 against a true 0.164. It was caught only because a
production checkpoint happened to be in the evaluation batch; without one, seven
cells of confidently-wrong nulls would have shipped. These tests pin the two
ways that safeguard can fail quietly: the anchor being dropped before it is
seen, and the anchor being seen but not checked.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ANALYZER = REPO / "experiments" / "ordering" / "analyze_train.py"

sys.path.insert(0, str(REPO))
from experiments.ordering.analyze_train import ANCHOR_RUN, load  # noqa: E402

ANCHOR_CKPT = "tinigsm_gsm8k/runs/cobit_raw_binary_bits_cfg/checkpoints/last.pt"
ARM_CKPT = "runs/tasks/tinygsm/ord_fs500k_none_s42/checkpoints/last.pt"


def _result(checkpoint, accuracy, n=40, seed=0, order_w=0.0, order_mode="none"):
    correct = [1.0] * int(round(accuracy * n)) + [0.0] * (n - int(round(accuracy * n)))
    return {
        "checkpoint": checkpoint, "accuracy": accuracy, "seed": seed,
        "order_w": order_w, "order_mode": order_mode, "steps": 256,
        "per_problem": {"idx": list(range(n)), "correct": correct,
                        "answer": [1 if c else None for c in correct]},
        "sample_records": [{"text": "def f():"} for _ in range(n)],
        "invalid_token_rate": 1e-5, "efficiency": {"samples_per_sec": 1.0},
    }


def _write(d: Path, name, payload):
    (d / f"{name}__uniform__s{payload['seed']}.json").write_text(json.dumps(payload))


def test_anchor_checkpoint_is_not_dropped(tmp_path):
    """The anchor lives outside runs/tasks/tinygsm; the old regex silently
    returned None for it, which defeats the entire point of having one."""
    p = tmp_path / "a__uniform__s0.json"
    p.write_text(json.dumps(_result(ANCHOR_CKPT, 0.164)))
    rec = load(p)
    assert rec is not None, "anchor result was dropped by load()"
    assert ANCHOR_RUN in rec["run"]


def test_normal_arm_still_parses(tmp_path):
    p = tmp_path / "b__uniform__s0.json"
    p.write_text(json.dumps(_result(ARM_CKPT, 0.15)))
    rec = load(p)
    assert rec is not None and rec["run"] == "ord_fs500k_none_s42"


def _run_analyzer(d: Path):
    return subprocess.run(
        [sys.executable, str(ANALYZER), "--dir", str(d),
         "--control", "ord_fs500k_none_s42", "--out", str(d / "summary.json")],
        capture_output=True, text=True, cwd=str(REPO), timeout=900,
    )


def test_missing_anchor_is_announced_loudly(tmp_path):
    for sd in (0, 1):
        _write(tmp_path, "ctrl", _result(ARM_CKPT, 0.15, seed=sd))
    out = _run_analyzer(tmp_path)
    assert "NO ANCHOR" in out.stdout, out.stdout
    assert "unverified" in out.stdout


def test_out_of_band_anchor_is_announced_loudly(tmp_path):
    """0.0000 from a known-good checkpoint means the decode path is broken --
    exactly the historical failure."""
    for sd in (0, 1):
        _write(tmp_path, "ctrl", _result(ARM_CKPT, 0.15, seed=sd))
        _write(tmp_path, "anchor", _result(ANCHOR_CKPT, 0.0, seed=sd))
    out = _run_analyzer(tmp_path)
    assert "OUT OF BAND" in out.stdout, out.stdout
    assert "decode path is broken" in out.stdout


def test_in_band_anchor_passes_quietly(tmp_path):
    for sd in (0, 1):
        _write(tmp_path, "ctrl", _result(ARM_CKPT, 0.15, seed=sd))
        _write(tmp_path, "anchor", _result(ANCHOR_CKPT, 0.164, seed=sd))
    out = _run_analyzer(tmp_path)
    assert "OK" in out.stdout
    assert "OUT OF BAND" not in out.stdout
    assert "NO ANCHOR" not in out.stdout

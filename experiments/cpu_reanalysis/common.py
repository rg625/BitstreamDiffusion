"""Shared helpers for the CPU-only reanalysis (docs/cpu_reanalysis.md).

Every script in this directory calls `cpu_guard()` before doing anything else:
no GPU is used, ever, and login-node work stays at 4 threads.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import glob
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "cpu_reanalysis"
GSM8K_TEST = REPO / "datasets" / "gsm8k" / "gsm8k_test.json"

TAG_RE = re.compile(r"_ema\d_(?P<run>.+)__ck(?P<step>\d{9})__s(?P<seed>\d+)\.json$")


def cpu_guard(threads: int = 4):
    import torch
    assert not torch.cuda.is_available(), "CPU-only reanalysis: CUDA must be hidden"
    torch.set_num_threads(threads)
    OUT.mkdir(parents=True, exist_ok=True)
    return torch


def is_fkc(d: dict, path: str = "") -> bool:
    """Exclude anything that could have come from FKC, erring on exclusion."""
    prov = d.get("provenance") or {}
    kinds = {str(d.get("sampler_kind", "")), str(prov.get("sampler_kind", "")),
             str(d.get("sampler", ""))}
    if any("fkc" in k.lower() for k in kinds):
        return True
    if any(k in d for k in ("num_particles", "beta", "particle_mean_accuracy",
                            "resampling_policy")):
        return True
    return "fkc" in Path(path).name.lower()


def load_eval(path) -> dict | None:
    d = json.loads(Path(path).read_text())
    if is_fkc(d, str(path)):
        return None
    d["_path"] = str(Path(path).relative_to(REPO))
    m = TAG_RE.search(Path(path).name)
    if m:
        d["_run"], d["_step"], d["_seed"] = m.group("run"), int(m.group("step")), int(m.group("seed"))
    return d


def load_dir(pattern: str) -> list[dict]:
    out = []
    for p in sorted(glob.glob(str(REPO / pattern))):
        d = load_eval(p)
        if d is not None and (d.get("per_problem") or {}).get("idx"):
            out.append(d)
    return out


def gold_answers():
    from evaluation.tasks.sandbox_gsm8k import _extract_gold_answer
    recs = json.loads(GSM8K_TEST.read_text())
    return [_extract_gold_answer(r["response_ground_truth"]) for r in recs], recs


def parse_answer(s):
    """Invert gsm8k_eval._answer_to_str: str(int) / str(float) / None / '<overflow>'."""
    from evaluation.tasks.sandbox_gsm8k import _to_number
    if s is None or s == "<overflow>":
        return None
    try:
        return _to_number(int(s))
    except ValueError:
        pass
    try:
        return _to_number(float(s))
    except ValueError:
        return None


def per_problem(d: dict) -> dict[int, tuple[int, object]]:
    """idx -> (correct, answer string)."""
    pp = d["per_problem"]
    ans = pp.get("answer") or [None] * len(pp["idx"])
    return {int(i): (int(c), a) for i, c, a in zip(pp["idx"], pp["correct"], ans)}


def bootstrap_mean(x, n_boot=10000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    """Mean of (a - b) over paired units with a percentile CI and two-sided p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    m = diff[idx].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return {"delta": float(diff.mean()), "ci95": [float(lo), float(hi)], "p": float(min(1.0, p)),
            "n": int(len(diff))}


def write_json(name: str, obj) -> Path:
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o),
                            allow_nan=False))
    return p

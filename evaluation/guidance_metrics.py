#evaluation/guidance_metrics.py
"""Diversity, repetition and guidance-trace metrics for the guidance study.

Guidance methods trade diversity for quality, so accuracy alone cannot tell a
genuine improvement from mode collapse: a model that emits the same well-formed
answer for every prompt can look better on a task metric while being strictly
worse as a generative model. These helpers supply the other half of that
picture, plus the bit-level diagnostics that are specific to a bitstream
diffusion model.

Everything here is pure and deterministic so it can be recomputed from saved
generations without re-running a sampler.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence


# -----------------------------------------------------------------------------
# Text diversity
# -----------------------------------------------------------------------------

def _ngrams(tokens: Sequence[str], n: int) -> List[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: Iterable[str], n: int) -> float:
    """Corpus-level distinct-n: unique n-grams / total n-grams.

    Computed over the whole corpus (not averaged per sample) so that a method
    which produces fluent but near-identical outputs across prompts is penalised
    -- which is precisely the failure mode strong guidance induces.
    """
    uniq, total = set(), 0
    for t in texts:
        g = _ngrams(t.split(), n)
        uniq.update(g)
        total += len(g)
    return (len(uniq) / total) if total else 0.0


def self_repetition(text: str, n: int = 4) -> float:
    """Fraction of a sample's own n-grams that are repeats of an earlier one.

    Degenerate looping is the classic high-guidance pathology; this catches it
    within a single generation, where distinct-n (a corpus statistic) does not.
    """
    g = _ngrams(text.split(), n)
    if not g:
        return 0.0
    counts = Counter(g)
    repeated = sum(c - 1 for c in counts.values())
    return repeated / len(g)


def unique_fraction(texts: Sequence[str]) -> float:
    """Fraction of generations that are not exact duplicates of another."""
    if not texts:
        return 0.0
    return len(set(texts)) / len(texts)


def token_entropy(texts: Iterable[str]) -> float:
    """Unigram entropy (nats) of the generated corpus."""
    counts: Counter = Counter()
    for t in texts:
        counts.update(t.split())
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in counts.values())


def text_metrics(texts: Sequence[str]) -> Dict[str, float]:
    """The full diversity panel for one run's generations."""
    texts = list(texts)
    if not texts:
        return {}
    lens = [len(t.split()) for t in texts]
    out: Dict[str, float] = {
        "n_samples": float(len(texts)),
        "mean_length_tokens": sum(lens) / len(lens),
        "empty_fraction": sum(1 for t in texts if not t.strip()) / len(texts),
        "unique_fraction": unique_fraction(texts),
        "token_entropy": token_entropy(texts),
        "self_repetition_4": sum(self_repetition(t, 4) for t in texts) / len(texts),
    }
    for n in (1, 2, 3, 4):
        out[f"distinct_{n}"] = distinct_n(texts, n)
    return out


# -----------------------------------------------------------------------------
# Guidance traces
# -----------------------------------------------------------------------------

def summarise_trace(traces: Sequence[Sequence[dict]]) -> Dict[str, object]:
    """Aggregate per-step guidance diagnostics across batches.

    Returns both a scalar summary (means over the trajectory) and the full
    per-sigma curve, since several of the questions we care about -- does
    self-guidance destabilise at low sigma? does the guidance/score ratio blow
    up? -- are about the *shape* in sigma, not the average.
    """
    if not traces:
        return {}

    by_step: Dict[int, List[dict]] = {}
    for tr in traces:
        for rec in tr:
            step = rec.get("step")
            key = -1 if step == "final" else int(step)
            by_step.setdefault(key, []).append(rec)

    numeric_keys = sorted({
        k for recs in by_step.values() for r in recs for k, v in r.items()
        if k != "step" and isinstance(v, (int, float))
    })

    curve = []
    for step in sorted(by_step):
        recs = by_step[step]
        row: Dict[str, float] = {"step": step}
        for k in numeric_keys:
            vals = [float(r[k]) for r in recs if k in r and _finite(r[k])]
            if vals:
                row[k] = sum(vals) / len(vals)
        curve.append(row)

    summary: Dict[str, object] = {"per_step": curve}
    for k in numeric_keys:
        vals = [row[k] for row in curve if k in row]
        if vals:
            summary[f"{k}_mean"] = sum(vals) / len(vals)
            summary[f"{k}_max"] = max(vals)
            summary[f"{k}_min"] = min(vals)
    return summary


def _finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


# -----------------------------------------------------------------------------
# Efficiency
# -----------------------------------------------------------------------------

def efficiency_metrics(
    *,
    wall_clock_s: float,
    n_samples: int,
    n_gen_tokens: int,
    model_evaluations: Optional[int] = None,
    nfe_per_sample: Optional[float] = None,
    peak_gpu_bytes: Optional[int] = None,
) -> Dict[str, float]:
    """Throughput and cost. `model_evaluations` counts denoiser evaluations in
    units of one batch, which is the quantity that differs between guidance
    methods at a fixed step count."""
    out: Dict[str, float] = {
        "wall_clock_s": float(wall_clock_s),
        "samples_per_sec": (n_samples / wall_clock_s) if wall_clock_s > 0 else 0.0,
        "tokens_per_sec": (n_gen_tokens / wall_clock_s) if wall_clock_s > 0 else 0.0,
    }
    if model_evaluations is not None:
        out["model_evaluations"] = float(model_evaluations)
    if nfe_per_sample is not None:
        out["nfe_per_sample"] = float(nfe_per_sample)
    if peak_gpu_bytes is not None:
        out["peak_gpu_gb"] = peak_gpu_bytes / (1024 ** 3)
    return out

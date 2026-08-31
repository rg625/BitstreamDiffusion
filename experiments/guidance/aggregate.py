#experiments/guidance/aggregate.py
"""Aggregate guidance result JSONs into one tidy table.

Every plot and every number in the report is produced from this table, so no
figure is ever hand-edited and no result is transcribed by hand. Run:

    python -m experiments.guidance.aggregate runs/guidance --out results/guidance

which writes `all_cells.csv` (one row per run) and `per_step.csv` (the guidance
diagnostics as a function of sigma).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Columns lifted out of each result JSON. Nested blocks are flattened with a
# prefix so the table stays flat enough for pandas/CSV without losing provenance.
SCALARS = [
    "task", "checkpoint", "sampler", "sampler_kind", "schedule", "gamma", "steps",
    "guidance_scale", "ag_scale", "bad_checkpoint", "bad_ema",
    "sg_scale", "sg_variant", "sg_delta", "sg_mf_mode",
    "posterior_temp", "posterior_temp_target", "score_temp_tau",
    "sigma_data", "num_examples", "accuracy", "bootstrap_accuracy",
    "ci95_low", "ci95_high", "num_correct", "invalid_token_rate",
    "seed", "batch_size",
]
NESTED = {
    "diversity": ["n_samples", "mean_length_tokens", "empty_fraction", "unique_fraction",
                  "token_entropy", "self_repetition_4",
                  "distinct_1", "distinct_2", "distinct_3", "distinct_4"],
    "efficiency": ["wall_clock_s", "samples_per_sec", "tokens_per_sec",
                   "nfe_per_sample", "peak_gpu_gb", "model_evaluations"],
    "provenance": ["git_commit", "git_branch", "git_dirty", "config", "gpu_name",
                   "hostname", "slurm_job_id", "slurm_array_task_id"],
}
# Trajectory-level guidance telemetry (means/max over the sigma grid).
DIAG_SUMMARY = [
    "cfg_dir_rms_mean", "ag_dir_rms_mean", "sg_dir_rms_mean",
    "score_rms_mean", "guidance_over_score_mean", "guidance_over_score_max",
    "bit_entropy_mean_mean", "p_mean_mean", "p_min_min", "p_max_max",
    "frac_p_lt_0.01_mean", "frac_p_gt_0.99_mean",
    "frac_p_lt_0.001_mean", "frac_p_gt_0.999_mean",
]


def _method(row: Dict) -> str:
    """A short label for the guidance combination, used to group plots."""
    parts = []
    if float(row.get("guidance_scale") or 0) > 0:
        parts.append("CFG")
    if float(row.get("ag_scale") or 0) > 0:
        parts.append("AG")
    if float(row.get("sg_scale") or 0) != 0:
        parts.append(f"SG-{row.get('sg_variant', 'prev')}")
    return "+".join(parts) if parts else "baseline"


def _bad_step(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    d = "".join(ch for ch in Path(path).stem if ch.isdigit())
    return int(d) if d else None


def load_rows(root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for f in sorted(root.rglob("gsm8k_results_*.json")):
        try:
            r = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"[aggregate] skipping unreadable {f}: {e}")
            continue
        if "accuracy" not in r:
            continue

        row: Dict = {"result_file": str(f), "grid": f.parent.name}
        for k in SCALARS:
            row[k] = r.get(k)
        for block, keys in NESTED.items():
            b = r.get(block) or {}
            for k in keys:
                row[f"{block}.{k}"] = b.get(k)
        diag = r.get("guidance_diagnostics") or {}
        for k in DIAG_SUMMARY:
            row[k] = diag.get(k)

        row["method"] = _method(row)
        row["bad_step"] = _bad_step(row.get("bad_checkpoint"))
        # Total denoiser evaluations per sample: the honest compute axis.
        nfe = row.get("efficiency.nfe_per_sample")
        row["nfe_total"] = float(nfe) if nfe else None
        rows.append(row)
    return rows


def load_per_step(root: Path) -> List[Dict]:
    """Flatten each run's per-sigma diagnostics into long format."""
    out: List[Dict] = []
    for f in sorted(root.rglob("gsm8k_results_*.json")):
        try:
            r = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        curve = ((r.get("guidance_diagnostics") or {}).get("per_step")) or []
        if not curve:
            continue
        base = {
            "result_file": str(f),
            "method": _method(r),
            "guidance_scale": r.get("guidance_scale"),
            "ag_scale": r.get("ag_scale"),
            "sg_scale": r.get("sg_scale"),
            "sg_variant": r.get("sg_variant"),
            "steps": r.get("steps"),
            "bad_step": _bad_step(r.get("bad_checkpoint")),
        }
        for rec in curve:
            out.append({**base, **rec})
    return out


def write_csv(rows: List[Dict], path: Path) -> None:
    import csv
    if not rows:
        print(f"[aggregate] nothing to write to {path}")
        return
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[aggregate] wrote {len(rows)} rows x {len(cols)} cols -> {path}")


def summarise(rows: List[Dict]) -> None:
    """Human-readable table, sorted by accuracy."""
    if not rows:
        print("[aggregate] no results found")
        return
    print(f"\n{'method':16} {'w':>5} {'ag':>5} {'sg':>5} {'nfe':>5} {'n':>5} "
          f"{'acc':>7} {'ci95':>15} {'dist2':>6} {'uniq':>5} {'s/sec':>7}")
    print("-" * 104)
    for r in sorted(rows, key=lambda x: -(x.get("accuracy") or 0)):
        ci = f"[{_f(r.get('ci95_low'))},{_f(r.get('ci95_high'))}]"
        print(f"{r['method']:16} {_g(r.get('guidance_scale')):>5} {_g(r.get('ag_scale')):>5} "
              f"{_g(r.get('sg_scale')):>5} {_g(r.get('steps')):>5} {_g(r.get('num_examples')):>5} "
              f"{_f(r.get('accuracy')):>7} {ci:>15} "
              f"{_f(r.get('diversity.distinct_2')):>6} "
              f"{_f(r.get('diversity.unique_fraction')):>5} "
              f"{_f(r.get('efficiency.samples_per_sec'), 2):>7}")


def _f(v, nd=4) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def _g(v) -> str:
    try:
        f = float(v)
        return f"{int(f)}" if f == int(f) else f"{f:g}"
    except (TypeError, ValueError):
        return "-"


def check_smoke(rows: List[Dict]) -> int:
    """Validate a smoke run: every cell present, well-formed, and distinguishable.

    Returns a non-zero exit code on failure so the SLURM job fails loudly
    instead of leaving a green log that nobody re-reads.
    """
    expected = {"baseline", "CFG", "AG", "SG-prev", "SG-exact", "CFG+AG+SG-prev"}
    found = {r["method"] for r in rows}
    problems: List[str] = []

    missing = expected - found
    if missing:
        problems.append(f"missing cells: {sorted(missing)}")

    for r in rows:
        tag = r["method"]
        acc = r.get("accuracy")
        if acc is None or not math.isfinite(float(acc)):
            problems.append(f"{tag}: accuracy is not finite")
        if not r.get("provenance.git_commit"):
            problems.append(f"{tag}: no git commit recorded")
        if (r.get("efficiency.wall_clock_s") or 0) <= 0:
            problems.append(f"{tag}: no wall-clock recorded")
        if r.get("cfg_dir_rms_mean") is None and tag.startswith("CFG"):
            problems.append(f"{tag}: CFG active but no cfg_dir_rms diagnostic")
        if r.get("ag_dir_rms_mean") is None and "AG" in tag:
            problems.append(f"{tag}: AG active but no ag_dir_rms diagnostic")

    # Guidance that changed nothing means the policy was silently dropped --
    # the failure most likely to survive all the way into a sweep.
    #
    # One benign cause must be excluded first: CoBit's SDT uses AdaLN-zero
    # (`AdaLNModulation` zero-initialises its output projection, models/sdt.py),
    # so an UNTRAINED model is exactly input-independent -- every branch returns
    # the same constant and every guidance direction is identically zero. That
    # is a property of the checkpoint, not of the guidance code, and it makes
    # the comparison below vacuous. Detect it and say so, rather than reporting
    # a bug that is not there.
    dir_keys = ("cfg_dir_rms_mean", "ag_dir_rms_mean", "sg_dir_rms_mean")
    observed = [r.get(k) for r in rows for k in dir_keys if r.get(k) is not None]
    model_is_inert = bool(observed) and all(abs(float(v)) == 0.0 for v in observed)
    if model_is_inert:
        problems.append(
            "every guidance direction is exactly zero across all cells. This is what an "
            "UNTRAINED checkpoint looks like (AdaLN-zero makes the model output "
            "independent of its input at initialisation), so the smoke run validated "
            "plumbing only. Re-run against a trained checkpoint before drawing any "
            "conclusion about guidance.")

    base = next((r for r in rows if r["method"] == "baseline"), None)
    if base is not None and not model_is_inert:
        for r in rows:
            if r["method"] == "baseline":
                continue
            if r.get("num_correct") == base.get("num_correct") and \
               abs((r.get("diversity.token_entropy") or 0)
                   - (base.get("diversity.token_entropy") or 0)) < 1e-9:
                problems.append(
                    f"{r['method']}: identical to baseline (guidance may be inert)")

    print("\n=== smoke check ===")
    for r in sorted(rows, key=lambda x: x["method"]):
        print(f"  {r['method']:16} acc={_f(r.get('accuracy'))} "
              f"nfe/sample={_g(r.get('nfe_total'))} "
              f"wall={_f(r.get('efficiency.wall_clock_s'), 1)}s "
              f"peakGB={_f(r.get('efficiency.peak_gpu_gb'), 2)}")
    if problems:
        print("\nFAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nAll smoke checks passed.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="directory of result JSONs (searched recursively)")
    ap.add_argument("--out", default=None, help="directory for the CSV outputs")
    ap.add_argument("--check-smoke", action="store_true",
                    help="validate a smoke run and exit non-zero on failure")
    args = ap.parse_args()

    root = Path(args.root)
    rows = load_rows(root)

    if args.check_smoke:
        raise SystemExit(check_smoke(rows))

    summarise(rows)
    if args.out:
        out = Path(args.out)
        write_csv(rows, out / "all_cells.csv")
        write_csv(load_per_step(root), out / "per_step.csv")


if __name__ == "__main__":
    main()

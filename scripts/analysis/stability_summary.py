"""Time-to-divergence across the seed matrix -- the PRIMARY endpoint.

Reads the TensorBoard logs of the stability runs and reports, per arm and seed,
the step at which the loss broke. "Break" is defined the same way the live guard
defines it, so the offline number and the live abort cannot disagree:

    EMA(loss) sustained above `factor` x its best-so-far for `patience` steps.

Also drops the duplicate points that validation batches used to write at a
single step (fixed in the trainer, but older event files still contain them).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import statistics
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def series(run_dir: str, tag: str):
    ev = sorted(glob.glob(f"{run_dir}/training_logs/events.out.tfevents*"))
    if not ev:
        return []
    ea = EventAccumulator(ev[-1], size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return []
    sc = ea.Scalars(tag)
    c = collections.Counter(s.step for s in sc)
    return [(s.step, s.value) for s in sc if c[s.step] == 1]


def time_to_divergence(loss, factor=4.0, patience=200, min_steps=2500,
                       ema_decay=0.99, smooth_k=5):
    """Returns (break_step, best_ema, peak_ratio). break_step is None if stable.

    Two corrections are needed to make an OFFLINE replay agree with the live
    guard, which sees every step while the log is subsampled:

    * the per-step decay is raised to the logging stride, or the replay would
      smooth ~20x harder than the guard and miss real breaks (that is exactly
      how factor=20 hid the first one);

    * a rolling MEDIAN is applied first. Each logged point is the loss at a
      single step, not the mean over the stride, so raising the decay also
      makes one spiky sample count as `stride` consecutive bad steps. Without
      the median a single 167x batch is enough to fake a divergence. The median
      removes isolated spikes and leaves sustained elevation untouched, which is
      what "diverged" is supposed to mean.
    """
    if len(loss) < 2:
        return None, None, None
    stride = max(1, loss[1][0] - loss[0][0])
    d = ema_decay ** stride
    if smooth_k > 1 and len(loss) >= smooth_k:
        h = smooth_k // 2
        sm = []
        for i, (st, _) in enumerate(loss):
            w = [v for _, v in loss[max(0, i - h):i + h + 1]]
            sm.append((st, statistics.median(w)))
        loss = sm
    ema = best = None
    strikes = 0
    brk = None
    peak = 0.0
    for s, v in loss:
        ema = v if ema is None else d * ema + (1 - d) * v
        if s < min_steps:
            continue
        if best is None or ema < best:
            best = ema
            strikes = 0
            continue
        peak = max(peak, ema / best)
        strikes = strikes + stride if ema > factor * best else 0
        if strikes >= patience and brk is None:
            brk = s
    return brk, best, peak


GUARD_RE = re.compile(
    r"\[divergence-guard\][^\n]*?at step (\d+)")
HDR_RE = re.compile(r"\[stab\] arm=(\S+) seed=(\S+)")


def scan_job_logs(log_glob="logs/arch/cobit_objstab_*.log"):
    """Map (arm, seed) -> break step reported by the LIVE guard.

    This is the authoritative measurement, not the TB replay. An aborting run
    stops at the break, so TensorBoard loses its final flush: for one seed the
    event file ends 481 steps BEFORE the guard fired, leaving no offline
    evidence at all. Replaying truncated series would report 0/6 diverged while
    the guard aborted 6/6 -- a silent disagreement between the detector and the
    thing it is meant to measure.
    """
    out = {}
    for f in sorted(glob.glob(log_glob)):
        try:
            txt = Path(f).read_text(errors="ignore")
        except OSError:
            continue
        h = HDR_RE.search(txt)
        if not h:
            continue
        steps = [int(m) for m in GUARD_RE.findall(txt)]
        out[(h.group(1), h.group(2))] = {
            "log": Path(f).name,
            "guard_break_step": min(steps) if steps else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs/tasks/tinygsm")
    ap.add_argument("--pattern", default="obj_binary_*_stab_s*")
    ap.add_argument("--factor", type=float, default=4.0)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--out", default="results/objective/stability_summary.json")
    ap.add_argument("--job-logs", default="logs/arch/cobit_objstab_*.log")
    args = ap.parse_args()

    jobs = scan_job_logs(args.job_logs)
    rows = []
    for d in sorted(glob.glob(f"{args.runs_root}/{args.pattern}")):
        name = Path(d).name
        arm = "binary_ce" if "binary_ce" in name else "binary_sm"
        seed = name.rsplit("_s", 1)[-1]
        loss = series(d, "loss/iter_train")
        if not loss:
            print(f"[stab] {name}: no loss series, skipping")
            continue
        replay_brk, best, peak = time_to_divergence(
            loss, args.factor, args.patience)
        job = jobs.get((arm, seed), {})
        guard_brk = job.get("guard_break_step")
        # The live guard wins where it spoke; the replay only covers runs that
        # ran to completion without aborting.
        brk = guard_brk if guard_brk is not None else replay_brk
        source = "live-guard" if guard_brk is not None else "tb-replay"
        surv = series(d, "objective/grad_survival")
        logit = series(d, "objective/logit_abs_mean")
        upd = series(d, "optim/update_rms")
        rows.append({
            "run": name, "arm": arm, "seed": seed,
            "last_step": loss[-1][0],
            "diverged": brk is not None,
            "break_step": brk,
            "break_source": source,
            "guard_break_step": guard_brk,
            "tb_replay_break_step": replay_brk,
            "best_loss_ema": best,
            "peak_ratio": peak,
            "survival_final": surv[-1][1] if surv else None,
            "logit_abs_mean_final": logit[-1][1] if logit else None,
            "update_rms_final": upd[-1][1] if upd else None,
        })

    print(f"{'run':32s} {'diverged':>9} {'break':>7} {'src':>11} {'tb_last':>8} "
          f"{'|ell|':>8} {'upd_rms':>9}")
    for r in rows:
        print(f"{r['run']:32s} {str(r['diverged']):>9} "
              f"{str(r['break_step']):>7} {r['break_source']:>11} "
              f"{r['last_step']:>8} "
              f"{(r['logit_abs_mean_final'] or float('nan')):>8.3g} "
              f"{(r['update_rms_final'] or float('nan')):>9.3g}")
        if (r["guard_break_step"] is not None
                and r["tb_replay_break_step"] is None):
            print(f"{'':32s}   (TB series ends at {r['last_step']}, "
                  f"{r['guard_break_step'] - r['last_step']} steps before the "
                  f"guard fired: final flush lost on abort)")

    print()
    for arm in ("binary_sm", "binary_ce"):
        a = [r for r in rows if r["arm"] == arm]
        if not a:
            continue
        nd = sum(r["diverged"] for r in a)
        brks = [r["break_step"] for r in a if r["break_step"] is not None]
        med = statistics.median(brks) if brks else None
        print(f"{arm}: {nd}/{len(a)} diverged" +
              (f", median break step {med}" if med else ""))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"factor": args.factor, "patience": args.patience, "runs": rows}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

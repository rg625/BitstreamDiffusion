"""Paired analysis of the TRAINING-time ordering arms.

Control = the arm trained AND decoded with uniform sigma. Every other arm is
compared against it on the same problems with the same seed, so the bootstrap
resamples the (control, arm) pair together.

Reports, per the brief: accuracy, paired delta, CI, T1/T2/T3 validity
breakdown, and the sigma/order statistics that show the intervention was active.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# The production 500k checkpoint, scoring 0.164 under karras/DDIM/256 steps.
ANCHOR_RUN = "cobit_raw_binary_bits_cfg"
ANCHOR_BAND = (0.12, 0.20)


def load(path):
    d = json.loads(Path(path).read_text())
    pp = d.get("per_problem") or {}
    if not pp.get("idx"):
        return None
    # Identify the arm from the RECORDED fields, not the filename: this result
    # schema does not store `tag`, so filename parsing silently collapsed every
    # arm to the same truncated string and made the whole table unreadable.
    ck = str(d.get("checkpoint") or "")
    mrun = re.search(r"runs/tasks/tinygsm/([^/]+)/checkpoints/(.+)$", ck)
    if mrun:
        run, ckpt = mrun.group(1), mrun.group(2)
    else:
        # The production anchor does not live under runs/tasks/tinygsm, and the
        # stricter pattern dropped it SILENTLY -- which defeats the whole point
        # of including an anchor. Fall back to the directory that holds the
        # checkpoints.
        mrun = re.search(r"([^/]+)/checkpoints/(.+)$", ck)
        if not mrun:
            return None
        run, ckpt = mrun.group(1), mrun.group(2)
    decode = "uniform" if float(d.get("order_w") or 0.0) == 0.0 else "matched"
    if "__uniform__" in Path(path).stem:
        decode = "uniform"
    elif "__matched__" in Path(path).stem:
        decode = "matched"
    seed = int(d.get("seed") or 0)
    fields = {"run": run, "decode": decode, "seed": seed, "ckpt": ckpt}
    ans = pp.get("answer") or []
    cor = np.asarray(pp["correct"], dtype=float)
    recs = d.get("sample_records") or []
    return {
        "run": fields["run"], "decode": fields["decode"], "seed": fields["seed"],
        "ckpt": fields["ckpt"], "train_mode": d.get("order_mode"),
        "decode_w": d.get("order_w"),
        "idx": np.asarray(pp["idx"]), "correct": cor,
        "accuracy": d.get("accuracy"),
        "T1": sum(1 for a in ans if a is None),
        "T2": sum(1 for a, c in zip(ans, cor) if a is not None and not c),
        "T3": sum(1 for r in recs if "def " not in (r.get("text") or "")),
        "n_recs": len(recs),
        "invalid": d.get("invalid_token_rate"),
        "sps": (d.get("efficiency") or {}).get("samples_per_sec"),
    }


def paired(ctrl, arm, n_boot=10000, seed=0):
    common, ci, ai = np.intersect1d(ctrl["idx"], arm["idx"], return_indices=True)
    c, a = ctrl["correct"][ci], arm["correct"][ai]
    d = float(a.mean() - c.mean())
    rng = np.random.default_rng(seed)
    bs = rng.integers(0, len(common), size=(n_boot, len(common)))
    dd = a[bs].mean(1) - c[bs].mean(1)
    lo, hi = np.percentile(dd, [2.5, 97.5])
    p = 2.0 * min((dd <= 0).mean(), (dd >= 0).mean())
    return d, float(lo), float(hi), float(min(p, 1.0)), len(common)


def sigma_stats(mode, w, n_tok=512, n_prompt=68, bpt=16,
                smin=0.002, smax=80.0, sigma=0.4):
    """The intervention's actual effect on per-position sigma, so 'it was
    active' is a measured statement rather than an assertion."""
    import torch
    from diffusion.continuous.ordering import training_position_sigma
    pm = torch.zeros(1, n_tok * bpt, dtype=torch.bool)
    pm[:, : n_prompt * bpt] = True
    s = training_position_sigma(
        torch.tensor([sigma]), n_bits=n_tok * bpt, bits_per_token=bpt, mode=mode,
        w=w, sigma_min=smin, sigma_max=smax, prefix_mask=pm,
        generator=torch.Generator().manual_seed(0))
    suf = s[0, n_prompt * bpt:]
    return {"sigma_min": float(suf.min()), "sigma_max": float(suf.max()),
            "spread": float(suf.max() / suf.min())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/ordering/train_eval")
    ap.add_argument("--control", default=None,
                    help="run name of the uniform-trained control")
    ap.add_argument("--out", default="results/ordering/train_summary.json")
    args = ap.parse_args()

    runs = [r for r in (load(p) for p in sorted(glob.glob(f"{args.dir}/*.json"))) if r]
    if not runs:
        print(f"[ord] no results in {args.dir}")
        return
    by = defaultdict(dict)
    for r in runs:
        by[(r["run"], r["ckpt"], r["decode"])][r["seed"]] = r

    # NB: keys are (run, ckpt, decode) triples; unpacking them as pairs raised
    # ValueError whenever --control was omitted.
    ctrl_run = args.control or next(
        (n for (n, _ck, _dec) in by if "none" in n), None)
    if ctrl_run is None:
        print("[ord] no control arm found"); return
    print(f"[ord] control = {ctrl_run}\n")

    # ---- known-good anchor -------------------------------------------------
    # A sampler bug once returned probabilities from sigma_{N-1} instead of a
    # final denoise and scored 0.000 against a true 0.164. It was caught only
    # because a production checkpoint was in the batch. An anchor that is
    # missing, or present but out of band, invalidates every other cell here,
    # so say so loudly rather than printing a clean-looking table of nulls.
    anchors = [r for r in runs if ANCHOR_RUN in r["run"]]
    if not anchors:
        print("!" * 72)
        print(f"[ord] NO ANCHOR in {args.dir}: nothing here verifies the decode path.")
        print(f"[ord] Expected a cell from a checkpoint under .../{ANCHOR_RUN}/.")
        print(f"[ord] Treat every number below as unverified.")
        print("!" * 72 + "\n")
    else:
        accs = [a["accuracy"] for a in anchors]
        lo, hi = ANCHOR_BAND
        ok = all(lo <= a <= hi for a in accs)
        line = ", ".join(f"{a:.4f}" for a in accs)
        print(f"[ord] anchor {ANCHOR_RUN}: {line} "
              f"(expected {lo:.2f}-{hi:.2f}) -> {'OK' if ok else 'OUT OF BAND'}")
        if not ok:
            print("!" * 72)
            print("[ord] The anchor is a KNOWN-GOOD checkpoint. Out of band means the")
            print("[ord] decode path is broken, not that the anchor regressed. Fix that")
            print("[ord] before reading anything below.")
            print("!" * 72)
        print()

    print(f"{'arm':26s} {'ckpt':>10s} {'decode':8s} {'seed':>4} {'acc':>7} {'delta':>8} "
          f"{'95% CI':>19} {'p':>7} {'T1':>5} {'T2':>5}")
    out = []
    for (run, ckpt, dec), seeds in sorted(by.items()):
        for sd, r in sorted(seeds.items()):
            # control must be the SAME checkpoint step, never pooled across 5k/30k
            c = by.get((ctrl_run, ckpt, "uniform"), {}).get(sd) or \
                by.get((ctrl_run, ckpt, "matched"), {}).get(sd)
            if c is None:
                continue
            if run == ctrl_run and dec in ("uniform", "matched"):
                print(f"{run[:26]:26s} {ckpt[-10:]:>10s} {dec:8s} {sd:>4} "
                      f"{r['accuracy']:>7.4f} {'-':>8} {'-':>19} {'-':>7} "
                      f"{r['T1']:>5} {r['T2']:>5}")
                continue
            d, lo, hi, p, n = paired(c, r)
            sig = "*" if (lo > 0 or hi < 0) else " "
            print(f"{run[:26]:26s} {ckpt[-10:]:>10s} {dec:8s} {sd:>4} "
                  f"{r['accuracy']:>7.4f} {d:>+8.4f} [{lo:>+7.4f},{hi:>+7.4f}] "
                  f"{p:>7.4f}{sig} {r['T1']:>5} {r['T2']:>5}")
            out.append({"run": run, "ckpt": ckpt, "decode": dec, "seed": sd, "n": n,
                        "accuracy": r["accuracy"], "control_accuracy": c["accuracy"],
                        "delta": d, "ci95": [lo, hi], "p": p,
                        "significant": bool(lo > 0 or hi < 0),
                        "T1": r["T1"], "T2": r["T2"], "T3": r["T3"],
                        "invalid_token_rate": r["invalid"],
                        "samples_per_sec": r["sps"]})

    print("\nper-position sigma actually applied (global sigma=0.4, 68-token prompt):")
    for mode in ("l2r", "r2l", "random"):
        st = sigma_stats(mode, 0.25)
        print(f"   {mode:7s} w=0.25  suffix sigma {st['sigma_min']:.4f}"
              f"-{st['sigma_max']:.4f}  spread {st['spread']:.1f}x")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[ord] wrote {args.out}   (* = 95% CI excludes zero)")


if __name__ == "__main__":
    main()

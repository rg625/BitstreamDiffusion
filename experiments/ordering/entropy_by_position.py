#!/usr/bin/env python
"""Empirical token entropy H(X_p) as a function of position p in TinyGSM.

WHY THIS BEARS ON ORDERING. An L2R or R2L training schedule assumes the
sequence has a direction: that one end is systematically easier to predict than
the other, so denoising it first gives the rest something to condition on. That
assumption is checkable directly in the data, without a model, by measuring how
uncertain the token at each position is across the corpus.

Two frames, because they answer different questions:

  absolute  -- position in the 512-token block. Dominated by the prompt/answer
               boundary and by padding, so it mostly describes the format.
  aligned   -- position relative to the start of the ANSWER (p - prompt_len).
               This is the region the ordering schedules actually reorder: the
               prompt is clamped clean and excluded from the loss.

ESTIMATOR. Plug-in entropy is biased low, badly so when the alphabet (49,153)
is large next to the sample. The Miller-Madow correction (+(K-1)/(2N ln2) bits,
K = distinct tokens observed at that position) is applied and the raw value is
kept alongside, so the size of the correction is visible rather than assumed
away. Positions are reported only where enough examples contribute.

    python experiments/ordering/entropy_by_position.py --n 500000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CACHE = Path("datasets/tinygsm")
STEM = ("tinygsm_bs512_answeronly_trainonpad_filtered_"
        "HuggingFaceTB__SmolLM-135M_capall_vr0.01_vs42")


def _entropy_bits(counts: np.ndarray) -> tuple[float, float, int, int]:
    """(plug-in, Miller-Madow, N, K) in bits for one position's token counts."""
    n = int(counts.sum())
    if n == 0:
        return float("nan"), float("nan"), 0, 0
    nz = counts[counts > 0]
    p = nz / n
    h = float(-(p * np.log2(p)).sum())
    k = int(nz.size)
    return h, h + (k - 1) / (2.0 * n * np.log(2.0)), n, k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500_000, help="examples to sample")
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--split", default="train", choices=["train", "validation"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-examples", type=int, default=5_000,
                    help="drop aligned positions supported by fewer examples")
    ap.add_argument("--out", default="results/ordering/entropy_by_position")
    args = ap.parse_args()

    meta = json.loads((CACHE / f"{STEM}.meta.json").read_text())
    n_total = int(meta[args.split]["n"])
    vocab = int(meta["tokenizer_len"])
    block = 512

    ids = np.memmap(CACHE / f"{STEM}_{args.split}_ids.uint16", dtype=np.uint16,
                    mode="r", shape=(n_total, block))
    plen = np.fromfile(CACHE / f"{STEM}_{args.split}_plen.int32", dtype=np.int32)
    assert plen.shape[0] == n_total, (plen.shape, n_total)

    n = min(args.n, n_total)
    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(n_total, size=n, replace=False))
    print(f"[ent] {args.split}: sampling {n:,} of {n_total:,} examples, "
          f"vocab={vocab}, block={block}")
    print(f"[ent] prompt_len: mean={plen[rows].mean():.1f} "
          f"median={np.median(plen[rows]):.0f} "
          f"p5={np.percentile(plen[rows],5):.0f} p95={np.percentile(plen[rows],95):.0f}")

    # counts[position, token]; the aligned frame gets one extra sentinel column
    # for "this example has no such answer position", never counted as a token.
    abs_counts = np.zeros((block, vocab), dtype=np.int32)
    ali_counts = np.zeros((block, vocab + 1), dtype=np.int32)
    ar = np.arange(block)

    for start in range(0, n, args.chunk):
        sel = rows[start:start + args.chunk]
        blk = np.asarray(ids[sel], dtype=np.int32)          # [c, 512]
        pl = plen[sel].astype(np.int32)

        for p in range(block):
            abs_counts[p] += np.bincount(blk[:, p], minlength=vocab)[:vocab]

        # shift each row so column r is answer-position r
        idx = pl[:, None] + ar[None, :]
        valid = idx < block
        ali = np.where(valid, np.take_along_axis(blk, np.clip(idx, 0, block - 1), 1),
                       vocab)                                # sentinel
        for r in range(block):
            ali_counts[r] += np.bincount(ali[:, r], minlength=vocab + 1)
        print(f"\r[ent] {min(start + args.chunk, n):,}/{n:,}", end="", flush=True)
    print()

    pad_id = int(np.argmax(abs_counts[block - 1]))           # tail is all padding
    print(f"[ent] inferred PAD id = {pad_id}")

    out = {"absolute": [], "aligned": [], "pad_id": pad_id, "n_examples": int(n),
           "split": args.split, "vocab": vocab,
           "prompt_len_mean": float(plen[rows].mean()),
           "prompt_len_median": float(np.median(plen[rows]))}

    for p in range(block):
        h, hmm, nn, k = _entropy_bits(abs_counts[p])
        pad_frac = float(abs_counts[p, pad_id]) / max(nn, 1)
        out["absolute"].append({"pos": p, "H": h, "H_mm": hmm, "n": nn,
                                "distinct": k, "pad_frac": pad_frac})

    for r in range(block):
        c = ali_counts[r][:vocab]                            # drop sentinel
        h, hmm, nn, k = _entropy_bits(c)
        if nn < args.min_examples:
            continue
        pad_frac = float(c[pad_id]) / max(nn, 1)
        # Entropy over REAL tokens only: padding is format, not content, and
        # left in it would manufacture a downward slope that says nothing about
        # how predictable the program is.
        c_nopad = c.copy(); c_nopad[pad_id] = 0
        h_np, hmm_np, nn_np, k_np = _entropy_bits(c_nopad)
        out["aligned"].append({"pos": r, "H": h, "H_mm": hmm, "n": nn,
                               "distinct": k, "pad_frac": pad_frac,
                               "H_nopad": h_np, "H_mm_nopad": hmm_np,
                               "n_nopad": nn_np})

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "entropy_by_position.json").write_text(json.dumps(out))
    print(f"[ent] wrote {outdir/'entropy_by_position.json'}")

    a = out["aligned"]
    print("\n answer-position entropy (PAD excluded, Miller-Madow corrected):")
    print(f"{'pos':>5} {'H_bits':>8} {'raw':>8} {'correction':>11} {'n':>9} {'pad%':>6}")
    for r in a:
        if r["pos"] in (0, 1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384, 448):
            print(f"{r['pos']:>5} {r['H_mm_nopad']:>8.3f} {r['H_nopad']:>8.3f} "
                  f"{r['H_mm_nopad']-r['H_nopad']:>11.4f} {r['n_nopad']:>9,} "
                  f"{100*r['pad_frac']:>5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

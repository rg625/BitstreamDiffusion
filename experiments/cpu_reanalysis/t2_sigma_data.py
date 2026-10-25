"""TASK 2: does evaluating with the wrong sigma_data hurt the denoiser? (audit T1)

Forward passes only, no sampling. For held-out TinyGSM examples (the cache's
`validation` split, which training never sees), at a fixed set of sigmas, the
denoiser is run twice on the SAME noisy input x_t = x0 + sigma * eps (eps drawn
once per example and sigma, then reused): once with sigma_data = the run's
sidecar value (what eval uses) and once with 0.5 (what ranks 1..N-1 trained
with). sigma_data enters only through EDM preconditioning (c_in in
models/sdt.py); the matched filter does not depend on it.

Logits go through `_model_logits_continuous`, so the matched-filter residual
is included exactly as in training and sampling (likelihood.py is NOT used:
it skips the matched filter, audit T9).

Self-conditioning is DISABLED in the training-consistent way: x0_hat is zeros
on the suffix with the clean prompt injected into the prefix, which is exactly
what the trainer feeds on the (1 - p_sc) = 50% of steps without a
self-conditioning pre-pass. It is identical in both arms.

Measured on the suffix only, split into non-padding tokens (answer + EOS) and
padding tokens: per-bit BCE, bit error rate, and token error rate of the
thresholded x0-hat (16 bits -> one token id). Paired bootstrap over examples
for every (sidecar - 0.5) difference.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.cpu_reanalysis.common import REPO, cpu_guard, paired_bootstrap, write_json  # noqa: E402

torch = cpu_guard()

from ml_collections import ConfigDict  # noqa: E402

from data.task_codec import token_ids_to_bits  # noqa: E402
from diffusion.continuous.logit_postprocess import _model_logits_continuous  # noqa: E402
from models import create_model  # noqa: E402
from utils.ema import EMA  # noqa: E402

SIGMAS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0)
METRICS = ("bce", "ber", "ter")


def load(run_dir: Path, ckpt: Path):
    cfg = ConfigDict(json.loads((run_dir / "config.json").read_text()))
    model = create_model(cfg).eval()
    ck = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
    sd = {k.replace("_orig_mod.", "").removeprefix("module."): v for k, v in ck["model"].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    ema = EMA(model, decay=float(getattr(cfg.train, "ema_decay", 0.9999)))
    ema.load_state_dict(ck["ema"])
    ema.apply(model)
    side = float(json.loads((run_dir / "sigma_data.json").read_text())["sigma_data"])
    info = {"global_step": int(ck["global_step"]), "missing_keys": len(missing),
            "unexpected_keys": len(unexpected), "ema_applied": True}
    return cfg, model, side, info


def heldout(cfg, n: int, seed: int):
    d = cfg.data
    root = REPO / d.root
    meta = sorted(root.glob("tinygsm_bs512_*vr0.01_vs42.meta.json"))
    assert len(meta) == 1, meta
    tag = meta[0].name.removesuffix(".meta.json")
    m = json.loads(meta[0].read_text())
    nv = int(m["validation"]["n"])
    ids = np.memmap(root / f"{tag}_validation_ids.uint16", dtype=np.uint16, mode="r",
                    shape=(nv, int(d.sequence_len_tokens)))
    plen = np.fromfile(root / f"{tag}_validation_plen.int32", dtype=np.int32)
    rows = np.sort(np.random.default_rng(seed).choice(nv, size=n, replace=False))
    return tag, rows, torch.from_numpy(ids[rows].astype(np.int64)), torch.from_numpy(plen[rows].astype(np.int64))


@torch.inference_mode()
def forward(model, cfg, xt, sigma, x0hat, sd):
    cfg.diffusion.continuous.sigma_data = float(sd)
    return _model_logits_continuous(model, cfg, xt, sigma, x0hat).float()


def metrics(logits, x0bits, tok_ids, region_tok, bpt):
    """Per-example means over a token region (bits of those tokens)."""
    B, S = x0bits.shape
    T = S // bpt
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, x0bits, reduction="none")
    hard = (logits > 0).float()
    berr = (hard != x0bits).float()
    w = 2 ** torch.arange(bpt - 1, -1, -1, dtype=torch.int64)       # MSB-first
    tok_hat = (hard.view(B, T, bpt).long() * w).sum(-1)
    terr = (tok_hat != tok_ids).float()
    rb = region_tok.repeat_interleave(bpt, dim=1).float()
    cnt_b = rb.sum(1).clamp(min=1)
    cnt_t = region_tok.float().sum(1).clamp(min=1)
    return {"bce": ((bce * rb).sum(1) / cnt_b), "ber": ((berr * rb).sum(1) / cnt_b),
            "ter": ((terr * region_tok.float()).sum(1) / cnt_t), "has": region_tok.any(1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="runs/tasks/tinygsm/ord_gh500k_none_s42")
    ap.add_argument("--ckpt", default=None, help="default: <run_dir>/checkpoints/last.pt")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", action="store_true", help="time one batch and exit")
    ap.add_argument("--tag", default="bits_control")
    args = ap.parse_args()

    run_dir = REPO / args.run_dir
    ckpt = Path(args.ckpt) if args.ckpt else run_dir / "checkpoints" / "last.pt"
    cfg, model, side, info = load(run_dir, ckpt)
    bpt = int(cfg.data.bits_per_token)
    from data.tinygsm import get_tokenizer, _special_ids
    _, eos, pad = _special_ids(get_tokenizer(str(cfg.data.tokenizer_name)))
    tag, rows, ids, plen = heldout(cfg, args.n, args.seed)
    T = ids.shape[1]
    pos = torch.arange(T).unsqueeze(0)
    suffix = pos >= plen.unsqueeze(1)
    is_pad = (ids == pad) & suffix
    content = suffix & ~is_pad
    x0bits = token_ids_to_bits(ids, bpt).float().view(len(rows), -1)
    prefix_bits = (~suffix).repeat_interleave(bpt, dim=1)
    print(f"[t2] {args.run_dir} step {info['global_step']} sidecar sigma_data={side:.6f}; "
          f"{len(rows)} held-out rows; suffix tokens/ex {suffix.sum(1).float().mean():.1f}, "
          f"content {content.sum(1).float().mean():.1f}, pad {is_pad.sum(1).float().mean():.1f}; "
          f"pad id {pad}, eos id {eos}", flush=True)

    if args.bench:
        b = slice(0, args.batch)
        g = torch.Generator().manual_seed(0)
        xt = x0bits[b] + 0.2 * torch.randn(x0bits[b].shape, generator=g)
        xt[prefix_bits[b]] = x0bits[b][prefix_bits[b]]
        x0hat = torch.zeros_like(xt); x0hat[prefix_bits[b]] = x0bits[b][prefix_bits[b]]
        forward(model, cfg, xt, torch.full((args.batch,), 0.2), x0hat, side)   # warm-up
        t0 = time.time()
        forward(model, cfg, xt, torch.full((args.batch,), 0.2), x0hat, side)
        dt = time.time() - t0
        total = dt * (args.n / args.batch) * len(SIGMAS) * 2
        print(f"[bench] batch {args.batch}: {dt:.2f} s/forward -> full run ~{total / 60:.1f} min")
        return

    acc = {s: {arm: {r: {m: [] for m in METRICS + ("has",)} for r in ("content", "pad")}
               for arm in ("sidecar", "sd0.5")} for s in SIGMAS}
    # Per-sigma checkpoint: a killed run resumes from the last finished sigma.
    partial_p = REPO / "results" / "cpu_reanalysis" / f"t2_partial_{args.tag}.json"
    done = {}
    if partial_p.exists():
        done = json.loads(partial_p.read_text())
        if done.get("rows") != rows.tolist() or done.get("sidecar") != side:
            raise SystemExit(f"{partial_p} is from a different setup; move it away")
        for k, v in done["acc"].items():
            acc[float(k)] = v
        print(f"[t2] resuming: sigmas already done {sorted(map(float, done['acc']))}", flush=True)
    t0 = time.time()
    for si, s in enumerate(SIGMAS):
        if str(s) in done.get("acc", {}):
            continue
        for start in range(0, len(rows), args.batch):
            b = slice(start, start + args.batch)
            nb = x0bits[b].shape[0]
            # One eps per (example, sigma), shared by both arms.
            eps = torch.stack([torch.randn(x0bits.shape[1], generator=torch.Generator().manual_seed(
                int(rows[start + k]) * 1000 + si)) for k in range(nb)])
            xt = x0bits[b] + s * eps
            xt[prefix_bits[b]] = x0bits[b][prefix_bits[b]]
            x0hat = torch.zeros_like(xt)
            x0hat[prefix_bits[b]] = x0bits[b][prefix_bits[b]]
            sig = torch.full((nb,), s)
            for arm, sd in (("sidecar", side), ("sd0.5", 0.5)):
                lg = forward(model, cfg, xt, sig, x0hat, sd)
                for region, mask in (("content", content[b]), ("pad", is_pad[b])):
                    mt = metrics(lg, x0bits[b], ids[b], mask, bpt)
                    for k in METRICS + ("has",):
                        acc[s][arm][region][k].extend(mt[k].tolist())
        done.setdefault("acc", {})[str(s)] = acc[s]
        done["rows"], done["sidecar"] = rows.tolist(), side
        tmp = partial_p.with_suffix(".tmp")
        tmp.write_text(json.dumps(done))
        tmp.replace(partial_p)
        print(f"[t2] sigma={s}: done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    table = []
    for s in SIGMAS:
        for region in ("content", "pad"):
            has = np.asarray(acc[s]["sidecar"][region]["has"], bool)
            for m in METRICS:
                a = np.asarray(acc[s]["sidecar"][region][m])[has]
                b_ = np.asarray(acc[s]["sd0.5"][region][m])[has]
                pb = paired_bootstrap(a, b_, n_boot=10000, seed=1)
                table.append({"sigma": s, "region": region, "metric": m,
                              "sidecar": float(a.mean()), "sd0.5": float(b_.mean()),
                              "delta_sidecar_minus_0.5": pb["delta"], "ci95": pb["ci95"],
                              "p": pb["p"], "n_examples": pb["n"],
                              "rel_delta": pb["delta"] / max(float(b_.mean()), 1e-12)})
    out = {"run_dir": args.run_dir, "checkpoint": str(ckpt.relative_to(REPO) if ckpt.is_relative_to(REPO) else ckpt),
           "checkpoint_info": info, "sidecar_sigma_data": side, "sigmas": SIGMAS,
           "heldout": {"cache_tag": tag, "split": "validation", "n": len(rows), "row_seed": args.seed},
           "self_conditioning": "disabled: x0_hat = zeros on suffix + clean prefix (trainer's no-SC branch), both arms",
           "noise": "eps seeded per (validation row, sigma index); identical across arms",
           "logits_path": "diffusion.continuous.logit_postprocess._model_logits_continuous (matched filter on)",
           "token_counts": {"suffix_mean": float(suffix.sum(1).float().mean()),
                            "content_mean": float(content.sum(1).float().mean()),
                            "pad_mean": float(is_pad.sum(1).float().mean())},
           "minutes": (time.time() - t0) / 60, "table": table}
    p = write_json(f"t2_sigma_data_{args.tag}.json", out)
    for r in table:
        if r["metric"] in METRICS:
            print(f"  s={r['sigma']:<5} {r['region']:<7} {r['metric']}: side {r['sidecar']:.5f} "
                  f"0.5 {r['sd0.5']:.5f}  d {r['delta_sidecar_minus_0.5']:+.5f} "
                  f"[{r['ci95'][0]:+.5f},{r['ci95'][1]:+.5f}]")
    print("->", p)


if __name__ == "__main__":
    main()

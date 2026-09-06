"""Reconstruct the gradient-survival endpoint offline, from saved checkpoints.

WHY THIS EXISTS
---------------
The in-training probe (Trainer._log_objective_probe) was wired into
_step_discrete, but every bitstream task runs framework == "continuous_score"
and is dispatched to _step_continuous. The call site was dead code and the
binary_sm/binary_ce pilot completed with no primary endpoint logged. The wiring
is fixed; this script recovers the same quantity from saved checkpoints.

WHAT IT MEASURES -- TWO SEPARATE THINGS
---------------------------------------
The objectives differ by exactly D(1-D) per bit:

    dL_ce/d_ell = w(sigma) * (D - x0)
    dL_sm/d_ell = w(sigma) * (D - x0) * D(1-D)

1. grad_survival = sum |D-x0| * D(1-D) / sum |D-x0|
       the fraction of CE's gradient magnitude that SM retains.
2. saturation    = fraction of free bits with D(1-D) below a threshold.

These are NOT interchangeable and are reported separately. Measured on the
production run, sigma=0.2 has 99.4% of bits saturated yet survival ~0.13,
because the unsaturated minority carries essentially the whole signal.

Both are properties of the model's D values, not of the loss being optimised,
so they are computed identically in every arm and are directly comparable.

SAMPLING DESIGN
---------------
Two complementary passes, because a full examples x sigma cross-product is not
affordable on one core:

* GRID  -- an explicit sigma list, FULLY PAIRED: every example is evaluated at
  every sigma, under the same fixed noise, for every checkpoint. This is the
  pass to use when comparing checkpoints or arms at a given sigma, because the
  difference carries no example- or noise-sampling variation at all.

* BINS  -- log-sigma bands. Each example draws its own sigma log-uniformly
  inside the band (fixed seed), so a band is summarised over many sigmas rather
  than at one arbitrary point. Cheaper per band, and it answers "how does this
  band behave" instead of "how does this exact sigma behave".

Every draw is seeded, so all arms and checkpoints see identical examples, noise
and sigmas. Repeat with several --noise-seeds to separate signal from noise.

Free (non-prompt) bits only: prompt positions are clamped to clean bits and
carry no gradient, so including them would dilute every statistic by whatever
fraction of the sequence happens to be prompt.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import zlib
from pathlib import Path

import torch

from data.tinygsm import TinyGSMDataset
from diffusion.continuous.logit_postprocess import _model_logits_continuous
from models import create_model
from utils.ema import EMA

SAT_THRESHOLDS = (0.01, 0.001)


def _load_config(path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("probe_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


def _clean(sd: dict) -> dict:
    return {re.sub(r"^_orig_mod\.", "", re.sub(r"^module\.", "", k)): v
            for k, v in sd.items()}


def _load_weights(model, ckpt_path: Path, cfg, device, apply_ema: bool):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(_clean(ckpt["model"]), strict=False)
    used_ema = False
    if apply_ema and ckpt.get("ema") is not None:
        ema = EMA(model, decay=float(getattr(cfg.train, "ema_decay", 0.9999)))
        ema.load_state_dict(ckpt["ema"])
        ema.to(device)
        ema.apply(model)
        used_ema = True
    step = int(ckpt.get("global_step", -1))
    del ckpt
    return step, used_ema


def _gen(*parts) -> torch.Generator:
    """Deterministic generator from a tuple of ints/strings.

    Uses crc32, not hash(): Python string hashing is salted per process, which
    would silently break reproducibility across runs.
    """
    key = "|".join(str(p) for p in parts).encode()
    return torch.Generator().manual_seed(zlib.crc32(key))


def _fixed_examples(cfg, n: int, seed: int):
    ds = TinyGSMDataset(cfg, split="val")
    idx = torch.randperm(len(ds), generator=_gen("examples", seed))[:n].tolist()
    x0 = torch.stack([ds[i]["x0"] for i in idx]).float()
    pm = torch.stack([ds[i]["prefix_mask"] for i in idx]).bool()
    return x0, pm


class _Acc:
    """Pooled accumulator. The ratio is sum/sum over bits, never a mean of
    per-batch ratios, which would weight small batches equally with large ones."""

    def __init__(self):
        self.num = self.den = self.supp = 0.0
        self.bits = 0
        self.sat = {t: 0.0 for t in SAT_THRESHOLDS}

    def add(self, d, t):
        supp = d * (1 - d)
        err = (d - t).abs()
        self.num += float((err * supp).sum())
        self.den += float(err.sum())
        self.supp += float(supp.sum())
        self.bits += int(d.numel())
        for th in SAT_THRESHOLDS:
            self.sat[th] += float((supp < th).sum())

    def as_dict(self):
        n = max(self.bits, 1)
        out = {
            "grad_survival": self.num / max(self.den, 1e-12),
            "mean_D1mD": self.supp / n,
            "n_free_bits": self.bits,
        }
        for th in SAT_THRESHOLDS:
            out[f"frac_D1mD_lt_{th}"] = self.sat[th] / n
        return out


@torch.no_grad()
def _forward_D(model, cfg, xb, mb, sigma, device):
    xt = xb + sigma.view(-1, 1) * _forward_D.eps
    xt[mb] = xb[mb]                     # prompt clamped clean, exactly as in training
    ell = _model_logits_continuous(model, cfg, xt, sigma, None)
    D = torch.sigmoid(ell.float().reshape(xb.shape[0], -1))
    keep = ~mb.reshape(D.shape)
    return D[keep], xb.reshape(D.shape)[keep]


@torch.no_grad()
def probe_grid(model, cfg, x0, pm, sigmas, device, micro_bs, seeds):
    """Fully paired: every example at every sigma, same noise across checkpoints."""
    rows = []
    for si, sig in enumerate(sigmas):
        acc = _Acc()
        for seed in seeds:
            for s in range(0, x0.shape[0], micro_bs):
                xb, mb = x0[s:s + micro_bs].to(device), pm[s:s + micro_bs].to(device)
                _forward_D.eps = torch.randn(
                    xb.shape, generator=_gen("grid", seed, si, s)).to(device)
                d, t = _forward_D(model, cfg, xb, mb,
                                  torch.full((xb.shape[0],), float(sig), device=device),
                                  device)
                acc.add(d, t)
        rows.append({"sigma": float(sig), **acc.as_dict()})
    return rows


@torch.no_grad()
def probe_bins(model, cfg, x0, pm, edges, device, micro_bs, seeds):
    """Per-band: each example draws its own log-uniform sigma inside the band."""
    rows = []
    for bi in range(len(edges) - 1):
        lo, hi = float(edges[bi]), float(edges[bi + 1])
        acc = _Acc()
        for seed in seeds:
            for s in range(0, x0.shape[0], micro_bs):
                xb, mb = x0[s:s + micro_bs].to(device), pm[s:s + micro_bs].to(device)
                g = _gen("binsigma", seed, bi, s)
                u = torch.rand(xb.shape[0], generator=g)
                sig = torch.exp(torch.log(torch.tensor(lo)) +
                                u * (torch.log(torch.tensor(hi)) -
                                     torch.log(torch.tensor(lo)))).to(device)
                _forward_D.eps = torch.randn(
                    xb.shape, generator=_gen("binnoise", seed, bi, s)).to(device)
                d, t = _forward_D(model, cfg, xb, mb, sig, device)
                acc.add(d, t)
        rows.append({"sigma_lo": lo, "sigma_hi": hi, **acc.as_dict()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tasks/tinygsm_bits_objective.py")
    ap.add_argument("--arms", nargs="+", default=["binary_sm", "binary_ce"],
                    help="arm name (resolved as <runs-root>/obj_<name>) or an "
                         "explicit 'name=/path/to/run_dir' pair")
    ap.add_argument("--runs-root", default="runs/tasks/tinygsm")
    ap.add_argument("--out", default="results/objective/offline_probe.json")
    ap.add_argument("--only-steps", nargs="+", type=int, default=None,
                    help="restrict to these global_step values")
    ap.add_argument("--n-grid", type=int, default=128,
                    help="examples for the fully-paired explicit-sigma pass")
    ap.add_argument("--n-bins", type=int, default=64,
                    help="examples per sigma band")
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--data-seed", type=int, default=1234)
    ap.add_argument("--noise-seeds", nargs="+", type=int, default=[5678, 91011])
    ap.add_argument("--sigmas", nargs="+", type=float,
                    default=[0.05, 0.2, 0.4, 1.0, 3.0, 10.0, 40.0])
    ap.add_argument("--bin-edges", nargs="+", type=float,
                    default=[0.002, 0.075, 0.15, 0.30, 0.50, 1.0, 3.0, 10.0, 40.0])
    ap.add_argument("--ema", action="store_true",
                    help="probe EMA weights (default: raw, which is what the "
                         "optimiser's gradient is actually computed from)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ.setdefault("OBJ_LOSS", "binary_sm")
    cfg = _load_config(args.config)

    n_max = max(args.n_grid, args.n_bins)
    x0_all, pm_all = _fixed_examples(cfg, n_max, args.data_seed)
    xg, pg = x0_all[:args.n_grid], pm_all[:args.n_grid]
    xb_, pb_ = x0_all[:args.n_bins], pm_all[:args.n_bins]
    print(f"[probe] examples: grid={args.n_grid} bins={args.n_bins}; "
          f"free bits/example={int((~pm_all[0]).sum())}; seeds={args.noise_seeds}")

    model = create_model(cfg).to(device).eval()

    results = {"meta": {
        "n_grid": args.n_grid, "n_bins": args.n_bins,
        "data_seed": args.data_seed, "noise_seeds": args.noise_seeds,
        "sigmas": args.sigmas, "bin_edges": args.bin_edges,
        "weights": "ema" if args.ema else "raw",
        "grid_design": "fully paired: every example at every sigma",
        "bin_design": "per-example log-uniform sigma within the band",
    }, "arms": {}}

    for spec in args.arms:
        if "=" in spec:
            arm, run_dir = spec.split("=", 1)
            ckdir = Path(run_dir) / "checkpoints"
        else:
            arm, ckdir = spec, Path(args.runs_root) / f"obj_{spec}" / "checkpoints"
        cks = sorted(ckdir.glob("step=*.pt"))
        if (ckdir / "last.pt").exists():
            cks.append(ckdir / "last.pt")
        if not cks:
            print(f"[probe] WARNING: no checkpoints under {ckdir}")
            continue
        results["arms"][arm] = []
        for ck in cks:
            step, used_ema = _load_weights(model, ck, cfg, device, args.ema)
            if args.only_steps and step not in args.only_steps:
                continue
            grid = probe_grid(model, cfg, xg, pg, args.sigmas, device,
                              args.micro_bs, args.noise_seeds)
            bins = probe_bins(model, cfg, xb_, pb_, args.bin_edges, device,
                              args.micro_bs, args.noise_seeds)
            results["arms"][arm].append({
                "checkpoint": ck.name, "global_step": step,
                "ema_applied": used_ema, "grid": grid, "bins": bins,
            })
            lo = grid[0]
            print(f"[probe] {arm:11s} step={step:>7d}  "
                  f"survival@s={lo['sigma']}: {lo['grad_survival']:.4f}  "
                  f"sat@s={lo['sigma']}: {lo['frac_D1mD_lt_0.001']:.4f}")
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(results, indent=2))

    print(f"[probe] wrote {args.out}")


if __name__ == "__main__":
    main()

"""Reconstruct the gradient-survival endpoint offline, from saved checkpoints.

WHY THIS EXISTS
---------------
The in-training probe (Trainer._log_objective_probe) was wired into
_step_discrete, but every bitstream task runs framework == "continuous_score"
and is dispatched to _step_continuous. The call site was therefore dead code and
the binary_sm/binary_ce pilot completed with no primary endpoint logged. The
wiring is fixed, but rather than pay 98 GPU-h again this script recovers the
same quantity from the milestone checkpoints the pilot did save.

WHAT IT MEASURES
----------------
The two objectives differ by exactly D(1-D) per bit:

    dL_ce/d_ell = w(sigma) * (D - x0)
    dL_sm/d_ell = w(sigma) * (D - x0) * D(1-D)

    grad_survival = sum|D - x0| * D(1-D) / sum|D - x0|

It is a property of the model's D values, not of the loss being optimised, so it
is computed identically in both arms and they are directly comparable.

WHY THIS IS BETTER THAN THE IN-TRAINING PROBE
---------------------------------------------
Offline, every checkpoint of every arm sees the SAME validation examples, the
SAME Gaussian noise and the SAME sigma grid (all fixed-seed). The measurement is
fully paired, so arm-to-arm and step-to-step differences carry no batch-sampling
noise -- the in-training version would have compared different random draws.
Sweeping an explicit sigma grid also resolves the suppression against sigma,
which an aggregate over a log-uniform draw hides.

Free (non-prompt) bits only: prompt positions are clamped to clean bits and
carry no gradient, so including them would dilute every statistic by whatever
fraction of the sequence happens to be prompt.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch

from data.tinygsm import TinyGSMDataset
from diffusion.continuous.logit_postprocess import _model_logits_continuous
from models import create_model
from utils.ema import EMA


def _load_config(path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("probe_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


def _clean(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        k = re.sub(r"^module\.", "", k)
        k = re.sub(r"^_orig_mod\.", "", k)
        out[k] = v
    return out


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


def _fixed_batch(cfg, n: int, seed: int):
    """A fixed set of validation examples, identical for every checkpoint."""
    ds = TinyGSMDataset(cfg, split="val")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    x0 = torch.stack([ds[i]["x0"] for i in idx]).float()
    pm = torch.stack([ds[i]["prefix_mask"] for i in idx]).bool()
    return x0, pm


@torch.no_grad()
def probe_checkpoint(model, cfg, x0, pm, sigmas, device, micro_bs: int, noise_seed: int):
    """grad_survival and saturation stats per sigma, on one set of weights."""
    rows = []
    for sig in sigmas:
        num = den = 0.0
        n_bits = 0
        sat_01 = sat_001 = 0.0
        supp_sum = 0.0
        for s in range(0, x0.shape[0], micro_bs):
            xb = x0[s:s + micro_bs].to(device)
            mb = pm[s:s + micro_bs].to(device)
            # Same noise for every checkpoint and both arms.
            g = torch.Generator(device="cpu").manual_seed(noise_seed + s)
            eps = torch.randn(xb.shape, generator=g).to(device)

            sigma = torch.full((xb.shape[0],), float(sig), device=device)
            xt = xb + sigma.view(-1, 1) * eps
            xt[mb] = xb[mb]          # prompt is clamped clean, exactly as in training

            ell = _model_logits_continuous(model, cfg, xt, sigma, None)
            D = torch.sigmoid(ell.float().reshape(xb.shape[0], -1))
            keep = ~mb.reshape(D.shape)
            d, t = D[keep], xb.reshape(D.shape)[keep]

            supp = d * (1 - d)
            err = (d - t).abs()
            num += float((err * supp).sum())
            den += float(err.sum())
            supp_sum += float(supp.sum())
            sat_01 += float((supp < 0.01).sum())
            sat_001 += float((supp < 0.001).sum())
            n_bits += int(keep.sum())

        rows.append({
            "sigma": float(sig),
            "grad_survival": num / max(den, 1e-12),
            "mean_D1mD": supp_sum / max(n_bits, 1),
            "frac_D1mD_lt_0.01": sat_01 / max(n_bits, 1),
            "frac_D1mD_lt_0.001": sat_001 / max(n_bits, 1),
            "n_free_bits": n_bits,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tasks/tinygsm_bits_objective.py")
    ap.add_argument("--arms", nargs="+", default=["binary_sm", "binary_ce"],
                    help="arm name (resolved under --runs-root as obj_<name>) or "
                         "an explicit 'name=/path/to/run_dir' pair, which lets the "
                         "same probe run against the production run for comparison")
    ap.add_argument("--runs-root", default="runs/tasks/tinygsm")
    ap.add_argument("--out", default="results/objective/offline_probe.json")
    ap.add_argument("--n-examples", type=int, default=128)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--data-seed", type=int, default=1234)
    ap.add_argument("--noise-seed", type=int, default=5678)
    ap.add_argument("--sigmas", nargs="+", type=float,
                    default=[0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6])
    ap.add_argument("--ema", action="store_true",
                    help="probe EMA weights (default: raw, which is what the "
                         "gradient the optimiser actually sees is computed from)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ.setdefault("OBJ_LOSS", args.arms[0])
    cfg = _load_config(args.config)

    x0, pm = _fixed_batch(cfg, args.n_examples, args.data_seed)
    print(f"[probe] fixed eval set: {tuple(x0.shape)}, "
          f"free bits/example={int((~pm[0]).sum())}")

    model = create_model(cfg).to(device).eval()

    results = {"meta": {
        "n_examples": args.n_examples, "data_seed": args.data_seed,
        "noise_seed": args.noise_seed, "sigmas": args.sigmas,
        "weights": "ema" if args.ema else "raw",
        "paired": "identical data, noise and sigma across all arms/checkpoints",
    }, "arms": {}}

    for spec in args.arms:
        if "=" in spec:
            arm, run_dir = spec.split("=", 1)
            ckdir = Path(run_dir) / "checkpoints"
        else:
            arm = spec
            ckdir = Path(args.runs_root) / f"obj_{arm}" / "checkpoints"
        cks = sorted(ckdir.glob("step=*.pt"))
        last = ckdir / "last.pt"
        if last.exists():
            cks.append(last)
        if not cks:
            print(f"[probe] WARNING: no checkpoints under {ckdir}")
            continue
        results["arms"][arm] = []
        for ck in cks:
            step, used_ema = _load_weights(model, ck, cfg, device, args.ema)
            rows = probe_checkpoint(model, cfg, x0, pm, args.sigmas,
                                    device, args.micro_bs, args.noise_seed)
            agg = sum(r["grad_survival"] for r in rows) / len(rows)
            results["arms"][arm].append({
                "checkpoint": ck.name, "global_step": step,
                "ema_applied": used_ema,
                "grad_survival_mean_over_sigma": agg,
                "by_sigma": rows,
            })
            print(f"[probe] {arm:10s} {ck.name:22s} step={step:>7d} "
                  f"grad_survival(mean over sigma)={agg:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[probe] wrote {out}")


if __name__ == "__main__":
    main()

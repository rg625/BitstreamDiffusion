"""Forward-only calibration of a frozen CoBit checkpoint for the gamma-dependent Euler grid (GPU; no JVPs).

    python scripts/dgc/profile_forward_increments.py --config CONFIG --data CLEAN.pt --out PREFIX
        [--n_seq 2048] [--K 64] [--sigma_lo 0.08] [--sigma_hi 80] [--closure null|two_pass] [--batch 64]
        [--no_amp] [--seed 0] [--nested_K 128]

The denoiser is the sampler's public one: _model_logits_continuous (matched-filter postprocessing included) followed
by the sigmoid, with EMA weights as in evaluation. Self-conditioning closure:
  null      one call per evaluation, self-condition input zeros (the sampler's initial state);
  two_pass  D0 = D(x, sigma | 0), then D(x, sigma | D0): a deterministic current-state closure, two calls.
CLEAN.pt holds clean sequences in the model's 0/1 bit coordinates: a float/uint8 tensor [M, d], or a dict with "x0"
and optionally "gen_mask" (bool [M, d]; prompted tasks noise only generated coordinates) and "ids". Held-out data is
the caller's responsibility; the JSON sidecar records its hash.

Writes PREFIX.npz (log_sigma_edges, A, B, q_increment, cross_T, noise_V, mse_edges, *_per_sequence, *_ci_lo/hi) and
PREFIX.json (checkpoint hash, EMA, closure, band, seed, precision, sizes, repository SHA). With --nested_K, a second
profile on a refined grid reuses the same clean sequences (a fresh Brownian path, same seed stream).
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from diffusion.continuous.forward_profile import STATS, bootstrap_ci, profile_batch  # noqa: E402
from diffusion.continuous.logit_postprocess import _model_logits_continuous  # noqa: E402
from evaluation.utils import load_checkpoint, load_config  # noqa: E402
from models import create_model  # noqa: E402
from utils.ema import EMA  # noqa: E402


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def make_denoiser(model, cfg, closure: str, amp: bool):
    def call(x, sigma, sc):
        sig = torch.full((x.shape[0],), float(sigma), device=x.device, dtype=torch.float32)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=amp):
            logits = _model_logits_continuous(model, cfg, x, sig, sc)
        return torch.sigmoid(logits.float())

    def denoise(x, sigma, sc0, gen):
        # sc0: the sampler's initial self-condition state (zeros, with the prompt clamped in on prompted tasks);
        # two_pass feeds D0 back with the prompt clamped, as the sampler's carry does
        d0 = call(x, sigma, sc0)
        if closure == "null":
            return d0
        return call(x, sigma, d0 if gen is None else torch.where(gen, d0, sc0))
    return denoise


def run_profile(denoise, x0_all, mask_all, edges_u, batch, seed, device):
    t_edges = torch.exp(2.0 * torch.as_tensor(edges_u, dtype=torch.float64))
    g = torch.Generator(device=device).manual_seed(int(seed))
    parts = []
    for i in range(0, len(x0_all), batch):
        x0 = x0_all[i:i + batch].to(device)
        mk = None if mask_all is None else mask_all[i:i + batch].to(device)
        sc0 = torch.zeros_like(x0) if mk is None else torch.where(mk.bool(), torch.zeros_like(x0), x0)
        gen = None if mk is None else mk.bool()
        parts.append(profile_batch(lambda x, sig, sc0=sc0, gen=gen: denoise(x, sig, sc0, gen), x0, t_edges,
                                   gen_mask=mk, generator=g))
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def save(prefix, edges_u, st, meta):
    arrays = dict(log_sigma_edges=np.asarray(edges_u, np.float64),
                  A=st["A"].mean(0).numpy(), B=st["B"].mean(0).numpy(),
                  q_increment=st["q"].mean(0).numpy(), cross_T=st["T"].mean(0).numpy(),
                  noise_V=st["V"].mean(0).numpy(), mse_edges=st["mse"].mean(0).numpy())
    for k in STATS:
        arrays[f"{k}_per_sequence"] = st[k].numpy()
        lo, hi = bootstrap_ci(st[k])
        arrays[f"{k}_ci_lo"], arrays[f"{k}_ci_hi"] = lo.numpy(), hi.numpy()
    np.savez(f"{prefix}.npz", **arrays)
    meta = dict(meta, profile_sha256=sha256(f"{prefix}.npz"))
    Path(f"{prefix}.json").write_text(json.dumps(meta, indent=2))
    bad = [k for k in ("A", "B") if not np.all(np.isfinite(arrays[k]))]
    if bad:
        raise RuntimeError(f"non-finite {bad} in {prefix}.npz: implementation failure, not a profile")
    print(f"wrote {prefix}.npz  A in [{arrays['A'].min():.4g}, {arrays['A'].max():.4g}]  "
          f"B in [{arrays['B'].min():.4g}, {arrays['B'].max():.4g}]  mean V {arrays['noise_V'].mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None, help="default: cfg.evaluation.checkpoint_path")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_seq", type=int, default=2048)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--nested_K", type=int, default=0)
    ap.add_argument("--sigma_lo", type=float, default=None, help="default: the config's terminal sigma")
    ap.add_argument("--sigma_hi", type=float, default=None, help="default: cfg.diffusion.continuous.sigma_max")
    ap.add_argument("--closure", default="null", choices=["null", "two_pass"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--no_ema", action="store_true")
    ap.add_argument("--task", action="store_true",
                    help="prompted task checkpoint: load it and resolve sigma_data exactly as evaluation/tasks does")
    ap.add_argument("--sigma_data", type=float, default=None, help="--task only: the eval's --sigma_data override")
    a = ap.parse_args()

    cfg = load_config(a.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(a.checkpoint or cfg.evaluation.checkpoint_path)
    sigma_data = None
    if a.task:
        from evaluation.tasks._task_common import load_model_and_sampler, resolve_sigma_data
        sigma_data, _ = resolve_sigma_data(cfg, ckpt.resolve().parent.parent, a.sigma_data)
        model, _ = load_model_and_sampler(cfg, str(ckpt), device, apply_ema=not a.no_ema)
    else:
        model = create_model(cfg).to(device)
        ema = EMA(model, decay=0.0)
        load_checkpoint(model, ema, ckpt, device, apply_ema=not a.no_ema)
    model.eval()

    ext = getattr(cfg.evaluation, "external_ppl", None)
    term = (list(getattr(ext, "terminal_sigmas", [])) or [None])[0] if ext is not None else None
    lo = a.sigma_lo if a.sigma_lo is not None else (term if term else float(cfg.diffusion.continuous.sigma_min))
    hi = a.sigma_hi if a.sigma_hi is not None else float(cfg.diffusion.continuous.sigma_max)

    data = torch.load(a.data, map_location="cpu", weights_only=False)
    x0_all = (data["x0"] if isinstance(data, dict) else data)[: a.n_seq].float()
    mask_all = data.get("gen_mask")[: a.n_seq] if isinstance(data, dict) and "gen_mask" in data else None
    if not torch.all((x0_all == 0) | (x0_all == 1)):
        raise ValueError("clean sequences must be in the model's 0/1 bit coordinates")

    denoise = make_denoiser(model, cfg, a.closure, amp=not a.no_amp)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    for K, suffix in [(a.K, "")] + ([(a.nested_K, f"_K{a.nested_K}")] if a.nested_K else []):
        edges_u = np.linspace(math.log(lo), math.log(hi), K + 1)
        t0 = time.time()
        st = run_profile(denoise, x0_all, mask_all, edges_u, a.batch, a.seed, device)
        meta = dict(config=a.config, checkpoint=str(ckpt), checkpoint_sha256=sha256(ckpt), ema=not a.no_ema,
                    data=a.data, data_sha256=sha256(a.data), n_seq=len(x0_all), d=int(x0_all.shape[1]),
                    gen_mask=mask_all is not None, K=K, sigma_lo=lo, sigma_hi=hi, closure=a.closure,
                    model_calls_per_sequence=(K + 1) * (1 if a.closure == "null" else 2), seed=a.seed,
                    precision="bf16 autocast, float64 statistics" if not a.no_amp else "fp32, float64 statistics",
                    repository_sha=git, estimator="direct residual increments v1", task=a.task,
                    sigma_data=sigma_data, seconds=round(time.time() - t0, 1))
        save(f"{a.out}{suffix}", edges_u, st, meta)


if __name__ == "__main__":
    main()

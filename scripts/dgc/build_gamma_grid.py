"""Build sigma grids for the gamma-dependent Euler study (CPU).

    python scripts/dgc/build_gamma_grid.py ARM --out GRIDS.npz --intervals 63,127,255 [--sigma_lo 0.08] [--sigma_hi 80]
        [--profile PROFILE.npz] [--table assets/entropy_tables/lm1b] [--gamma 0.2 | --S 1] [--eps 0]

ARM:
  log_uniform     uniform in log sigma
  karras          the EDM grid, rho = 7
  table           the entropy-rate table (--table) with the CDF paired to bin EDGES and inverted in log sigma, on
                  [sigma_lo, sigma_hi] (the corrected version of the legacy midpoint / linear-sigma inversion)
  dgc_profile     ordinary DGC from the forward profile: density ∝ sqrt(q_increment)
  gamma           the gamma-dependent Euler density sqrt(A / (S^2 B + eta)) from the forward profile; S = N * gamma
                  (--gamma, realized per-interval churn) or fixed S (--S, theory sweep: gamma = S / N)

Each interval count N gets its own grid, key "N<N>", N + 1 descending sigmas with the requested endpoints; the
sampler's grid_file schedule reads them (num_steps = N + 1 points; one model call per interval plus the final
decode, so N + 1 calls with the null/carry closure). A JSON sidecar records the arm, its inputs and their hashes, and
per-grid S, eta, maximum log step and the lambda range. Never reuse a gamma grid for another (N, gamma).
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from diffusion.continuous.gamma_grid import (  # noqa: E402
    finite_check_grid, gamma_euler_grid, grid_from_density, karras_grid, log_uniform_grid, ordinary_dgc_grid,
    restrict_density, table_density,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["log_uniform", "karras", "table", "dgc_profile", "gamma"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--intervals", required=True, help="comma list of N (intervals, not NFE)")
    ap.add_argument("--sigma_lo", type=float, default=0.08)
    ap.add_argument("--sigma_hi", type=float, default=80.0)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--table", default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--S", type=float, default=None)
    ap.add_argument("--eps", type=float, default=0.0)
    a = ap.parse_args()
    Ns = [int(x) for x in a.intervals.split(",")]
    grids, info = {}, dict(arm=a.arm, sigma_lo=a.sigma_lo, sigma_hi=a.sigma_hi, eps=a.eps, grids={})
    prof = None
    if a.arm in ("dgc_profile", "gamma"):
        if not a.profile:
            raise SystemExit(f"arm {a.arm} needs --profile")
        prof = dict(np.load(a.profile))
        info.update(profile=a.profile, profile_sha256=sha(a.profile))
        e = prof["log_sigma_edges"]
        if abs(e[0] - math.log(a.sigma_lo)) > 1e-9 or abs(e[-1] - math.log(a.sigma_hi)) > 1e-9:
            raise SystemExit(f"profile band [{math.exp(e[0]):.4g}, {math.exp(e[-1]):.4g}] != requested "
                             f"[{a.sigma_lo}, {a.sigma_hi}]: recalibrate or pass the profile's band")
    if a.arm == "gamma" and (a.gamma is None) == (a.S is None):
        raise SystemExit("arm gamma needs exactly one of --gamma (realized churn) or --S (fixed S)")
    if a.arm == "table":
        if not a.table:
            raise SystemExit("arm table needs --table")
        ld = lambda f: torch.load(f"{a.table}/{f}", map_location="cpu", weights_only=True).double().numpy()
        e_t, p_t = table_density(ld("entropy_edges.pt"), ld("entropy_pdf.pt"))
        e_b, p_b = restrict_density(e_t, p_t, math.log(a.sigma_lo), math.log(a.sigma_hi))
        info.update(table=a.table, table_pdf_sha256=sha(f"{a.table}/entropy_pdf.pt"))
    for N in Ns:
        if a.arm == "log_uniform":
            s, gi = log_uniform_grid(a.sigma_lo, a.sigma_hi, N), {}
        elif a.arm == "karras":
            s, gi = karras_grid(a.sigma_lo, a.sigma_hi, N), {}
        elif a.arm == "table":
            s, gi = grid_from_density(e_b, p_b, N), {}
        elif a.arm == "dgc_profile":
            s, gi = ordinary_dgc_grid(prof, N), {}
        else:
            gamma = a.gamma if a.gamma is not None else a.S / N
            s, gi = gamma_euler_grid(prof, N, gamma, eps=a.eps)
        finite_check_grid(s, N, a.sigma_lo, a.sigma_hi)
        d = np.log(s[:-1] / s[1:])
        gi.update(points=len(s), max_log_step=float(d.max()), min_log_step=float(d.min()),
                  grid_sha256=hashlib.sha256(s.tobytes()).hexdigest())
        grids[f"N{N}"], info["grids"][f"N{N}"] = s, gi
    np.savez(a.out, **grids)
    Path(a.out).with_suffix(".json").write_text(json.dumps(info, indent=2))
    for k, gi in info["grids"].items():
        print(k, {kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in gi.items() if kk != "grid_sha256"})


if __name__ == "__main__":
    main()

"""TASK 7(c): zero / near-zero bins of the learned entropy-rate pdf (audit T8).

For each run, lists the contiguous runs of bins whose pdf mass is exactly 0 or
below a near-zero threshold (1e-3 of the peak bin), with the sigma range they
cover (bin edges). The trainer's sigma range and gamma_max are reported
alongside: with gamma_max = 1.0, a zero bin is never drawn again and cannot
refill (utils/schedule_controller.py), so a zero run inside the training range
is a sigma band the model stopped being trained on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.cpu_reanalysis.common import REPO, cpu_guard, write_json  # noqa: E402

torch = cpu_guard()

RUNS = {
    "production": "tinigsm_gsm8k/runs/cobit_raw_binary_bits_cfg",
    "bits_control": "runs/tasks/tinygsm/ord_gh500k_none_s42",
    "token": "runs/tasks/tinygsm/tok_gh500k_token_ce_s42",
}
NEAR = 1e-3


def runs_of(mask):
    out, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def main():
    res = {}
    for name, rd in RUNS.items():
        rd = REPO / rd
        pdf = torch.load(rd / "entropy_pdf.pt", map_location="cpu", weights_only=True).double()
        edges = torch.load(rd / "entropy_edges.pt", map_location="cpu", weights_only=True).double()
        cfg = json.loads((rd / "config.json").read_text())
        dc, tr = cfg["diffusion"]["continuous"], cfg["train"]
        smin = float(dc.get("sigma_min", edges[0])); smax = float(dc.get("sigma_max", edges[-1]))
        peak = float(pdf.max())

        def describe(lo, hi):
            a, b = float(edges[lo]), float(edges[hi + 1])
            return {"bins": [lo, hi], "n_bins": hi - lo + 1, "sigma_lo": a, "sigma_hi": b,
                    "mass": float(pdf[lo:hi + 1].sum()),
                    "inside_train_range": bool(b > smin and a < smax)}

        zero = [describe(lo, hi) for lo, hi in runs_of((pdf == 0).tolist())]
        near = [describe(lo, hi) for lo, hi in runs_of(((pdf > 0) & (pdf < NEAR * peak)).tolist())]
        nz = (pdf > 0).nonzero().flatten()
        res[name] = {
            "run_dir": str(rd.relative_to(REPO)), "n_bins": int(pdf.numel()),
            "sigma_min": smin, "sigma_max": smax,
            "gamma_max": tr.get("entropy_gamma_max", tr.get("gamma_max")),
            "peak_mass": peak, "peak_sigma": float(edges[int(pdf.argmax())]),
            "support_sigma": [float(edges[int(nz[0])]), float(edges[int(nz[-1]) + 1])],
            "n_zero_bins": int((pdf == 0).sum()), "zero_runs": zero,
            "near_zero_threshold": NEAR * peak, "near_zero_runs": near,
        }
        print(f"{name}: {res[name]['n_zero_bins']} zero bins; support sigma "
              f"[{res[name]['support_sigma'][0]:.4g}, {res[name]['support_sigma'][1]:.4g}]; "
              f"train range [{smin:.4g}, {smax:.4g}], gamma_max={res[name]['gamma_max']}")
        for z in zero:
            print(f"   zero  bins {z['bins']}  sigma [{z['sigma_lo']:.4g}, {z['sigma_hi']:.4g}]"
                  f"  inside train range={z['inside_train_range']}")
        for z in near:
            print(f"   near0 bins {z['bins']}  sigma [{z['sigma_lo']:.4g}, {z['sigma_hi']:.4g}]  mass {z['mass']:.2e}")
    print("->", write_json("t7_entropy_bins.json", res))


if __name__ == "__main__":
    main()

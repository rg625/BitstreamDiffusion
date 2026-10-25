#!/usr/bin/env python
"""
Analysis of the paired generation trajectories (trajectories.py): where along the sampler do the CoBit-S
and CoBit-M trajectories separate, and how does that compare with the static denoising curves?

Inputs  runs/denoising_mse/traj/<tag>/chunk*.npz (+ scores.json)   paired S vs M runs
        runs/denoising_mse/traj/<local>/chunk*.npz                 same-input decomposition (optional)
        experiments/denoising_mse/results/main/curves.npz          static MSE curves (optional overlay)
Outputs experiments/denoising_mse/results/traj/: fig_traj_distance.png, fig_traj_vs_rate.png,
        traj_curves.csv, traj_summary.json

Per sampler step i (254 churned Euler steps + the final denoise), across examples:
  state gap         sqrt(dx2): RMS per bit of x_S - x_M after step i (at sigma_{i+1})
  relative gap      sqrt(dx2) / sigma_{i+1}: the gap in units of the remaining noise
  denoiser gap      dD2 = mean_bits (D_S - D_M)^2, each model on its own trajectory
  same-input gap    ldD2 = mean_bits (D_S - D_M(S's input))^2 along S's trajectory (decomposition)
  token disagreement of the rounded denoiser predictions (trajectory / same input)
Bands: mean with the central 68% and 95% of examples shaded (±1 / ±2 sd equivalents; the spread of
individual trajectories). traj_curves.csv also has the sd and the standard error of the mean, sd/sqrt(N).

Densities over log sigma (unit area on sigma >= 0.08) compared in fig_traj_vs_rate.png:
  divergence density   growth per unit (-log sigma) of the mean token disagreement and of the mean squared
                       state gap: where along the sampler the two generations become different
  same-input rate      ldD2 / sigma_hat^2: disagreement of the two denoisers on identical inputs (S's
                       states), the on-trajectory analogue of the static rates; with shared noise the state
                       gap obeys δx_{i+1} = r_i δx_i + (1 - r_i) δD_i (r_i = sigma_{i+1}/sigma_hat_i), so
                       δD^2/sigma^2 is the squared gap injected per unit log sigma while the runs are close
  static rates         MSE_S/sigma^2, MSE_M/sigma^2 and (MSE_S - MSE_M)/sigma^2 from results/main
(The disagreement rate along each model's own trajectory is not used as a density: once the runs have
committed to different samples D_S - D_M stays O(1) and dividing by sigma^2 just grows.)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
C = {"traj": "#3d3d3a", "local": "#4a3aa7", "S": "#2a78d6", "M": "#eb6834", "surplus": "#9a9994",
     "rate": "#1baf7a"}

# Task switch (default OWT): DMSE_TASK=tinygsm runs the identical analysis on the TinyGSM runs.
import os as _os
TASK = _os.environ.get("DMSE_TASK", "owt").lower()
TASKCFG = {
    "owt": dict(results="results", table_S="assets/entropy_tables/owt", sig_lo=0.08, name="OWT"),
    "tinygsm": dict(results="results_tinygsm", table_S="runs/tasks/tinygsm/cobit_raw_binary_bits", sig_lo=0.1,
                    name="TinyGSM", runs="runs/denoising_mse/tinygsm",
                    tagmap={"main": "traj_paired_SM", "local": "traj_local_S", "localM": "traj_local_M",
                            "truechain": "traj_true_SM"}),
}[TASK]
SIG_LO = TASKCFG["sig_lo"]


def load(tag):
    if TASK == "tinygsm":
        fs = sorted(glob.glob(str(REPO / TASKCFG["runs"] / TASKCFG["tagmap"].get(tag, tag) / "chunk*.npz")))
    else:
        fs = sorted(glob.glob(str(REPO / f"runs/denoising_mse/traj/{tag}/chunk*.npz")))
    if not fs:
        return None
    zs = [np.load(f) for f in fs]
    out = {k: np.concatenate([z[k] for z in zs], 1) for k in zs[0].files
           if zs[0][k].ndim == 2 and not k.startswith("bits_")}
    for k in ("sigma_state", "gamma", "sigmas"):
        if k in zs[0].files:
            out[k] = zs[0][k]
    if "final_tokdis" in zs[0].files:
        out["final_tokdis"] = np.concatenate([z["final_tokdis"] for z in zs])
    out["examples"] = np.concatenate([z["examples"] for z in zs])
    for k in ("correct_a", "correct_b"):  # TinyGSM paired runs: per-problem GSM8K correctness
        if k in zs[0].files:
            out[k] = np.concatenate([z[k] for z in zs])
    if "gamma" not in out and "sigma_state" in out and "sigmas" in out:  # TinyGSM runs: per-step churn from the sigmas
        sh, sg = out["sigma_state"].astype(float), out["sigmas"].astype(float)
        out["gamma"] = np.r_[sh[:-1] / sg[:len(sh) - 1] - 1.0, 0.0]
    for a, b in (("HA", "HS"), ("HB", "HM")):  # TinyGSM key names for each model's prediction entropy
        if a in out and b not in out:
            out[b] = out.pop(a)
    return out


def entropy_time(sig, table=TASKCFG["table_S"]):
    import torch

    c = torch.load(REPO / table / "entropy_cdf.pt", map_location="cpu").numpy().astype(float)
    s = torch.load(REPO / table / "entropy_sigmas.pt", map_location="cpu").numpy().astype(float)
    c[-1] = 1.0
    return np.interp(np.log(sig), np.log(s), c, left=0.0, right=1.0)


def style(ax):
    ax.grid(True, color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8984")
    ax.tick_params(colors="#52514e", labelsize=8)


def band(ax, x, a, color, label, lw=1.8):
    """Mean line with the central 68% and 95% of examples shaded (the ±1 sd / ±2 sd equivalents for a
    Gaussian; percentiles keep the bands inside the range of these non-negative quantities)."""
    mu = a.mean(1)
    q = np.percentile(a, [2.5, 16, 84, 97.5], axis=1)
    ax.fill_between(x, q[0], q[3], color=color, alpha=0.10, lw=0)
    ax.fill_between(x, q[1], q[2], color=color, alpha=0.24, lw=0)
    ax.plot(x, mu, color=color, lw=lw, label=label)
    return mu


def slope(y, x, k=3):
    """Centred finite difference over ±k steps (x increasing), one-sided at the ends."""
    n = len(y)
    out = np.empty(n)
    for i in range(n):
        a, b = max(0, i - k), min(n - 1, i + k)
        out[i] = (y[b] - y[a]) / (x[b] - x[a])
    return out


def first_cross(x, y, level):
    i = np.argmax(y >= level)
    return float(x[i]) if y[i] >= level else None


def unit_area(y, ls, act):
    y = np.where(act, np.clip(y, 0, None), 0.0)
    order = np.argsort(ls)
    return y / np.trapz(y[order], ls[order])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--local-tag", default="local")
    ap.add_argument("--static", default=str(HERE / TASKCFG["results"] / ("main" if TASK == "owt" else "static") / "curves.npz"))
    ap.add_argument("--out", default=str(HERE / TASKCFG["results"] / "traj"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    T = load(args.tag)
    L = load(args.local_tag)
    N = T["dx2"].shape[1]
    sig_hat = T["sigma_state"].astype(float)                         # sigma at which D was evaluated
    sigmas = T["sigmas"].astype(float)
    sig_after = np.r_[sigmas[1:], sigmas[-1]]                        # state after step i; final: unchanged
    t = entropy_time(sig_hat)
    gap = np.sqrt(T["dx2"])
    rel = gap / sig_after[:, None]
    if L is not None:
        common = np.intersect1d(T["examples"], L["examples"])
        lsel = np.searchsorted(L["examples"], common)
        tsel = np.searchsorted(T["examples"], common)
    scores = {}
    sp = REPO / f"runs/denoising_mse/traj/{args.tag}/scores.json"
    if sp.exists():
        scores = json.loads(sp.read_text())
    if "correct_a" in T:  # TinyGSM: GSM8K accuracy of the paired samples instead of GenPPL
        scores = {"gsm8k_accuracy": {"S": float(T["correct_a"].mean()), "M": float(T["correct_b"].mean())}}

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})

    # ---------------- Figure 1: distances along the sampler ----------------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.6), sharex=True)
    ax = axs[0, 0]
    band(ax, t, gap, C["traj"], "S vs M trajectories")
    ax.set_ylabel("state gap  RMS(x_S − x_M) per bit")
    ax.set_title("(a) State gap", loc="left")
    ax = axs[0, 1]
    band(ax, t, rel, C["traj"], "S vs M trajectories")
    ax.set_yscale("log")
    ax.set_ylabel("relative gap  RMS(x_S − x_M) / σ")
    ax.set_title("(b) State gap relative to the remaining noise", loc="left")
    if TASK == "tinygsm":  # the TinyGSM sampler integrates to sigma = 0.002, where dividing by sigma explodes
        ax.set_ylim(1e-4, 10)
    ax.axhline(1, color="#8a8984", lw=0.8, ls="--")
    ax = axs[0, 2]
    band(ax, t, T["HS"], C["S"], "CoBit-S")
    band(ax, t, T["HM"], C["M"], "CoBit-M")
    ax.set_ylabel("mean binary entropy of D [bits]")
    ax.set_title("(c) How committed each model's prediction is", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1, 0]
    band(ax, t, T["dD2"], C["traj"], "own trajectories")
    if L is not None:
        band(ax, t, L["ldD2"][:, lsel], C["local"], "same input (S's trajectory)")
    ax.set_ylabel("denoiser gap  mean (D_S − D_M)²")
    ax.set_title("(d) Denoiser disagreement", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axs[1, 1]
    band(ax, t, T["tokdis"], C["traj"], "own trajectories")
    if L is not None:
        band(ax, t, L["ltokdis"][:, lsel], C["local"], "same input (S's trajectory)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of tokens predicted differently")
    ax.set_title("(e) Token disagreement of the predictions", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axs[1, 2]
    rate_traj = T["dD2"] / sig_hat[:, None] ** 2
    if L is not None:
        band(ax, t, L["ldD2"][:, lsel] / sig_hat[:, None] ** 2, C["local"], "same input (S's trajectory)")
        ax.set_yscale("log")
    ax.set_ylabel("same-input disagreement rate  (D_S − D_M)² / σ²")
    ax.set_title("(f) Same-input disagreement rate (analogue of MSE/σ²)", loc="left")
    if TASK == "tinygsm":
        ax.set_ylim(1e-8, None)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axs.flat:
        style(ax)
        ax.set_xlim(1.0, 0.0)
    for ax in axs[1]:
        ax.set_xlabel("diffusion time t  (CoBit-S schedule; sampling runs left → right)")
    sc = ""
    if "gsm8k_accuracy" in scores:
        sc = (f";  GSM8K accuracy S {100 * scores['gsm8k_accuracy']['S']:.1f}%, M {100 * scores['gsm8k_accuracy']['M']:.1f}%")
    if "S" in scores and "M" in scores:
        sc = (f";  GenPPL S {scores['S']['genppl']:.1f} (ent {scores['S']['mean_entropy']:.2f}), "
              f"M {scores['M']['genppl']:.1f} (ent {scores['M']['mean_entropy']:.2f})")
    head = ("Paired generation, same sampler and random numbers (γ = 0.13, 256 NFE, CoBit-S schedule), " if TASK == "owt" else
            "Paired generation on GSM8K prompts, same sampler and random numbers (γ = 0.31, 256 NFE, CoBit-S schedule), ")
    fig.suptitle(head + f"N = {N}{sc}. Bands: central 68% / 95% of examples.", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_traj_distance.png", dpi=160)
    plt.close(fig)

    # ---------------- Figure 2: rates over log sigma vs the static curves --------------------------
    ls_hat = np.log(sig_hat)
    act = sig_hat >= SIG_LO
    steps = slice(0, len(sig_hat) - 1)  # churned Euler steps; the final denoise has no sigma step of its own
    xs_ = -ls_hat[steps]                 # increases along sampling
    div_tok = slope(T["tokdis"].mean(1)[steps], xs_)
    div_gap = slope(T["dx2"].mean(1)[steps], xs_)
    curves = {
        "divergence density: growth of token disagreement": (unit_area(div_tok, ls_hat[steps], act[steps]),
                                                             C["traj"], "-", 2.2, sig_hat[steps]),
        "divergence density: growth of squared state gap": (unit_area(div_gap, ls_hat[steps], act[steps]),
                                                            C["traj"], ":", 1.6, sig_hat[steps]),
    }
    if L is not None:
        curves["same-input disagreement rate  (D_S − D_M)²/σ²"] = (
            unit_area((L["ldD2"][:, lsel] / sig_hat[:, None] ** 2).mean(1), ls_hat, act), C["local"], "-", 2.0, sig_hat)
    static = Path(args.static)
    if static.exists():
        a = np.load(static)
        s = a["sigma"]
        ls_s = np.log(s)
        act_s = s >= SIG_LO
        curves["static entropy rate  MSE_S/σ²"] = (unit_area(a["mse_train_S"] / s ** 2, ls_s, act_s), C["S"], "--", 1.5, s)
        curves["static entropy rate  MSE_M/σ²"] = (unit_area(a["mse_train_M"] / s ** 2, ls_s, act_s), C["M"], "--", 1.5, s)
        curves["static surplus rate  (MSE_S − MSE_M)/σ²"] = (
            unit_area((a["mse_train_S"] - a["mse_train_M"]) / s ** 2, ls_s, act_s), C["surplus"], "-", 1.6, s)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    summary_rates = {}
    for lab, (y, col, lsty, lw, xs) in curves.items():
        ax.plot(xs, y, color=col, ls=lsty, lw=lw, label=lab)
        order = np.argsort(xs)
        c = np.concatenate([[0], np.cumsum(0.5 * (y[order][1:] + y[order][:-1]) * np.diff(np.log(xs[order])))])
        c /= c[-1]
        summary_rates[lab] = dict(sigma_peak=float(xs[np.argmax(y)]),
                                  sigma_median=float(np.exp(np.interp(0.5, c, np.log(xs[order])))),
                                  sigma_q10=float(np.exp(np.interp(0.1, c, np.log(xs[order])))),
                                  sigma_q90=float(np.exp(np.interp(0.9, c, np.log(xs[order])))))
    ax.set_xscale("log")
    ax.set_xlim(SIG_LO, 80)
    ax.set_ylim(0, None)
    ax.set_xlabel("noise level σ")
    ax.set_ylabel("density per unit log σ (unit area, σ ≥ 0.08)")
    ax.set_title("Where the two models' generations separate vs the static rates", loc="left")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig_traj_vs_rate.png", dpi=160)
    plt.close(fig)

    # ---------------- tables and summary --------------------------------------------------------
    cols = dict(step=np.arange(len(sig_hat)), sigma_hat=sig_hat, sigma_after=sig_after, t=t, gamma=T["gamma"])
    for name, arr in (("state_gap", gap), ("rel_gap", rel), ("dD2", T["dD2"]), ("tokdis", T["tokdis"]),
                      ("HS", T["HS"]), ("HM", T["HM"]), ("rate_traj", rate_traj)):
        cols[f"{name}_mean"], cols[f"{name}_sd"] = arr.mean(1), arr.std(1, ddof=1)
        cols[f"{name}_se"] = cols[f"{name}_sd"] / np.sqrt(arr.shape[1])
    if L is not None:
        for name, arr in (("ldD2", L["ldD2"][:, lsel]), ("ltokdis", L["ltokdis"][:, lsel])):
            cols[f"{name}_mean"], cols[f"{name}_sd"] = arr.mean(1), arr.std(1, ddof=1)
            cols[f"{name}_se"] = cols[f"{name}_sd"] / np.sqrt(arr.shape[1])
    with open(out / "traj_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(cols))
        for i in range(len(sig_hat)):
            w.writerow([f"{float(cols[k][i]):.9g}" for k in cols])
    tok = T["tokdis"].mean(1)
    summ = dict(
        n_examples=N, n_local=(len(common) if L is not None else 0), scores=scores,
        final_token_disagreement=float(T["final_tokdis"].mean()),
        final_state_gap=float(gap[-1].mean()),
        t_when_token_disagreement_reaches={str(q): first_cross(t, tok, q * tok[-1]) for q in (0.1, 0.5, 0.9)},
        sigma_when_token_disagreement_reaches={str(q): first_cross(sig_hat, tok, q * tok[-1]) for q in (0.1, 0.5, 0.9)},
        t_when_relative_gap_reaches={str(v): first_cross(t, rel.mean(1), v) for v in (0.1, 0.3, 1.0)},
        rates=summary_rates,
    )
    if L is not None:
        lt = L["ltokdis"][:, lsel].mean(1)
        summ["same_input_token_disagreement_peak"] = float(lt.max())
        summ["t_at_same_input_peak"] = float(t[np.argmax(lt)])
    (out / "traj_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

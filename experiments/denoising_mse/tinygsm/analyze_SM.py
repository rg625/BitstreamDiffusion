#!/usr/bin/env python
"""
CoBit-S vs CoBit-M analysis for TinyGSM (the TinyGSM counterpart of ../analyze*.py). Each part runs only if its
inputs exist, so the script can be re-run as the experiment chain progresses.

Inputs (runs/denoising_mse/tinygsm/)
  static/{S,M}       mse_tinygsm.py           static denoising error, answer region, prompt clean
  traj_paired_SM     traj_tinygsm.py paired   shared sampler + randomness, GSM8K prompts, per-problem accuracy
  traj_local_S/_M    traj_tinygsm.py local    same-input disagreement along S's / M's trajectories
  traj_true_SM       traj_tinygsm.py truechain  zero / one-step / carry self-conditioning on true chains
  gsm8k/*.json       gsm8k_eval.py            GSM8K accuracy on each model's own schedule
Outputs experiments/denoising_mse/results_tinygsm/SM/: fig_static.png, fig_token.png, fig_schedule.png,
  fig_traj.png, fig_same_input.png, fig_truechain.png, fig_gray_curve.png, fig_rates_all.png, summary.json, *.csv

Conventions as for OWT: training objective = 0.5 (no-SC + SC); diff = S - M (positive: CoBit-M better), paired
per sequence; SE across sequences. "Loss region" = answer + EOS + pad (the training loss), "answer" = answer + EOS.
Diffusion time t = F_S(sigma), CoBit-S's TinyGSM schedule CDF, as the common axis. Rates are densities over
log sigma with unit area on sigma >= 0.1 (excludes the tiny-noise error spikes, see the CoBit-S report).
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
R = REPO / "runs/denoising_mse/tinygsm"
OUT = HERE.parent / "results_tinygsm" / "SM"
RUNS = {"S": REPO / "runs/tasks/tinygsm/cobit_raw_binary_bits", "M": REPO / "assets/cobit_m_bits_s42"}
COL = {"S": "#2a78d6", "M": "#eb6834", "d": "#3d3d3a", "loc": "#4a3aa7", "true": "#1baf7a", "gr": "#9a9994"}
LABEL = {"S": "CoBit-S (130M)", "M": "CoBit-M (462M)"}
SIG_LO = 0.1


def style(ax):
    ax.grid(True, color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8984")
    ax.tick_params(colors="#52514e", labelsize=8)


def table(m):
    rd = lambda n: torch.load(RUNS[m] / n, map_location="cpu").numpy().astype(float)  # noqa: E731
    pdf, cdf, mids, edges = rd("entropy_pdf.pt"), rd("entropy_cdf.pt"), rd("entropy_sigmas.pt"), rd("entropy_edges.pt")
    cdf[-1] = 1.0
    return dict(cdf=cdf, mids=mids, dens=pdf / np.diff(np.log(edges)))


TAB = {m: table(m) for m in "SM"}


def t_of(sig, m="S"):
    return np.interp(np.log(sig), np.log(TAB[m]["mids"]), TAB[m]["cdf"], left=0, right=1)


def dens_of(sig, m):
    return np.interp(np.log(sig), np.log(TAB[m]["mids"]), TAB[m]["dens"], left=0, right=0)


def unit(y, ls, act):
    y = np.where(act, np.clip(y, 0, None), 0.0)
    o = np.argsort(ls)
    return y / np.trapz(y[o], ls[o])


def qstats(y, xs):
    o = np.argsort(xs)
    c = np.concatenate([[0], np.cumsum(0.5 * (y[o][1:] + y[o][:-1]) * np.diff(np.log(xs[o])))])
    c /= c[-1]
    q = lambda p: float(np.exp(np.interp(p, c, np.log(xs[o]))))  # noqa: E731
    return dict(sigma_peak=float(xs[np.argmax(y)]), sigma_median=q(0.5), sigma_q10=q(0.1), sigma_q90=q(0.9))


def mse_(a):
    return a.mean(1), a.std(1, ddof=1) / np.sqrt(a.shape[1])


def pband(ax, x, a, color, label, lw=1.7, ls="-", fill=True):
    if fill:
        q = np.percentile(a, [2.5, 16, 84, 97.5], axis=1)
        ax.fill_between(x, q[0], q[3], color=color, alpha=0.08, lw=0)
        ax.fill_between(x, q[1], q[2], color=color, alpha=0.2, lw=0)
    ax.plot(x, a.mean(1), color=color, lw=lw, ls=ls, label=label)


def load_chunks(d):
    fs = sorted(glob.glob(str(d / "chunk*.npz")))
    if not fs:
        return None
    zs = [np.load(f) for f in fs]
    out = {k: np.concatenate([z[k] for z in zs], 1) for k in zs[0].files if zs[0][k].ndim == 2}
    for k in zs[0].files:
        if zs[0][k].ndim == 1 and k not in ("examples", "idx", "correct_a", "correct_b", "final_tokdis"):
            out[k] = zs[0][k]
    for k in ("examples", "idx", "correct_a", "correct_b", "final_tokdis"):
        if k in zs[0].files:
            out[k] = np.concatenate([z[k] for z in zs])
    return out


def write_csv(path, cols):
    keys = list(cols)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(len(cols[keys[0]])):
            w.writerow([f"{float(cols[k][i]):.9g}" for k in keys])


def slope(y, x, k=3):
    n = len(y)
    return np.array([(y[min(n - 1, i + k)] - y[max(0, i - k)]) / (x[min(n - 1, i + k)] - x[max(0, i - k)]) for i in range(n)])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white"})
    summ, rates = {}, {}

    # ----------------------------------------------------------------- GSM8K on own schedules
    g = {}
    for f in sorted(glob.glob(str(R / "gsm8k/*.json"))):
        d = json.loads(Path(f).read_text())
        if d["n"] == 1319 and d["rope"] == "fp32":
            g[f"{d['model']}_g{d['gamma']}_s{d['steps']}"] = {k: d[k] for k in ("accuracy", "num_correct", "ci95")}
    summ["gsm8k_own_schedule"] = g

    # ----------------------------------------------------------------- static S vs M
    st = {m: load_chunks(R / f"static/{m}") for m in "SM"}
    if st["S"] is not None and st["M"] is not None:
        common = np.intersect1d(st["S"]["idx"], st["M"]["idx"])
        for m in "SM":
            sel = np.searchsorted(st[m]["idx"], common)
            n_seq = len(st[m]["idx"])
            for k in list(st[m]):  # per-sequence arrays only ([G, N]); pos_* sums have a different shape
                if isinstance(st[m][k], np.ndarray) and st[m][k].ndim == 2 and st[m][k].shape[1] == n_seq \
                        and not k.startswith("pos_"):
                    st[m][k] = st[m][k][:, sel]
            for pre in ("mse", "mseC", "berC", "terC", "tnllC"):
                st[m][f"{pre}_train"] = 0.5 * (st[m][f"{pre}_nosc"] + st[m][f"{pre}_sc"])
        s = st["S"]["sigmas"]
        ls = np.log(s)
        N = len(common)
        t = t_of(s)
        cols = dict(sigma=s, t_S=t, t_M=t_of(s, "M"))
        for pre in ("mse", "mseC", "terC", "tnllC"):
            for m in "SM":
                cols[f"{pre}_{m}"], cols[f"{pre}_{m}_se"] = mse_(st[m][f"{pre}_train"])
            cols[f"{pre}_diff"], cols[f"{pre}_diff_se"] = mse_(st["S"][f"{pre}_train"] - st["M"][f"{pre}_train"])
        write_csv(OUT / "static_curves.csv", cols)
        act = s >= SIG_LO
        dr = (st["S"]["mse_train"] - st["M"]["mse_train"]) / s[:, None] ** 2
        surplus = unit(dr.mean(1), ls, act)
        rates["static surplus rate (MSE_S − MSE_M)/σ²"] = (surplus, s, COL["gr"], "-", 1.6)
        rates["static entropy rate MSE_S/σ²"] = (unit(cols["mse_S"] / s ** 2, ls, act), s, COL["S"], "--", 1.2)
        rates["static entropy rate MSE_M/σ²"] = (unit(cols["mse_M"] / s ** 2, ls, act), s, COL["M"], "--", 1.2)
        i_pk = int(np.argmax(np.where(act, cols["mse_diff"], -np.inf)))
        sched = {}
        for m in "SM":
            dn = unit(dens_of(s, m), ls, act)
            tm = t_of(s, m)
            pos = np.clip(dr.mean(1), 0, None)
            cs = np.concatenate([[0], np.cumsum(0.5 * (pos[1:] + pos[:-1]) * np.diff(ls))])
            cs /= cs[-1]
            sched[m] = dict(overlap=float(np.trapz(np.minimum(dn, surplus), ls)),
                            surplus_in_first_20pct_steps=float(np.interp(0.2, tm, cs)),
                            surplus_in_central_50pct_steps=float(np.interp(0.75, tm, cs) - np.interp(0.25, tm, cs)),
                            schedule=qstats(dn, s))
        summ["static"] = dict(
            n=int(N),
            peak_diff_loss_region=dict(value=float(cols["mse_diff"][i_pk]), se=float(cols["mse_diff_se"][i_pk]),
                                       sigma=float(s[i_pk]), t=float(t[i_pk])),
            peak_diff_answer=dict(value=float(np.nanmax(np.where(act, cols["mseC_diff"], np.nan))),
                                  sigma=float(s[int(np.nanargmax(np.where(act, cols["mseC_diff"], np.nan)))])),
            min_diff_loss_region=dict(value=float(np.nanmin(np.where(act, cols["mse_diff"], np.nan))),
                                      sigma=float(s[int(np.nanargmin(np.where(act, cols["mse_diff"], np.nan)))])),
            sigma_M_worse=[float(x) for x in s[act & (cols["mse_diff"] / np.maximum(cols["mse_diff_se"], 1e-30) < -3)]],
            terC_diff_peak=dict(value=float(np.max(cols["terC_diff"])), t=float(t[np.argmax(cols["terC_diff"])]),
                                S=float(cols["terC_S"][np.argmax(cols["terC_diff"])]), M=float(cols["terC_M"][np.argmax(cols["terC_diff"])])),
            tnll_diff_peak_bits=dict(value=float(np.max(cols["tnllC_diff"]) / np.log(2)), t=float(t[np.argmax(cols["tnllC_diff"])])),
            tnll_diff_min_bits=dict(value=float(np.min(np.where(act, cols["tnllC_diff"], np.inf)) / np.log(2)),
                                    t=float(t[int(np.argmin(np.where(act, cols["tnllC_diff"], np.inf)))])),
            int_rate_answer_bits_per_token={m: float(np.trapz((cols[f"mseC_{m}"] / s ** 2)[act], ls[act]) * 16 / np.log(2))
                                            for m in "SM"},
            surplus_rate=qstats(surplus, s), schedules=sched,
            rate_M_rule_vs_table_TV=float(0.5 * np.trapz(np.abs(unit(s ** 3 / (s ** 3 + 1e-3) * np.sqrt(cols["mse_M"] / s ** 2), ls, s >= 0.002)
                                                              - unit(dens_of(s, "M"), ls, s >= 0.002)), ls)),
        )
        fig, axs = plt.subplots(1, 3, figsize=(14.5, 3.8))
        for m in "SM":
            axs[0].plot(s, cols[f"mseC_{m}"], color=COL[m], lw=1.8, label=LABEL[m])
        axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_ylim(1e-7, 0.2)
        axs[0].set_ylabel("answer-token per-bit MSE"); axs[0].legend(frameon=False, fontsize=8)
        for ax, k, lab in ((axs[1], "mseC", "ΔMSE (answer) = S − M"), (axs[2], "mse", "ΔMSE (loss region) = S − M")):
            mu, se = cols[f"{k}_diff"], cols[f"{k}_diff_se"]
            ax.axhline(0, color="#8a8984", lw=0.8)
            ax.fill_between(s, mu - 2 * se, mu + 2 * se, color=COL["d"], alpha=0.25, lw=0)
            ax.plot(s, mu, color=COL["d"], lw=1.8)
            ax.set_xscale("log"); ax.set_ylabel(lab)
        for ax in axs:
            ax.set_xlabel("noise level σ"); style(ax)
        fig.suptitle(f"Static denoising error on TinyGSM validation (prompt clean), N = {N}, ±2 SE", x=0.01, ha="left",
                     fontsize=10)
        fig.tight_layout(); fig.savefig(OUT / "fig_static.png", dpi=160); plt.close(fig)
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
        for ax, k, sc, lab in ((axs[0], "terC", 100, "Δ answer token error [pp]"), (axs[1], "tnllC", 1 / np.log(2), "Δ answer NLL [bits/token]")):
            mu, se = cols[f"{k}_diff"] * sc, cols[f"{k}_diff_se"] * sc
            ax.axhline(0, color="#8a8984", lw=0.8)
            ax.fill_between(t, mu - 2 * se, mu + 2 * se, color=COL["d"], alpha=0.25, lw=0)
            ax.plot(t, mu, color=COL["d"], lw=1.8)
            ax.set_xlim(1, 0); ax.set_xlabel("diffusion time t (CoBit-S schedule; sampling runs →)"); ax.set_ylabel(lab); style(ax)
        fig.tight_layout(); fig.savefig(OUT / "fig_token.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8.5, 4))
        ax.plot(s, dens_of(s, "S") / np.trapz(dens_of(s, "S")[act], ls[act]), color=COL["S"], lw=1.6, label="schedule, CoBit-S")
        ax.plot(s, dens_of(s, "M") / np.trapz(dens_of(s, "M")[act], ls[act]), color=COL["M"], lw=1.6, label="schedule, CoBit-M")
        ax.plot(s, surplus, color=COL["d"], lw=2.2, label="surplus (MSE_S − MSE_M)/σ²")
        ax.set_xscale("log"); ax.set_xlim(SIG_LO, 80); ax.set_ylim(0, None)
        ax.set_xlabel("noise level σ"); ax.set_ylabel("density per unit log σ (unit area, σ ≥ 0.1)")
        ax.legend(frameon=False, fontsize=8); style(ax)
        fig.tight_layout(); fig.savefig(OUT / "fig_schedule.png", dpi=160); plt.close(fig)

    # ----------------------------------------------------------------- paired trajectories
    P = load_chunks(R / "traj_paired_SM")
    if P is not None:
        sh = P["sigma_state"].astype(float)
        tt = t_of(sh)
        nP = P["dx2"].shape[1]
        sa = np.r_[P["sigmas"][1:], P["sigmas"][-1]].astype(float)
        ca, cb = P["correct_a"].astype(float), P["correct_b"].astype(float)
        rng = np.random.default_rng(0)
        bt = rng.integers(0, len(ca), (10000, len(ca)))
        dd = (cb[bt] - ca[bt]).mean(1)
        tok = P["tokdis"].mean(1)
        fc = lambda y, lev: float(tt[np.argmax(y >= lev)]) if (y >= lev).any() else None  # noqa: E731
        summ["paired"] = dict(
            n=int(nP), acc_S=float(ca.mean()), acc_M=float(cb.mean()), diff_M_minus_S=float((cb - ca).mean()),
            diff_ci95=[float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
            only_M=int(((cb == 1) & (ca == 0)).sum()), only_S=int(((ca == 1) & (cb == 0)).sum()), both=int(((ca == 1) & (cb == 1)).sum()),
            final_token_disagreement=float(P["final_tokdis"].mean()),
            # answer tokens already predicted differently at the first step (pure noise, prompt only)
            tokdis_first_step=float(tok[0]), tokdis_last_step=float(tok[-1]),
            # onset from the denoiser gap, which starts near 0: t at 10 / 50 / 90 % of its rise
            t_dD2_10_50_90=[fc((P["dD2"].mean(1) - P["dD2"].mean(1).min()) / (P["dD2"].mean(1)[-1] - P["dD2"].mean(1).min()), q)
                            for q in (0.1, 0.5, 0.9)],
            sigma_dD2_10_50_90=[float(sh[np.argmax((P["dD2"].mean(1) - P["dD2"].mean(1).min())
                                                   / (P["dD2"].mean(1)[-1] - P["dD2"].mean(1).min()) >= q)]) for q in (0.1, 0.5, 0.9)],
            t_rel_gap_1=fc((np.sqrt(P["dx2"]) / sa[:, None]).mean(1), 1.0))
        steps = slice(0, len(sh) - 1)
        div = slope(tok[steps], -np.log(sh[steps]))
        rates["token divergence, paired generation"] = (unit(np.r_[div, 0.0], np.log(sh), sh >= SIG_LO), sh, COL["d"], "-", 2.2)
        divD = slope(P["dD2"].mean(1)[steps], -np.log(sh[steps]))
        rates["denoiser-gap growth, paired generation"] = (unit(np.r_[divD, 0.0], np.log(sh), sh >= SIG_LO), sh, COL["d"], ":", 1.8)
        fig, axs = plt.subplots(1, 4, figsize=(17, 3.7))
        pband(axs[0], tt, np.sqrt(P["dx2"]) / sa[:, None], COL["d"], "S vs M")
        axs[0].set_yscale("log"); axs[0].set_ylim(1e-4, 10); axs[0].axhline(1, color="#8a8984", lw=0.8, ls="--")
        axs[0].set_ylabel("state gap / σ (answer region)")
        pband(axs[1], tt, P["dD2"], COL["d"], "own trajectories")
        axs[1].set_ylabel("denoiser gap mean (D_S − D_M)²")
        pband(axs[2], tt, P["tokdis"], COL["d"], "own trajectories")
        axs[2].set_ylim(0, 1); axs[2].set_ylabel("answer tokens predicted differently")
        pband(axs[3], tt, P["HA"], COL["S"], LABEL["S"]); pband(axs[3], tt, P["HB"], COL["M"], LABEL["M"])
        axs[3].set_ylabel("mean binary entropy of D [bits]"); axs[3].legend(frameon=False, fontsize=8)
        for ax in axs:
            ax.set_xlim(1, 0); ax.set_xlabel("diffusion time t (sampling runs →)"); style(ax)
        fig.suptitle(f"Paired generation on GSM8K prompts (γ = 0.31, 256 NFE, CoBit-S schedule), N = {nP}; GSM8K accuracy "
                     f"S {100 * ca.mean():.1f}%, M {100 * cb.mean():.1f}%. Bands: central 68% / 95%.", x=0.01, ha="left", fontsize=10)
        fig.tight_layout(); fig.savefig(OUT / "fig_traj.png", dpi=160); plt.close(fig)

    # ----------------------------------------------------------------- same-input decompositions
    L = {k: load_chunks(R / f"traj_local_{k}") for k in "SM"}
    if P is not None and all(v is not None for v in L.values()):
        fig, axs = plt.subplots(1, 2, figsize=(12, 3.9))
        pband(axs[0], tt, P["dD2"][:, :L["S"]["ldD2"].shape[1]], COL["d"], "own trajectories (paired)")
        for k, c in (("S", COL["S"]), ("M", COL["M"])):
            pband(axs[0], tt, L[k]["ldD2"], c, f"same input, along {LABEL[k]} trajectories")
            pband(axs[1], tt, L[k]["ltokdis"], c, f"same input, along {LABEL[k]} trajectories")
            rates[f"same-input rate, {k} trajectories"] = (unit(L[k]["ldD2"].mean(1) / sh ** 2, np.log(sh), sh >= SIG_LO), sh, c, "-", 1.8)
        pband(axs[1], tt, P["tokdis"][:, :L["S"]["ltokdis"].shape[1]], COL["d"], "own trajectories (paired)")
        axs[0].set_yscale("log"); axs[0].set_ylim(1e-6, 1); axs[0].set_ylabel("denoiser gap mean (D_S − D_M)²")
        axs[1].set_ylim(0, 1); axs[1].set_ylabel("answer tokens predicted differently"); axs[1].legend(frameon=False, fontsize=8)
        for ax in axs:
            ax.set_xlim(1, 0); ax.set_xlabel("diffusion time t (sampling runs →)"); style(ax)
        fig.tight_layout(); fig.savefig(OUT / "fig_same_input.png", dpi=160); plt.close(fig)
        own = P["dD2"].mean(1)
        summ["same_input"] = {k: dict(tokdis_peak=float(L[k]["ltokdis"].mean(1).max()), t_at_peak=float(tt[np.argmax(L[k]["ltokdis"].mean(1))]),
                                      check_max_abs=float(np.load(sorted(glob.glob(str(R / f'traj_local_{k}/chunk*.npz')))[0]).get('check_max_abs', np.nan))
                                      if 'check_max_abs' in np.load(sorted(glob.glob(str(R / f'traj_local_{k}/chunk*.npz')))[0]).files else None)
                              for k in "SM"}
        loc = L["S"]["ldD2"].mean(1)
        fx = lambda k: float(tt[np.argmax((own > k * loc) & (tt < 0.99))])  # noqa: E731
        summ["same_input"]["own_over_same_input_2x_10x_50x_t"] = [fx(2), fx(10), fx(50)]

    # ----------------------------------------------------------------- true chains
    T = load_chunks(R / "traj_true_SM")
    if T is not None and P is not None:
        nT = T["mse_S_zeros"].shape[1]
        fig, axs = plt.subplots(1, 2, figsize=(12, 3.9))
        for m in "SM":
            for v, lst in (("zeros", "--"), ("onestep", "-."), ("carry", "-")):
                axs[0].plot(tt, T[f"mse_{m}_{v}"].mean(1), color=COL[m], ls=lst, lw=1.6, label=f"{m}, {v} SC")
            for v, lst in (("onestep", "-."), ("carry", "-")):
                axs[1].plot(tt, T[f"mse_{m}_{v}"].mean(1) - T[f"mse_{m}_zeros"].mean(1), color=COL[m], ls=lst, lw=1.6, label=f"{m}, {v} − zero")
        axs[1].axhline(0, color="#8a8984", lw=0.8)
        axs[0].set_ylabel("MSE vs true x₀ (answer tokens)"); axs[1].set_ylabel("MSE(with SC) − MSE(zero SC)")
        for ax in axs:
            ax.set_xlim(1, 0); ax.set_xlabel("diffusion time t (sampling runs →)"); ax.legend(frameon=False, fontsize=7.5); style(ax)
        fig.tight_layout(); fig.savefig(OUT / "fig_truechain.png", dpi=160); plt.close(fig)
        gray = {}
        fig, axs = plt.subplots(1, 2, figsize=(12.5, 4))
        vs = [("zeros", COL["gr"], "--"), ("onestep", COL["loc"], "-"), ("carry", COL["true"], "-")]
        if summ.get("static"):
            sr = (st["S"]["mse_train"] - st["M"]["mse_train"]).mean(1)
            # static surplus over the ANSWER region, comparable with the chains (answer tokens)
            srA = (st["S"]["mseC_train"] - st["M"]["mseC_train"]).mean(1)
            y = srA / s ** 2
            axs[0].plot(s[act], y[act], color="#1b1b1a", lw=1.4, label="static (training mix), answer region")
            axs[1].plot(s, unit(y, ls, act), color="#1b1b1a", lw=1.4, label="static (training mix)")
            gray["static_answer"] = dict(area=float(np.trapz(y[act], ls[act])), **qstats(unit(y, ls, act), s))
        for v, c, lst in vs:
            y = (T[f"mse_S_{v}"] - T[f"mse_M_{v}"]).mean(1) / sh ** 2
            a_ = sh >= SIG_LO
            o = np.argsort(sh[a_])
            axs[0].plot(sh[a_], y[a_], color=c, ls=lst, lw=1.8, label=f"true chains, {v} SC")
            yy = unit(y, np.log(sh), a_)
            axs[1].plot(sh, yy, color=c, ls=lst, lw=1.8, label=f"true chains, {v} SC")
            gray[v] = dict(area=float(np.trapz(y[a_][o], np.log(sh[a_][o]))), **qstats(yy, sh))
            if v == "carry":
                rates["same-input rate, true chains (carry SC)"] = (unit(T["dD2_carry"].mean(1) / sh ** 2, np.log(sh), a_), sh, c, "-", 1.8)
        for ax in axs:
            ax.set_xscale("log"); ax.set_xlim(SIG_LO, 80); ax.set_xlabel("noise level σ"); style(ax)
        axs[0].axhline(0, color="#8a8984", lw=0.8)
        axs[0].set_ylabel("surplus rate (MSE_S − MSE_M)/σ² (answer region)"); axs[1].set_ylabel("unit area, σ ≥ 0.1")
        axs[0].legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / "fig_gray_curve.png", dpi=160); plt.close(fig)
        cz = {m: T[f"mse_{m}_carry"].mean(1) - T[f"mse_{m}_zeros"].mean(1) for m in "SM"}
        oz = {m: T[f"mse_{m}_onestep"].mean(1) - T[f"mse_{m}_zeros"].mean(1) for m in "SM"}
        summ["truechain"] = dict(n=int(nT), gray=gray,
                                 carry_minus_zero={m: dict(max=float(cz[m].max()), t=float(tt[np.argmax(cz[m])])) for m in "SM"},
                                 onestep_minus_zero={m: dict(max=float(oz[m].max()), min=float(oz[m].min())) for m in "SM"},
                                 tokdis_peak={v: float(T[f"tokdis_{v}"].mean(1).max()) for v in ("zeros", "onestep", "carry")})

    # ----------------------------------------------------------------- all rates
    if rates:
        fig, ax = plt.subplots(figsize=(9, 4.4))
        summ["rates"] = {}
        for lab, (y, xs, c, lst, lw) in rates.items():
            ax.plot(xs, y, color=c, ls=lst, lw=lw, label=lab)
            summ["rates"][lab] = qstats(y, xs)
        ax.set_xscale("log"); ax.set_xlim(SIG_LO, 80); ax.set_ylim(0, None)
        ax.set_xlabel("noise level σ"); ax.set_ylabel("density per unit log σ (unit area, σ ≥ 0.1)")
        ax.legend(frameon=False, fontsize=7.5); style(ax)
        fig.tight_layout(); fig.savefig(OUT / "fig_rates_all.png", dpi=160); plt.close(fig)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1)[:6000])


if __name__ == "__main__":
    main()

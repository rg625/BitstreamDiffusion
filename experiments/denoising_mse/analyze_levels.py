#!/usr/bin/env python
"""
Sequence-level, token-level and "global" views of the S-vs-M denoising error, and the overlay of the
entropy-rate schedules on the parameter surplus (experiments/denoising_mse/).

Inputs
  runs/denoising_mse/<main>/{S,M}/chunk*.npz  per-sequence bit MSE, 4096 seqs x 129 sigmas
  runs/denoising_mse/<tok>/{S,M}/chunk*.npz   + token metrics (mse_curves.py run_chunk docstring),
                                              2048 seqs x 97 sigmas
Outputs (experiments/denoising_mse/results/levels/)
  fig_sequence_level.png   per-sequence MSE trajectories + quantile bands; distribution of the
                           per-sequence paired difference; fraction of sequences where M is better
  fig_token_level.png      token error rate, token NLL, bit error rate (content tokens) and S - M
  fig_global_time.png      per-sequence totals vs diffusion time: wrong tokens, NLL bits
  fig_token_position.png   token error rate vs position in the sequence at fixed t
  fig_schedule_overlay.png entropy-rate schedules vs the parameter surplus
  *.csv, levels.npz, schedule_match.json

Conventions as in analyze.py: variant "train" = 0.5 (no-SC + SC) (the training objective); diff = S - M
(positive: CoBit-M better); SE across sequences, paired. Diffusion time t = F_S(sigma), the CoBit-S
entropy-rate CDF, is used as the common time axis for both models (so differences at fixed t are
model differences at a fixed noise level).

Schedule vs surplus. The trainer builds the schedule as (utils/schedule_controller.py)
    pdf(log-sigma bin) ∝ g(sigma) * rate(sigma)^0.5,  rate = MSE/sigma^2,  g = s^3/(s^3 + 0.1^3)
where rate is the I-MMSE information rate per unit log sigma (its integral over log sigma is the
denoising bound on the NLL, up to a constant factor). The parameter surplus in the same units is
    Δrate(sigma) = (MSE_S - MSE_M) / sigma^2,
i.e. the density over log sigma of the NLL-bound reduction bought by the extra parameters. We compare
the schedule density with Δrate restricted to sigma >= 0.08 (the sampler's terminal sigma; both
schedules put zero mass below it).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COL = {"S": "#2a78d6", "M": "#eb6834", "diff": "#3d3d3a", "spag": "#9a9994"}
LABEL = {"S": "CoBit-S (130M)", "M": "CoBit-M (462M)"}

# Task switch (default OWT): DMSE_TASK=tinygsm runs the identical analysis on the TinyGSM runs.
import os as _os
TASK = _os.environ.get("DMSE_TASK", "owt").lower()
TASKCFG = {
    "owt": dict(tag_prefix="", results="results", tables={"S": "assets/entropy_tables/owt", "M": "assets/entropy_tables/owt_medium"},
                sig_lo=0.08, name="OWT"),
    "tinygsm": dict(tag_prefix="tinygsm/", results="results_tinygsm",
                    tables={"S": "runs/tasks/tinygsm/cobit_raw_binary_bits", "M": "assets/cobit_m_bits_s42"},
                    sig_lo=0.1, name="TinyGSM"),
}[TASK]
SIGMA_TERMINAL = TASKCFG["sig_lo"]
LN2 = np.log(2.0)


# ---------------------------------------------------------------------------- loading
def load_run(tag: str):
    """Paired per-sequence arrays for both models: dict key -> {"S": [G,N], "M": [G,N]}."""
    tag = TASKCFG["tag_prefix"] + tag
    out, idx, sig, meta = {}, {}, None, {}
    raw = {}
    for m in ("S", "M"):
        fs = sorted(glob.glob(str(REPO / f"runs/denoising_mse/{tag}/{m}/chunk*.npz")))
        if not fs:
            raise SystemExit(f"no chunks for {tag}/{m}")
        zs = [np.load(f) for f in fs]
        idx[m] = np.concatenate([z["idx"] for z in zs])
        sig = zs[0]["sigmas"]
        keys = [k for k in zs[0].files if k.split("_")[0] in ("mse", "mseC", "berC", "terC", "tnllC")]
        raw[m] = {k: np.concatenate([z[k] for z in zs], 1) for k in keys}
        for k in [k for k in zs[0].files if k.startswith("tokpos_") and k != "tokpos_count"]:
            raw[m][k] = sum(z[k] for z in zs)
        if "tokpos_count" in zs[0].files:
            raw[m]["tokpos_count"] = sum(z["tokpos_count"] for z in zs)
        for k in [k for k in zs[0].files if k.startswith("tokposR")]:
            raw[m][k] = sum(z[k] for z in zs)
        meta[m] = json.loads((REPO / f"runs/denoising_mse/{tag}/{m}/meta.json").read_text())
    common = np.intersect1d(idx["S"], idx["M"])
    for m in ("S", "M"):
        sel = np.searchsorted(idx[m], common)
        assert np.array_equal(idx[m][sel], common)
        for k, v in raw[m].items():
            if k.startswith("tokpos"):
                continue
            out.setdefault(k, {})[m] = v[:, sel]
    # training-objective mix of the two trained self-conditioning inputs
    for pre in {k.rsplit("_", 1)[0] for k in out}:
        if f"{pre}_nosc" in out:
            out[f"{pre}_train"] = {m: 0.5 * (out[f"{pre}_nosc"][m] + out[f"{pre}_sc"][m]) for m in ("S", "M")}
    tokpos = {m: {k: raw[m][k] for k in raw[m] if k.startswith("tokpos")} for m in ("S", "M")}
    return dict(sig=sig, idx=common, d=out, tokpos=tokpos, meta=meta)


def n_content_tokens(idx, cfg_root="datasets/openwebtext_gpt2_trainm100k"):
    if TASK == "tinygsm":  # answer + EOS tokens of each validation sequence
        sys.path.insert(0, str(HERE / "tinygsm"))
        import common as TC

        return np.asarray(TC.TinyGSMVal().alen[np.asarray(idx)], float)
    root = REPO / cfg_root
    name = "cache_val_gpt2_gpt2id_bpe16_v65536_base1024_len1024"
    meta = json.loads((root / f"{name}.meta.json").read_text())
    mm = np.memmap(root / f"{name}.uint16", dtype=np.uint16, mode="r",
                   shape=(meta["num_sequences"], meta["code_sequence_len_tokens"]))
    return np.array([(np.asarray(mm[int(i)]) != meta["pad_id"]).sum() for i in idx], float)


def load_table(table_dir):
    import torch

    rd = lambda n: torch.load(REPO / table_dir / n, map_location="cpu").numpy().astype(float)  # noqa: E731
    pdf, cdf, mids, edges = rd("entropy_pdf.pt"), rd("entropy_cdf.pt"), rd("entropy_sigmas.pt"), rd("entropy_edges.pt")
    cdf[-1] = 1.0
    dens = pdf / np.diff(np.log(edges))  # probability per unit log sigma
    return dict(pdf=pdf, cdf=cdf, mids=mids, edges=edges, dens=dens)


def t_of(sig, tab):
    return np.interp(np.log(sig), np.log(tab["mids"]), tab["cdf"], left=0.0, right=1.0)


def mean_se(a):
    return a.mean(1), a.std(1, ddof=1) / np.sqrt(a.shape[1])


def to_time(per_seq, ls_src, ls_t):
    """Interpolate per-sequence curves [G, N] (in log sigma) onto log sigma(t) [T] -> [T, N]."""
    return np.stack([np.interp(ls_t, ls_src, per_seq[:, n]) for n in range(per_seq.shape[1])], 1)


# ---------------------------------------------------------------------------- plotting helpers
def _setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelcolor": "#0b0b0b",
                         "axes.titlecolor": "#0b0b0b", "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
    return plt


def _style(ax):
    ax.grid(True, color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8984")
    ax.tick_params(colors="#52514e", labelsize=8)


def _diff_axis(ax, x, mu, se, ylabel):
    ax.axhline(0, color="#8a8984", lw=0.8)
    ax.fill_between(x, mu - 2 * se, mu + 2 * se, color=COL["diff"], alpha=0.22, lw=0, label="±2 SE")
    ax.plot(x, mu, color=COL["diff"], lw=1.8, label="mean S − M")
    ax.set_ylabel(ylabel)
    _style(ax)


def _pair_axis(ax, x, curves, ylabel, logy=False):
    for m in ("S", "M"):
        mu, se = curves[m]
        ax.plot(x, mu, color=COL[m], lw=1.8, label=LABEL[m])
        ax.fill_between(x, mu - 2 * se, mu + 2 * se, color=COL[m], alpha=0.2, lw=0)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    _style(ax)


def write_csv(path, cols):
    keys = list(cols)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(len(cols[keys[0]])):
            w.writerow([f"{float(cols[k][i]):.9g}" for k in keys])


# ---------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-tag", default="main")
    ap.add_argument("--tok-tag", default="tok")
    ap.add_argument("--n-spaghetti", type=int, default=24)
    args = ap.parse_args()
    out = HERE / TASKCFG["results"] / "levels"
    out.mkdir(parents=True, exist_ok=True)
    plt = _setup()

    tabS = load_table(TASKCFG["tables"]["S"])
    tabM = load_table(TASKCFG["tables"]["M"])
    tg = np.linspace(0.0, 1.0, 201)
    keep = np.r_[True, np.diff(tabS["cdf"]) > 1e-12]
    ls_t = np.interp(tg, tabS["cdf"][keep], np.log(tabS["mids"][keep]))  # log sigma at t (S schedule)
    saved = dict(t=tg, sigma_at_t=np.exp(ls_t))
    summary = {}

    # ===================================================================== sequence level (main)
    R = load_run(args.main_tag)
    sig, ls = R["sig"], np.log(R["sig"])
    N = len(R["idx"])
    seqT = {m: to_time(R["d"]["mse_train"][m], ls, ls_t) for m in ("S", "M")}   # [T, N]
    dT = seqT["S"] - seqT["M"]
    qs = (1, 10, 50, 90, 99)
    cols = dict(t=tg, sigma=np.exp(ls_t))
    for m in ("S", "M"):
        for q in qs:
            cols[f"mse_{m}_q{q}"] = np.percentile(seqT[m], q, axis=1)
        cols[f"mse_{m}_mean"] = seqT[m].mean(1)
    for q in qs:
        cols[f"diff_q{q}"] = np.percentile(dT, q, axis=1)
    cols["diff_mean"], cols["diff_se"] = mean_se(dT)
    cols["frac_seq_M_better"] = (dT > 0).mean(1)
    cols["frac_seq_M_worse"] = (dT < 0).mean(1)
    write_csv(out / "sequence_level_time.csv", cols)
    saved.update({f"seq_{k}": v for k, v in cols.items() if k not in ("t", "sigma")})

    rng = np.random.default_rng(0)
    pick = rng.choice(N, size=min(args.n_spaghetti, N), replace=False)
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.4))
    for j, m in enumerate(("S", "M")):
        ax = axs[0, j]
        for n in pick:
            ax.plot(tg, seqT[m][:, n], color=COL["spag"], lw=0.5, alpha=0.6)
        ax.fill_between(tg, cols[f"mse_{m}_q1"], cols[f"mse_{m}_q99"], color=COL[m], alpha=0.12, lw=0,
                        label="1–99%")
        ax.fill_between(tg, cols[f"mse_{m}_q10"], cols[f"mse_{m}_q90"], color=COL[m], alpha=0.25, lw=0,
                        label="10–90%")
        ax.plot(tg, cols[f"mse_{m}_q50"], color=COL[m], lw=2, label="median")
        ax.plot([], [], color=COL["spag"], lw=0.8, label=f"{len(pick)} individual sequences")
        ax.set_title(f"{LABEL[m]}: per-sequence MSE trajectories", loc="left")
        ax.set_ylabel("per-sequence mean bit MSE")
        ax.set_ylim(0, None)
        _style(ax)
        ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    ax = axs[1, 0]
    ax.axhline(0, color="#8a8984", lw=0.8)
    ax.fill_between(tg, cols["diff_q10"], cols["diff_q90"], color=COL["diff"], alpha=0.14, lw=0,
                    label="10–90% of sequences")
    ax.fill_between(tg, cols["diff_mean"] - 2 * cols["diff_se"], cols["diff_mean"] + 2 * cols["diff_se"],
                    color=COL["diff"], alpha=0.35, lw=0, label="mean ±2 SE")
    ax.plot(tg, cols["diff_q50"], color=COL["diff"], lw=1, ls="--", label="median")
    ax.plot(tg, cols["diff_mean"], color=COL["diff"], lw=1.8, label="mean")
    ax.set_title("per-sequence paired difference S − M", loc="left")
    ax.set_ylabel("ΔMSE (per sequence)")
    _style(ax)
    ax.legend(frameon=False, fontsize=7.5)
    ax = axs[1, 1]
    ax.axhline(0.5, color="#8a8984", lw=0.8)
    ax.plot(tg, cols["frac_seq_M_better"], color=COL["diff"], lw=1.8)
    ax.set_ylim(0, 1)
    ax.set_title("fraction of sequences where CoBit-M has lower MSE", loc="left")
    ax.set_ylabel("fraction of sequences")
    _style(ax)
    for ax in axs[1]:
        ax.set_xlabel("diffusion time t = F_S(σ)  (CoBit-S entropy-rate schedule)")
    for ax in axs[0]:
        ax.set_xlabel("diffusion time t")
    fig.suptitle(f"Sequence level: training-objective denoising MSE, N={N} {TASKCFG['name']} validation sequences",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_sequence_level.png", dpi=160)
    plt.close(fig)
    summary["sequence_level"] = dict(
        n_sequences=N,
        max_frac_seq_M_better=float(cols["frac_seq_M_better"].max()),
        t_at_max_frac_M_better=float(tg[np.argmax(cols["frac_seq_M_better"])]),
    )

    # ===================================================================== schedule overlay (main)
    mseS, mseM = R["d"]["mse_train"]["S"], R["d"]["mse_train"]["M"]
    act = sig >= SIGMA_TERMINAL
    rate = {m: R["d"]["mse_train"][m] / sig[:, None] ** 2 for m in ("S", "M")}  # [G, N]
    drate_seq = rate["S"] - rate["M"]
    drate, drate_se = mean_se(drate_seq)
    dmse, dmse_se = mean_se(mseS - mseM)
    la = ls[act]
    norm_r = np.trapz(np.abs(drate[act]), la)
    norm_m = np.trapz(np.abs(dmse[act]), la)
    densS = np.interp(ls, np.log(tabS["mids"]), tabS["dens"], left=0, right=0)
    densM = np.interp(ls, np.log(tabM["mids"]), tabM["dens"], left=0, right=0)
    # the schedule rule applied to our measured MSE (consistency check against the stored tables)
    g = sig ** 3 / (sig ** 3 + 0.1 ** 3)
    rec = {}
    for m in ("S", "M"):
        r = rate[m].mean(1)
        q = g * np.sqrt(np.clip(r, 0, None))
        q[~act] = 0
        rec[m] = q / np.trapz(q, ls)
    # matched-schedule measure: on a uniform-t grid the per-step gain is Δrate * dlogσ/dt; it is flat
    # iff schedule density ∝ Δrate. Cumulative version: surplus fraction vs schedule CDF.
    cum = {}
    pos = np.where(act, np.clip(drate, 0, None), 0.0)  # same sigma range as the other panels
    for m, tab in (("S", tabS), ("M", tabM)):
        tm = t_of(sig, tab)
        cs = np.concatenate([[0], np.cumsum(0.5 * (pos[1:] + pos[:-1]) * np.diff(ls))])
        cum[m] = (tm, cs / cs[-1])
    ovl = {}
    q_surplus = np.where(act, pos, 0)
    q_surplus = q_surplus / np.trapz(q_surplus, ls)
    for m, d, tab in (("S", densS, tabS), ("M", densM, tabM)):
        ovl[m] = dict(
            overlap_coefficient=float(np.trapz(np.minimum(d, q_surplus), ls)),
            total_variation=float(0.5 * np.trapz(np.abs(d - q_surplus), ls)),
            corr_density=float(np.corrcoef(d[act], q_surplus[act])[0, 1]),
            sigma_peak_schedule=float(sig[np.argmax(d)]),
            sigma_median_schedule=float(np.exp(np.interp(0.5, tab["cdf"], np.log(tab["mids"])))),
            surplus_fraction_in_schedule_5_95=float(_mass_between(q_surplus, ls, d, 0.05, 0.95)),
            surplus_fraction_in_schedule_25_75=float(_mass_between(q_surplus, ls, d, 0.25, 0.75)),
            schedule_vs_rule_on_measured_mse_TV=float(0.5 * np.trapz(np.abs(d - rec[m]), ls)),
        )
    cq = np.concatenate([[0], np.cumsum(0.5 * (q_surplus[1:] + q_surplus[:-1]) * np.diff(ls))])
    summary["schedule_vs_surplus"] = dict(
        sigma_range=[SIGMA_TERMINAL, float(sig[-1])],
        sigma_peak_surplus_rate=float(sig[np.argmax(pos)]),
        sigma_median_surplus_rate=float(np.exp(np.interp(0.5, cq, ls))),
        sigma_peak_dmse=float(sig[np.argmax(dmse)]),
        net_surplus_rate_integral=float(np.trapz(drate[act], la)),
        net_surplus_rate_integral_se=float(np.trapz(drate_seq[act], la, axis=0).std(ddof=1) / np.sqrt(N)),
        negative_part_of_surplus_rate=float(np.trapz(np.clip(drate[act], None, 0), la)),
        per_schedule=ovl,
    )
    write_csv(out / "schedule_overlay.csv", dict(
        sigma=sig, log_sigma=ls, t_S=t_of(sig, tabS), t_M=t_of(sig, tabM),
        schedule_density_S=densS, schedule_density_M=densM,
        schedule_rule_on_measured_mse_S=rec["S"], schedule_rule_on_measured_mse_M=rec["M"],
        rate_S=rate["S"].mean(1), rate_M=rate["M"].mean(1),
        surplus_rate=drate, surplus_rate_se=drate_se, surplus_rate_unit_area=np.where(act, drate / norm_r, np.nan),
        dmse=dmse, dmse_se=dmse_se, dmse_unit_area=np.where(act, dmse / norm_m, np.nan),
        cum_surplus_fraction=cq))

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1.35, 1.1, 0.9]))
    ax = axs[0]
    ax.axhline(0, color="#8a8984", lw=0.8)
    ax.plot(sig, densS, color=COL["S"], lw=1.8, label="schedule density, CoBit-S")
    ax.plot(sig, densM, color=COL["M"], lw=1.8, label="schedule density, CoBit-M")
    y, e = np.where(act, drate / norm_r, np.nan), np.where(act, drate_se / norm_r, np.nan)
    ax.fill_between(sig, y - 2 * e, y + 2 * e, color=COL["diff"], alpha=0.22, lw=0)
    ax.plot(sig, y, color=COL["diff"], lw=2, label="surplus Δrate = (MSE_S − MSE_M)/σ²")
    ax.plot(sig, np.where(act, dmse / norm_m, np.nan), color=COL["diff"], lw=1.1, ls="--",
            label="ΔMSE = MSE_S − MSE_M")
    ax.set_xscale("log")
    ax.set_xlim(SIGMA_TERMINAL * 0.8, sig[-1])
    ax.set_xlabel("noise level σ")
    ax.set_ylabel(f"density per unit log σ (each curve: unit area, σ ≥ {SIGMA_TERMINAL:g})")
    ax.set_title("Schedules vs parameter surplus", loc="left")
    ax.legend(frameon=False, fontsize=7.5)
    _style(ax)
    ax = axs[1]
    # surplus covered by each slice of sampler steps: steps are uniform in t, so integrate Δrate over
    # the log-sigma interval of each of nb equal-t bins; normalised so a matched schedule gives 1
    nb = 50
    tb = np.linspace(0, 1, nb + 1)
    fine = np.linspace(ls[act][0], ls[-1], 4000)
    dr_f = np.interp(fine, ls, drate)
    dr_seq_f = None
    for m, tab in (("S", tabS), ("M", tabM)):
        kk = np.r_[True, np.diff(tab["cdf"]) > 1e-12]
        lb = np.interp(tb, tab["cdf"][kk], np.log(tab["mids"][kk]))
        lb[0], lb[-1] = max(lb[0], fine[0]), fine[-1]
        cs = np.concatenate([[0], np.cumsum(0.5 * (dr_f[1:] + dr_f[:-1]) * np.diff(fine))])
        per_bin = np.diff(np.interp(lb, fine, cs))
        # SE per bin from per-sequence integrals
        cseq = np.concatenate([np.zeros((1, N)), np.cumsum(0.5 * (drate_seq[1:] + drate_seq[:-1])
                                                          * np.diff(ls)[:, None], 0)])
        pb_seq = np.diff(np.stack([np.interp(lb, ls, cseq[:, n]) for n in range(N)], 1), axis=0)
        se_bin = pb_seq.std(1, ddof=1) / np.sqrt(N)
        z = per_bin.sum() / nb
        xs = np.repeat(tb, 2)[1:-1]
        ax.fill_between(xs, np.repeat((per_bin - 2 * se_bin) / z, 2), np.repeat((per_bin + 2 * se_bin) / z, 2),
                        color=COL[m], alpha=0.18, lw=0)
        ax.plot(xs, np.repeat(per_bin / z, 2), color=COL[m], lw=1.6,
                label=f"{m} schedule ({nb} bins of {100 // nb}% of steps)")
        summary.setdefault("surplus_per_step_bin", {})[m] = dict(
            t_bin_edges=tb.tolist(), share_per_bin_mean1=(per_bin / z).tolist(),
            se=(se_bin / z).tolist(), max_share=float((per_bin / z).max()),
            t_of_max=float(0.5 * (tb[:-1] + tb[1:])[np.argmax(per_bin)]))
    ax.axhline(1, color="#8a8984", lw=0.9, ls="--", label="matched schedule (flat at 1)")
    ax.axhline(0, color="#8a8984", lw=0.6)
    ax.set_xlabel("diffusion time t (own schedule; sampling runs t = 1 → 0)")
    ax.set_ylabel("surplus per slice of steps (1 = matched)")
    ax.set_title("Surplus covered per slice of sampler steps", loc="left")
    ax.legend(frameon=False, fontsize=7.5)
    _style(ax)
    ax = axs[2]
    ax.plot([0, 1], [0, 1], color="#8a8984", lw=0.9, ls="--", label="perfect match")
    for m in ("S", "M"):
        ax.plot(cum[m][0], cum[m][1], color=COL[m], lw=1.8, label=f"{m} schedule")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("t = fraction of sampler steps, counted from the clean end")
    ax.set_ylabel("cumulative fraction of surplus (positive part)")
    ax.set_title("Cumulative: steps vs surplus", loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    _style(ax)
    fig.suptitle(f"Entropy-rate schedule overlaid on the S − M denoising-error surplus "
                 f"(training objective, N={N}, ±2 SE)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_schedule_overlay.png", dpi=160)
    plt.close(fig)

    # ===================================================================== token level (tok)
    try:
        T = load_run(args.tok_tag)
    except SystemExit as e:
        print(f"[levels] token run not available ({e}); skipping token-level outputs")
        T = None
    if T is not None:
        sigT, lsT = T["sig"], np.log(T["sig"])
        NT = len(T["idx"])
        ntok = n_content_tokens(T["idx"])
        mets = [("terC", "token error rate (content tokens)", 1.0),
                ("tnllC", "token NLL [bits/token]", 1 / LN2),
                ("berC", "bit error rate (content bits)", 1.0)]
        cols = dict(t=tg, sigma=np.exp(ls_t))
        curves = {}
        for key, lab, sc in mets:
            per = {m: to_time(T["d"][f"{key}_train"][m] * sc, lsT, ls_t) for m in ("S", "M")}
            for m in ("S", "M"):
                cols[f"{key}_{m}"], cols[f"{key}_{m}_se"] = mean_se(per[m])
            cols[f"{key}_diff"], cols[f"{key}_diff_se"] = mean_se(per["S"] - per["M"])
            curves[key] = per
        # global per-sequence totals
        glob_ = {}
        for key, lab, sc in (("terC", "wrong tokens per sequence", 1.0),
                             ("tnllC", "NLL per sequence [kbit]", 1 / LN2 / 1000)):
            per = {m: to_time(T["d"][f"{key}_train"][m] * ntok[None, :] * sc, lsT, ls_t) for m in ("S", "M")}
            g_ = "wrongtok_per_seq" if key == "terC" else "nll_kbit_per_seq"
            for m in ("S", "M"):
                cols[f"{g_}_{m}"], cols[f"{g_}_{m}_se"] = mean_se(per[m])
                cols[f"{g_}_{m}_q10"] = np.percentile(per[m], 10, axis=1)
                cols[f"{g_}_{m}_q90"] = np.percentile(per[m], 90, axis=1)
            cols[f"{g_}_diff"], cols[f"{g_}_diff_se"] = mean_se(per["S"] - per["M"])
            glob_[g_] = (lab, per)
        write_csv(out / "token_level_time.csv", cols)
        saved.update({f"tok_{k}": v for k, v in cols.items() if k not in ("t", "sigma")})
        # same on the sigma grid (no interpolation)
        cs = dict(sigma=sigT, t_S=t_of(sigT, tabS), t_M=t_of(sigT, tabM))
        for key, lab, sc in mets:
            for m in ("S", "M"):
                cs[f"{key}_{m}"], cs[f"{key}_{m}_se"] = mean_se(T["d"][f"{key}_train"][m] * sc)
            cs[f"{key}_diff"], cs[f"{key}_diff_se"] = mean_se((T["d"][f"{key}_train"]["S"]
                                                              - T["d"][f"{key}_train"]["M"]) * sc)
        write_csv(out / "token_level_sigma.csv", cs)

        fig, axs = plt.subplots(3, 2, figsize=(11, 9.6), sharex=True)
        for i, (key, lab, sc) in enumerate(mets):
            _pair_axis(axs[i, 0], tg, {m: (cols[f"{key}_{m}"], cols[f"{key}_{m}_se"]) for m in ("S", "M")}, lab)
            _diff_axis(axs[i, 1], tg, cols[f"{key}_diff"], cols[f"{key}_diff_se"], "S − M")
        axs[0, 0].legend(frameon=False, loc="upper left")
        axs[0, 1].legend(frameon=False, loc="upper right", fontsize=7.5)
        for ax in axs[-1]:
            ax.set_xlabel("diffusion time t = F_S(σ)")
        fig.suptitle(f"Token level (content tokens, training objective), N={NT} sequences, ±2 SE",
                     x=0.01, ha="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "fig_token_level.png", dpi=160)
        plt.close(fig)

        fig, axs = plt.subplots(2, 2, figsize=(11, 6.8), sharex=True)
        for j, g_ in enumerate(("wrongtok_per_seq", "nll_kbit_per_seq")):
            lab = glob_[g_][0]
            ax = axs[0, j]
            for m in ("S", "M"):
                ax.fill_between(tg, cols[f"{g_}_{m}_q10"], cols[f"{g_}_{m}_q90"], color=COL[m], alpha=0.15, lw=0)
                ax.plot(tg, cols[f"{g_}_{m}"], color=COL[m], lw=1.8, label=f"{LABEL[m]} mean (band: 10–90% of seqs)")
            ax.set_ylabel(lab)
            _style(ax)
            _diff_axis(axs[1, j], tg, cols[f"{g_}_diff"], cols[f"{g_}_diff_se"], f"S − M, {lab}")
            axs[1, j].set_xlabel("diffusion time t = F_S(σ)")
        axs[0, 0].legend(frameon=False, loc="upper left", fontsize=7.5)
        axs[1, 0].legend(frameon=False, fontsize=7.5)
        fig.suptitle(f"Global (whole-sequence) denoising error vs diffusion time, N={NT}, mean ±2 SE",
                     x=0.01, ha="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "fig_global_time.png", dpi=160)
        plt.close(fig)

        # token error rate vs position at fixed t
        cnt = np.minimum(T["tokpos"]["S"]["tokpos_count"], T["tokpos"]["M"]["tokpos_count"])
        ok = cnt >= 200
        tsel = (0.1, 0.3, 0.5, 0.7)
        fig, axs = plt.subplots(1, len(tsel), figsize=(14, 3.6), sharey=False)
        posout = dict(position=np.arange(len(cnt)), n_content=cnt)
        for j, tt in enumerate(tsel):
            s_t = np.exp(np.interp(tt, tabS["cdf"][keep], np.log(tabS["mids"][keep])))
            gi = int(np.argmin(np.abs(lsT - np.log(s_t))))
            ax = axs[j]
            for m in ("S", "M"):
                tp = T["tokpos"][m]
                # each model's own content-token counts (its chunks may cover different sequences)
                r = 0.5 * (tp["tokpos_nosc"][gi] + tp["tokpos_sc"][gi]) / np.maximum(tp["tokpos_count"], 1)
                posout[f"ter_{m}_t{tt}"] = r
                ax.plot(np.where(ok)[0], r[ok], color=COL[m], lw=1.0, label=LABEL[m])
            ax.set_title(f"t ≈ {t_of(np.array([sigT[gi]]), tabS)[0]:.2f}  (σ = {sigT[gi]:.3g})", loc="left")
            ax.set_xlabel("token position")
            _style(ax)
        axs[0].set_ylabel("token error rate")
        axs[0].legend(frameon=False, fontsize=7.5)
        fig.suptitle(f"Token error rate vs position in the sequence (positions with ≥200 content tokens), N={NT}",
                     x=0.01, ha="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "fig_token_position.png", dpi=160)
        plt.close(fig)
        write_csv(out / "token_position.csv", posout)
        if "tokposR_count" in T["tokpos"]["S"]:  # TinyGSM: position measured from the answer start
            cntR = np.minimum(T["tokpos"]["S"]["tokposR_count"], T["tokpos"]["M"]["tokposR_count"])
            okR = cntR >= 200
            fig, axs = plt.subplots(1, len(tsel), figsize=(14, 3.6))
            posR = dict(position_from_answer_start=np.arange(len(cntR)), n_answer_tokens=cntR)
            for j, tt in enumerate(tsel):
                s_t = np.exp(np.interp(tt, tabS["cdf"][keep], np.log(tabS["mids"][keep])))
                gi = int(np.argmin(np.abs(lsT - np.log(s_t))))
                for m in ("S", "M"):
                    tp = T["tokpos"][m]
                    r = 0.5 * (tp["tokposR_nosc"][gi] + tp["tokposR_sc"][gi]) / np.maximum(tp["tokposR_count"], 1)
                    posR[f"ter_{m}_t{tt}"] = r
                    axs[j].plot(np.where(okR)[0], r[okR], color=COL[m], lw=1.0, label=LABEL[m])
                axs[j].set_title(f"t ≈ {t_of(np.array([sigT[gi]]), tabS)[0]:.2f}  (σ = {sigT[gi]:.3g})", loc="left")
                axs[j].set_xlabel("token position from the answer start")
                _style(axs[j])
            axs[0].set_ylabel("token error rate")
            axs[0].legend(frameon=False, fontsize=7.5)
            fig.suptitle(f"Answer-token error rate vs position from the answer start (positions with ≥200 answer tokens), N={NT}",
                         x=0.01, ha="left", fontsize=10)
            fig.tight_layout()
            fig.savefig(out / "fig_token_position_rel.png", dpi=160)
            plt.close(fig)
            write_csv(out / "token_position_rel.csv", posR)
        summary["token_level"] = dict(
            n_sequences=NT, mean_content_tokens_per_seq=float(ntok.mean()),
            max_terC_diff=float(cols["terC_diff"].max()), t_at_max_terC_diff=float(tg[np.argmax(cols["terC_diff"])]),
            max_wrongtok_per_seq_diff=float(cols["wrongtok_per_seq_diff"].max()),
            max_nll_kbit_per_seq_diff=float(cols["nll_kbit_per_seq_diff"].max()),
        )

    np.savez(out / "levels.npz", **saved)
    (out / "schedule_match.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _mass_between(q, ls, dens, lo, hi):
    """Mass of density q (over log sigma) inside the [lo, hi] quantile interval of density dens."""
    c = np.concatenate([[0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(ls))])
    c = c / c[-1]
    a, b = np.interp(lo, c, ls), np.interp(hi, c, ls)
    m = (ls >= a) & (ls <= b)
    return np.trapz(q[m], ls[m])


if __name__ == "__main__":
    main()

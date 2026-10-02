"""
Analyses for the adaptive-compute study (PLAN.md §3–§13); definitions in NOTEBOOK.md (20:45).

Inputs : results/rollouts/s{s}_c{c}_k{k}.json  (nll/ntok/entropy as [R=4][200] per cell)
         results/features.json
Outputs: results/analysis/{heterogeneity,predictors,oracle}.json and arrays.npz for figures.

Notation: Q = −NLL/token of a final sample. d_r(n,s,c) = Q_c − Q_0 in rollout r (common random
numbers). M̂ = mean_r d_r, M̂_A = mean over r∈{0,1}, M̂_B = mean over r∈{2,3}.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
ADIR = RES / "analysis"
N_SEG = 8
ARMS = [0, 1, 2, 4]
BLOCK = 200
N_BOOT = 2000
SEED = 20261002
MATERIAL = 0.01
EXTRA_NFE = {s: (30 if s == 7 else 32) for s in range(N_SEG)}

STATE_B = ["H_mean", "H_tok_sd", "frac_uncertain_bits", "conf_mean", "conf_sd", "frac_uncertain_tokens"]
STATE_C = STATE_B + ["dH", "d2H"]
STATE_D = STATE_C + ["pred_change_1", "pred_change_4", "frac_stable_8"]
STATE_E = STATE_D + ["H_msb", "H_lsb", "resid_rms", "x_rms_over_sigma"]


# -----------------------------------------------------------------------------
def load(blocks: Optional[List[int]] = None):
    rdir = RES / "rollouts"
    if blocks is None:
        blocks = sorted({int(p.stem.split("_k")[1]) for p in rdir.glob("s*_c*_k*.json")})
    Q, E, src = {}, {}, {}
    for s in range(N_SEG):
        for c in ARMS:
            qs, es = [], []
            for k in blocks:
                p = rdir / f"s{s}_c{c}_k{k}.json"
                if not p.exists():
                    qs = None
                    break
                d = json.loads(p.read_text())
                nll, ntok = np.asarray(d["nll"]), np.asarray(d["ntok"])
                qs.append(-nll / np.maximum(ntok, 1))
                es.append(np.asarray(d["entropy"]))
                src[f"s{s}_c{c}_k{k}"] = str(p)
            if qs is not None:
                Q[(s, c)] = np.concatenate(qs, axis=1)   # [R, N]
                E[(s, c)] = np.concatenate(es, axis=1)
    return Q, E, blocks, src


def _boot_idx(n: int, b: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(b, n))


def _cov(a, b):
    return float(np.cov(a, b, ddof=1)[0, 1])


def _ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


# -----------------------------------------------------------------------------
# Experiment 1: heterogeneity
# -----------------------------------------------------------------------------
def heterogeneity(Q, E) -> Dict:
    out = {}
    for s in range(N_SEG):
        if (s, 0) not in Q:
            continue
        q0 = Q[(s, 0)]
        N = q0.shape[1]
        idx = _boot_idx(N)
        row = {"n_states": int(N)}
        # placebo: two independent zero-compute differences per state -> covariance should be ~0
        pA, pB = q0[0] - q0[1], q0[2] - q0[3]
        row["placebo_cov"] = _cov(pA, pB)
        row["placebo_cov_ci"] = _ci([_cov(pA[i], pB[i]) for i in idx])
        # how much of final quality is already fixed by the state at checkpoint s
        q0A, q0B = q0[:2].mean(0), q0[2:].mean(0)
        row["Q0_mean"] = float(q0.mean())
        row["Q0_between_state_sd"] = float(math.sqrt(max(_cov(q0A, q0B), 0)))
        row["Q0_single_rollout_sd"] = float(q0.std(ddof=1))
        arms = {}
        for c in ARMS[1:]:
            if (s, c) not in Q:
                continue
            d = Q[(s, c)] - q0
            M, MA, MB = d.mean(0), d[:2].mean(0), d[2:].mean(0)
            cov = _cov(MA, MB)
            boot_cov = np.array([_cov(MA[i], MB[i]) for i in idx])
            boot_mean = M[idx].mean(1)
            rAB = float(np.corrcoef(MA, MB)[0, 1])
            dE = (E[(s, c)] - E[(s, 0)]).mean(0)
            var_M = float(M.var(ddof=1))
            tau = math.sqrt(max(cov, 0.0))
            arms[str(c)] = dict(
                extra_nfe=c * EXTRA_NFE[s],
                mean_M=float(M.mean()), mean_M_ci=_ci(boot_mean),
                eta_per_extra_nfe=float(M.mean() / (c * EXTRA_NFE[s])),
                tau=tau, cov_AB=cov, cov_AB_ci=_ci(boot_cov),
                tau_ci=[math.sqrt(max(v, 0)) for v in _ci(boot_cov)],
                sd_M_hat_4rollouts=math.sqrt(var_M), noise_sd_M_hat=math.sqrt(max(var_M - cov, 0)),
                sd_single_rollout_d=float(d.std(ddof=1)),
                r_AB=rAB, reliability_4=float(2 * rAB / (1 + rAB)) if rAB > -1 else float("nan"),
                frac_states_M_pos=float((M > 0).mean()),
                mean_d_entropy=float(dE.mean()), corr_M_dentropy=float(np.corrcoef(M, dE)[0, 1]),
                case_B_cell=bool(boot_cov.size and np.percentile(boot_cov, 2.5) > 0 and tau >= MATERIAL),
            )
            # entropy-adjusted heterogeneity
            dEA = (E[(s, c)][:2] - E[(s, 0)][:2]).mean(0)
            dEB = (E[(s, c)][2:] - E[(s, 0)][2:]).mean(0)
            beta = float(np.polyfit(dE, M, 1)[0]) if dE.std() > 0 else 0.0
            arms[str(c)]["entropy_beta"] = beta
            arms[str(c)]["tau_entropy_adjusted"] = math.sqrt(max(_cov(MA - beta * dEA, MB - beta * dEB), 0))
        # marginal steps
        steps = {}
        for a, b in [(0, 1), (1, 2), (2, 4)]:
            if (s, a) in Q and (s, b) in Q:
                d = Q[(s, b)] - Q[(s, a)]
                steps[f"{a}->{b}"] = dict(mean=float(d.mean()), ci=_ci(d.mean(0)[idx].mean(1)),
                                          eta_per_extra_nfe=float(d.mean() / ((b - a) * EXTRA_NFE[s])))
        row["arms"], row["marginal_steps"] = arms, steps
        out[str(s)] = row
    cells = [(s, c) for s in out for c in out[s]["arms"] if out[s]["arms"][c]["case_B_cell"]]
    underpowered = [(s, c) for s in out for c in out[s]["arms"]
                    if out[s]["arms"][c]["tau"] >= MATERIAL and not out[s]["arms"][c]["case_B_cell"]]
    return {"per_checkpoint": out, "case": "B" if cells else "A",
            "case_B_cells": [f"s{s}_c{c}" for s, c in cells],
            "tau_ge_0.01_but_ci_includes_0": [f"s{s}_c{c}" for s, c in underpowered]}


# -----------------------------------------------------------------------------
# Experiment 2: predictors
# -----------------------------------------------------------------------------
def _features(N: int):
    f = json.loads((RES / "features.json").read_text())["features"]
    return {s: {k: np.asarray(v)[:N] for k, v in f[str(s)].items()} for s in range(N_SEG)}


def _ridge_fit_predict(Xtr, ytr, Xte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    A1 = np.hstack([A, np.ones((A.shape[0], 1))])
    B1 = np.hstack([B, np.ones((B.shape[0], 1))])
    reg = lam * np.eye(A1.shape[1])
    reg[-1, -1] = 0.0
    w = np.linalg.solve(A1.T @ A1 + reg, A1.T @ ytr)
    return B1 @ w


def _design(F, rows_s, rows_n, cols, onehot_t=False, log_sigma=False):
    X = [np.array([F[s][c][n] for s, n in zip(rows_s, rows_n)]) for c in cols]
    if log_sigma:
        X.append(np.array([F[s]["log_sigma"][n] for s, n in zip(rows_s, rows_n)]))
    X = np.stack(X, 1) if X else np.zeros((len(rows_s), 0))
    if onehot_t:
        oh = np.zeros((len(rows_s), N_SEG))
        oh[np.arange(len(rows_s)), rows_s] = 1
        X = np.hstack([X, oh])
    return X


def _within(v, rows_s):
    out = v.astype(float).copy()
    for s in np.unique(rows_s):
        m = rows_s == s
        out[m] -= out[m].mean()
    return out


def predictors(Q, arm: int) -> Dict:
    N = Q[(0, 0)].shape[1]
    F = _features(N)
    rows_s = np.repeat(np.arange(N_SEG), N)
    rows_n = np.tile(np.arange(N), N_SEG)
    MA = np.concatenate([(Q[(s, arm)] - Q[(s, 0)])[:2].mean(0) for s in range(N_SEG)])
    MB = np.concatenate([(Q[(s, arm)] - Q[(s, 0)])[2:].mean(0) for s in range(N_SEG)])
    models = {
        "A_timestep": dict(cols=[], onehot_t=True),
        "B_state": dict(cols=STATE_B),
        "C_state+dynamics": dict(cols=STATE_C),
        "D_+prediction_dynamics": dict(cols=STATE_D),
        "E_+bitstream(state-only full)": dict(cols=STATE_E),
        "F_full+timestep": dict(cols=STATE_E, onehot_t=True),
    }
    folds = np.random.default_rng(SEED).permutation(N) % 5  # group by state (sample) index
    fold_of_row = folds[rows_n]
    preds = {}
    for name, spec in models.items():
        X = _design(F, rows_s, rows_n, spec["cols"], spec.get("onehot_t", False))
        p = np.zeros_like(MA)
        for k in range(5):
            tr, te = fold_of_row != k, fold_of_row == k
            if X.shape[1] == 0:
                p[te] = MA[tr].mean()
            else:
                p[te] = _ridge_fit_predict(X[tr], MA[tr], X[te])
        preds[name] = p
    preds["half_A_estimate_itself(noisy, not a ceiling)"] = MA

    # bootstrap over states (all checkpoints of a state together)
    idx = _boot_idx(N, b=1000)
    def metrics(p):
        pooled = spearmanr(p, MB).correlation
        within = spearmanr(_within(p, rows_s), _within(MB, rows_s)).correlation
        r2 = 1 - np.mean((MB - p) ** 2) / np.var(MB)
        per_s = [float(spearmanr(p[rows_s == s], MB[rows_s == s]).correlation)
                 if np.std(p[rows_s == s]) > 0 else float("nan") for s in range(N_SEG)]
        return pooled, within, r2, per_s

    out = {}
    for name, p in preds.items():
        pooled, within, r2, per_s = metrics(p)
        bw = []
        for b in idx[:300]:
            rows = (b[None, :] + N * np.arange(N_SEG)[:, None]).reshape(-1)
            if np.std(_within(p[rows], rows_s[rows])) > 0:
                bw.append(spearmanr(_within(p[rows], rows_s[rows]), _within(MB[rows], rows_s[rows])).correlation)
        # calibration: regress M_B on prediction (slope 1 = calibrated)
        slope = float(np.polyfit(p, MB, 1)[0]) if np.std(p) > 0 else float("nan")
        out[name] = dict(spearman_pooled=float(pooled), spearman_within_timestep=float(within),
                         spearman_within_ci=_ci(bw) if bw else [float("nan")] * 2,
                         r2_vs_MB=float(r2), calibration_slope=slope,
                         spearman_per_checkpoint=per_s)

    # cross-timestep generalisation: train on 6 checkpoints, test on held-out pair
    xt = {}
    for label, spec in {"state_only": dict(cols=STATE_E), "state+log_sigma": dict(cols=STATE_E, log_sigma=True)}.items():
        X = _design(F, rows_s, rows_n, spec["cols"], log_sigma=spec.get("log_sigma", False))
        p = np.zeros_like(MA)
        for held in [(0, 1), (2, 3), (4, 5), (6, 7)]:
            te = np.isin(rows_s, held)
            p[te] = _ridge_fit_predict(X[~te], MA[~te], X[te])
        xt[label] = dict(spearman_pooled=float(spearmanr(p, MB).correlation),
                         spearman_within_timestep=float(spearmanr(_within(p, rows_s), _within(MB, rows_s)).correlation),
                         r2_vs_MB=float(1 - np.mean((MB - p) ** 2) / np.var(MB)))
    # best single feature (chosen on half A, pooled)
    singles = {}
    for c in STATE_E + ["log_sigma"]:
        v = np.array([F[s][c][n] for s, n in zip(rows_s, rows_n)])
        if v.std() > 0:
            singles[c] = float(spearmanr(v, MA).correlation)
    best = max(singles, key=lambda k: abs(singles[k]))
    return dict(arm=arm, n_states=int(N), models=out, cross_timestep=xt,
                single_feature_spearman_vs_MA=singles, best_single_feature=best,
                best_single_sign=float(np.sign(singles[best])),
                predictor_cost="features reuse D from the previous step; O(sequence) arithmetic, no extra NFE"), \
        dict(preds=preds, MA=MA, MB=MB, rows_s=rows_s, rows_n=rows_n, F=F, best=best,
             best_sign=float(np.sign(singles[best])))


# -----------------------------------------------------------------------------
# Experiment 3: oracle allocation (select on half A, evaluate on half B)
# -----------------------------------------------------------------------------
def oracle(Q, arm: int, pr: Dict) -> Dict:
    N = Q[(0, 0)].shape[1]
    rows_s, rows_n = pr["rows_s"], pr["rows_n"]
    gA, gB = pr["MA"], pr["MB"]
    nfe_cell = np.array([arm * EXTRA_NFE[s] for s in rows_s], dtype=float)
    fr = [1 / 8, 2 / 8, 4 / 8, 6 / 8, 1.0]
    ncell = len(gA)
    F = pr["F"]
    best = np.array([F[s][pr["best"]][n] for s, n in zip(rows_s, rows_n)]) * pr["best_sign"]
    seg_rank = np.argsort([-gA[rows_s == s].mean() for s in range(N_SEG)])
    samp_score = np.array([gA[rows_n == n].sum() for n in range(N)])
    score = {
        "state_oracle(t x example)": gA,
        "learned_state_only(E)": pr["preds"]["E_+bitstream(state-only full)"],
        "learned_state+t(F)": pr["preds"]["F_full+timestep"],
        f"heuristic_{pr['best']}": best,
        "in-sample_oracle(optimistic,B on B)": gB,
    }
    idx = _boot_idx(N, b=1000)

    def take_top(sc, k):
        sel = np.zeros(ncell, bool)
        sel[np.argsort(-sc, kind="stable")[:k]] = True
        return sel

    res = {}
    for f in fr:
        k = int(round(f * ncell))
        sels = {}
        # temporal: whole checkpoints, best first (by half A)
        nseg = int(round(f * N_SEG))
        sels["temporal_oracle"] = np.isin(rows_s, seg_rank[:nseg])
        # example: whole samples, best first (by half A)
        top_n = np.argsort(-samp_score, kind="stable")[: int(round(f * N))]
        sels["example_oracle"] = np.isin(rows_n, top_n)
        for name, sc in score.items():
            sels[name] = take_top(sc, k)
        row = {}
        uni = f * gB.mean() * N_SEG  # expected per-sample gain of random selection
        for name, sel in sels.items():
            g = np.bincount(rows_n[sel], weights=gB[sel], minlength=N)  # per-sample gain (sum over cells)
            nfe = np.bincount(rows_n[sel], weights=nfe_cell[sel], minlength=N)
            per_sample = float(g.mean())
            # paired vs uniform-random: uniform per-sample gain = f * sum_s gB(n,s)
            u = f * np.bincount(rows_n, weights=gB, minlength=N)
            diff = g - u
            row[name] = dict(gain_per_sample=per_sample, extra_nfe_per_sample=float(nfe.mean()),
                             gain_minus_uniform=float(diff.mean()), gain_minus_uniform_ci=_ci(diff[idx].mean(1)),
                             share_cells_by_checkpoint=[float(sel[rows_s == s].mean()) for s in range(N_SEG)])
        row["uniform"] = dict(gain_per_sample=float(uni), extra_nfe_per_sample=float(f * nfe_cell.sum() / N))
        res[f"{f:.3f}"] = row
    return dict(arm=arm, budgets_fraction_of_cells=fr, baseline_nfe=255, results=res,
                note=("gain = Σ over selected (state, checkpoint) cells of M̂_B, per sample, nats/token; "
                      "assumes gains add across checkpoints of one sample (not tested); "
                      "local (token/bit) allocation not testable with this branch design"))


# -----------------------------------------------------------------------------
def main(blocks=None) -> None:
    ADIR.mkdir(parents=True, exist_ok=True)
    Q, E, blocks, src = load(blocks)
    het = heterogeneity(Q, E)
    het["blocks"], het["n_cells_loaded"] = blocks, len(Q)
    (ADIR / "heterogeneity.json").write_text(json.dumps(het, indent=1))
    print("[analyze] case:", het["case"], "cells:", het["case_B_cells"], "underpowered:",
          het["tau_ge_0.01_but_ci_includes_0"])
    arrays = {}
    if all((s, c) in Q for s in range(N_SEG) for c in ARMS):
        preds_all, orc_all = {}, {}
        for arm in (1, 4):
            p, pr = predictors(Q, arm)
            preds_all[str(arm)] = p
            orc_all[str(arm)] = oracle(Q, arm, pr)
            for k in ("MA", "MB", "rows_s", "rows_n"):
                arrays[f"{k}_c{arm}"] = pr[k]
            for k, v in pr["preds"].items():
                arrays[f"pred_c{arm}_{k}"] = v
        (ADIR / "predictors.json").write_text(json.dumps(preds_all, indent=1))
        (ADIR / "oracle.json").write_text(json.dumps(orc_all, indent=1))
    for (s, c), q in Q.items():
        arrays[f"Q_s{s}_c{c}"] = q
        arrays[f"E_s{s}_c{c}"] = E[(s, c)]
    np.savez_compressed(ADIR / "arrays.npz", **arrays)
    print(f"[analyze] wrote {ADIR}")

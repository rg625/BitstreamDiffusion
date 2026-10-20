"""Analysis of the adaptive-compute branches (CPU only).

Quality Q = -(GPT-2-large NLL per token) of one final sample (nats/token, higher
is better). For state (chunk, sample i, checkpoint s) and continuation stream r:

    M_k(i,s,r) = Q(k extra units in segment s, stream r) - Q(0 extra, stream r)

Paired within a stream: the same starting state and the same churn noise, so
M is the effect of the extra compute on that continuation.

Heterogeneity is separated from rollout noise with the two independent streams:
    signal variance  = Cov_i(M(r=0), M(r=1))  -- variance of E[M | state]
    noise variance   = Var_i(M(r=0) - M(r=1)) / 2
Predictors and allocation policies are fitted on one stream and scored on the
other (both directions averaged), so selection never sees its own noise.

    python -m experiments.compute_allocation.analyze
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.compute_allocation.run import K_UNITS, N_SEG, OUT_DIR, STATE_DIR

RNG = np.random.default_rng(0)
N_BOOT = 2000

STATE_FEATS = ["H_mean", "frac_unc_bits", "conf_mean", "conf_var", "tokH_mean", "tokH_max",
               "tokH_top10_share", "frac_unc_tokens", "resid"]
DYN_FEATS = ["dH1", "dH8", "d2H"]
PRED_FEATS = ["flip1", "flip8", "bitflip1"]
BIT_FEATS = [f"bitplaneH_{j}" for j in range(15)]
FEATURE_SETS = {
    "A_timestep": [],
    "B_state": STATE_FEATS,
    "C_state+dyn": STATE_FEATS + DYN_FEATS,
    "D_+pred_dyn": STATE_FEATS + DYN_FEATS + PRED_FEATS,
    "E_+bitplanes": STATE_FEATS + DYN_FEATS + PRED_FEATS + BIT_FEATS,
}


# ---------------------------------------------------------------- data
def load(out_dir: Path, state_dir: Path):
    files = sorted(glob.glob(str(out_dir / "scores_chunk*.npz")))
    if not files:
        raise SystemExit(f"no scores in {out_dir}")
    Q, E, feats, anchors, null = {}, {}, {}, [], []
    ref_nll = ref_tok = 0.0
    ref_ent = []
    for f in files:
        z = np.load(f)
        c = int(z["chunk"][0])
        q = -z["nll_sum"] / np.maximum(z["n_tok"], 1)
        s, k, r, i = z["s"], z["k"], z["r"], z["i"]
        n = int(i.max()) + 1
        Qc = np.full((n, N_SEG, len(K_UNITS), 2), np.nan)
        Ec = np.full_like(Qc, np.nan)
        kidx = {kk: j for j, kk in enumerate(K_UNITS)}
        br = s >= 0
        Qc[i[br], s[br], [kidx[x] for x in k[br]], r[br]] = q[br]
        Ec[i[br], s[br], [kidx[x] for x in k[br]], r[br]] = z["entropy"][br]
        Q[c], E[c] = Qc, Ec
        ref = s < 0
        ref_nll += float(np.nansum(z["nll_sum"][ref]))
        ref_tok += float(z["n_tok"][ref].sum())
        ref_ent.append(z["entropy"][ref])
        # null replay: (k=0, r=0) re-runs the reference from each saved state
        ids_ref = z["ids"][ref]
        for ss in range(N_SEG):
            m = br & (s == ss) & (k == 0) & (r == 0)
            o = np.argsort(i[m])
            same = (z["ids"][m][o] == ids_ref).all(1)
            null.append({"chunk": c, "s": ss, "identical_frac": float(same.mean()),
                         "abs_dQ_mean": float(np.nanmean(np.abs(q[m][o] - q[ref])))})
        ref_pt = torch.load(state_dir / f"chunk{c}" / "reference.pt", weights_only=False)
        starts = [a for a, _ in ref_pt["segments"]]
        names = list(ref_pt["features"][starts[0]].keys())
        feats[c] = {nm: np.stack([ref_pt["features"][a][nm].numpy() for a in starts], 1) for nm in names}
        aj = out_dir / f"anchor_chunk{c}.json"
        if aj.exists():
            anchors.append(json.loads(aj.read_text()))
    chunks = sorted(Q)
    Qa = np.concatenate([Q[c] for c in chunks], 0)
    Ea = np.concatenate([E[c] for c in chunks], 0)
    Fa = {nm: np.concatenate([feats[c][nm] for c in chunks], 0) for nm in feats[chunks[0]]}
    pooled = {"n": int(Qa.shape[0]), "genppl_batched": math.exp(ref_nll / ref_tok),
              "entropy_mean": float(np.concatenate(ref_ent).mean())}
    return Qa, Ea, Fa, anchors, pooled, null


def boot_ci(fn, *arrays, n_boot=N_BOOT):
    n = arrays[0].shape[0]
    vals = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        vals.append(fn(*[a[idx] for a in arrays]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def _cov(a, b):
    return float(np.mean((a - a.mean()) * (b - b.mean())))


# ---------------------------------------------------------------- H1: heterogeneity
def heterogeneity(Q, E):
    out = []
    for kj, k in enumerate(K_UNITS):
        if k == 0:
            continue
        for s in range(N_SEG):
            m0 = Q[:, s, kj, 0] - Q[:, s, 0, 0]
            m1 = Q[:, s, kj, 1] - Q[:, s, 0, 1]
            ok = np.isfinite(m0) & np.isfinite(m1)
            m0, m1 = m0[ok], m1[ok]
            mbar = (m0 + m1) / 2
            dE = ((E[:, s, kj, 0] - E[:, s, 0, 0]) + (E[:, s, kj, 1] - E[:, s, 0, 1]))[ok] / 2
            cov = _cov(m0, m1)
            out.append({
                "k": k, "s": s, "n": int(ok.sum()),
                "mean_M": float(mbar.mean()), "mean_M_ci": boot_ci(lambda a: a.mean(), mbar),
                "mean_dE": float(dE.mean()),
                "signal_var": cov, "signal_var_ci": boot_ci(_cov, m0, m1),
                "signal_sd": math.sqrt(max(cov, 0.0)),
                "noise_var": float(np.var(m0 - m1) / 2),
                "obs_var_single": float((np.var(m0) + np.var(m1)) / 2),
                "retest_r": float(np.corrcoef(m0, m1)[0, 1]),
                "retest_r_ci": boot_ci(lambda a, b: np.corrcoef(a, b)[0, 1], m0, m1),
                "frac_helped": float((mbar > 0).mean()),
            })
    return out


def decide(het, k=1, abs_thr=0.01):
    """Pre-registered (docs/adaptive_compute_lm1b.md): Case A/B/C at k=1."""
    rows = [h for h in het if h["k"] == k]
    strong = [h["s"] for h in rows
              if h["signal_var_ci"][0] > 0 and h["signal_sd"] >= max(abs(h["mean_M"]), abs_thr)]
    any_signal = [h["s"] for h in rows if h["signal_var_ci"][0] > 0]
    if strong:
        return "B", strong
    if not any_signal:
        return "A", []
    return "weak", any_signal


# ---------------------------------------------------------------- Exp 2: predictors
def _design(F, names, smp, s_idx, add_t):
    """Rows are (sample, checkpoint) pairs; F[name] is [N, N_SEG]."""
    cols = [F[nm][smp, s_idx] for nm in names]
    X = np.stack(cols, 1) if cols else np.zeros((len(s_idx), 0))
    if add_t:
        X = np.concatenate([X, np.eye(N_SEG)[s_idx]], 1)
    return X


def _ridge_fit_predict(Xtr, ytr, Xte, alphas=(0.1, 1, 10, 100, 1000)):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    best, best_a = None, None
    # inner 3-fold for alpha
    n = len(ytr)
    fold = np.arange(n) % 3
    for a in alphas:
        err = 0.0
        for f in range(3):
            tr, va = fold != f, fold == f
            w = np.linalg.solve(Xtr[tr].T @ Xtr[tr] + a * np.eye(Xtr.shape[1]), Xtr[tr].T @ (ytr[tr] - ym))
            err += float(((Xtr[va] @ w + ym - ytr[va]) ** 2).sum())
        if best is None or err < best:
            best, best_a = err, a
    w = np.linalg.solve(Xtr.T @ Xtr + best_a * np.eye(Xtr.shape[1]), Xtr.T @ (ytr - ym))
    return Xte @ w + ym


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def predictors(Q, F, k=1, n_folds=5):
    """Fit on stream a, score against stream b (both directions); grouped by sample."""
    kj = K_UNITS.index(k)
    N = Q.shape[0]
    M = Q[:, :, kj, :] - Q[:, :, 0:1, :][:, :, 0, :]          # [N, S, 2]
    smp = np.repeat(np.arange(N), N_SEG)
    s_idx = np.tile(np.arange(N_SEG), N)
    y = M.reshape(N * N_SEG, 2)
    ok = np.isfinite(y).all(1)
    fold = (smp * 7919) % n_folds
    sets = {nm: (feats, False) for nm, feats in FEATURE_SETS.items() if feats}
    sets["A_timestep"] = ([], True)
    sets["F_full+t"] = (FEATURE_SETS["E_+bitplanes"], True)
    sets["B_state+t"] = (STATE_FEATS, True)
    pooled_cov = _cov(y[ok, 0], y[ok, 1])
    ceiling_r = math.sqrt(max(pooled_cov, 0) / np.var(y[ok, 0])) if np.var(y[ok, 0]) > 0 else float("nan")
    res = {"_ceiling_corr_vs_other_stream": ceiling_r}
    preds_all = {}
    for name, (feats, add_t) in sets.items():
        X = _design(F, feats, smp, s_idx, add_t)
        r_pear, r_spear = [], []
        pred_full = np.full((N * N_SEG, 2), np.nan)
        for a, b in ((0, 1), (1, 0)):
            pred = np.full(N * N_SEG, np.nan)
            for f in range(n_folds):
                tr, te = ok & (fold != f), ok & (fold == f)
                pred[te] = _ridge_fit_predict(X[tr], y[tr, a], X[te])
            pred_full[:, a] = pred
            r_pear.append(float(np.corrcoef(pred[ok], y[ok, b])[0, 1]))
            r_spear.append(_spearman(pred[ok], y[ok, b]))
        # within-timestep: does it rank states at the SAME checkpoint?
        within = []
        for s in range(N_SEG):
            m = ok & (s_idx == s)
            within.append(np.mean([_spearman(pred_full[m, a], y[m, 1 - a]) for a in (0, 1)]))
        res[name] = {"pearson_vs_other_stream": float(np.mean(r_pear)),
                     "spearman_vs_other_stream": float(np.mean(r_spear)),
                     "within_t_spearman_mean": float(np.nanmean(within)),
                     "within_t_spearman": [float(w) for w in within]}
        preds_all[name] = pred_full
    # cross-timestep generalisation: leave one checkpoint out, state only vs state + log_sigma
    loco = {}
    for name, feats in (("state_only", FEATURE_SETS["E_+bitplanes"]),
                        ("state+log_sigma", FEATURE_SETS["E_+bitplanes"] + ["log_sigma"])):
        X = _design(F, feats, smp, s_idx, False)
        rs = []
        for s in range(N_SEG):
            tr, te = ok & (s_idx != s), ok & (s_idx == s)
            r_ab = []
            for a in (0, 1):
                p = _ridge_fit_predict(X[tr], y[tr, a], X[te])
                r_ab.append(_spearman(p, y[te, 1 - a]))
            rs.append(float(np.mean(r_ab)))
        loco[name] = rs
    res["leave_one_checkpoint_out_spearman"] = loco
    return res, preds_all


# ---------------------------------------------------------------- Exp 3: offline oracle
def _mck(gains, costs, budget_per_item):
    """Multiple-choice knapsack by Lagrangian bisection. gains [n, K], costs [K].
    Returns choice [n] meeting mean cost <= budget_per_item (largest lambda feasible)."""
    lo, hi = 0.0, float(np.nanmax(np.abs(gains))) * 10 + 1.0
    best = np.zeros(gains.shape[0], dtype=int)
    for _ in range(60):
        lam = (lo + hi) / 2
        ch = np.nanargmax(gains - lam * costs[None, :], 1)
        if costs[ch].mean() <= budget_per_item + 1e-12:
            best, hi = ch, lam
        else:
            lo = lam
    return best


def oracle(Q, preds_by_k=None, budgets=(1, 2, 4, 8, 16)):
    """Units per sample, summed over its 8 checkpoints; gains assumed additive."""
    N = Q.shape[0]
    K = np.array(K_UNITS, dtype=float)
    G = Q - Q[:, :, 0:1, :]                                    # [N, S, K, 2]
    ok = np.isfinite(G).all(axis=(1, 2, 3))
    G = G[ok]
    out = []
    for B in budgets:
        row = {"budget_units_per_sample": B, "extra_nfe_frac": B * 32 / 255}
        vals = {"uniform": [], "temporal": [], "example": [], "state": [], "predicted": []}
        for a, b in ((0, 1), (1, 0)):
            Ga, Gb = G[..., a], G[..., b]
            if B % N_SEG == 0 and (B // N_SEG) in K_UNITS:
                kj = K_UNITS.index(B // N_SEG)
                vals["uniform"].append(Gb[:, :, kj].sum(1).mean())
            # temporal: one k per checkpoint, same for every sample
            mean_a = Ga.mean(0)                                # [S, K]
            best, best_v = None, -np.inf
            for combo in itertools.product(range(len(K_UNITS)), repeat=N_SEG):
                if K[list(combo)].sum() > B:
                    continue
                v = mean_a[np.arange(N_SEG), combo].sum()
                if v > best_v:
                    best, best_v = combo, v
            vals["temporal"].append(Gb[:, np.arange(N_SEG), best].sum(1).mean())
            # example: one k per sample at every checkpoint (cost 8k)
            ch = _mck(Ga.sum(1), K * N_SEG, B)
            vals["example"].append(Gb.sum(1)[np.arange(len(ch)), ch].mean())
            # state: one k per (sample, checkpoint)
            ch = _mck(Ga.reshape(-1, len(K)), K, B / N_SEG).reshape(-1, N_SEG)
            vals["state"].append(np.take_along_axis(Gb, ch[..., None], 2)[..., 0].sum(1).mean())
            if preds_by_k is not None:
                P = preds_by_k[..., a][ok]                     # predicted gains from stream a
                ch = _mck(P.reshape(-1, len(K)), K, B / N_SEG).reshape(-1, N_SEG)
                vals["predicted"].append(np.take_along_axis(Gb, ch[..., None], 2)[..., 0].sum(1).mean())
        for kname, v in vals.items():
            row[kname] = float(np.mean(v)) if v else None
        out.append(row)
    return out


def predicted_gains(Q, F, feats=None):
    """Cross-fitted predicted gain for every k, from the full feature set + checkpoint."""
    feats = feats or FEATURE_SETS["E_+bitplanes"]
    N = Q.shape[0]
    P = np.zeros((N, N_SEG, len(K_UNITS), 2))
    s_idx = np.tile(np.arange(N_SEG), N)
    smp = np.repeat(np.arange(N), N_SEG)
    fold = (smp * 7919) % 5
    X = _design(F, feats, smp, s_idx, True)
    for kj, k in enumerate(K_UNITS):
        if k == 0:
            continue
        y = (Q[:, :, kj, :] - Q[:, :, 0, :]).reshape(N * N_SEG, 2)
        ok = np.isfinite(y).all(1)
        for a in (0, 1):
            pred = np.zeros(N * N_SEG)
            for f in range(5):
                tr, te = ok & (fold != f), fold == f
                pred[te] = _ridge_fit_predict(X[tr], y[tr, a], X[te])
            P[:, :, kj, a] = pred.reshape(N, N_SEG)
    return P


# ---------------------------------------------------------------- figures
def figures(Q, F, het, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kj = K_UNITS.index(1)
    m0 = Q[:, :, kj, 0] - Q[:, :, 0, 0]
    m1 = Q[:, :, kj, 1] - Q[:, :, 0, 1]
    mbar = (m0 + m1) / 2
    noise = (m0 - m1) / 2          # same variance as the noise part of mbar
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    cmap = plt.get_cmap("viridis")
    for s in range(N_SEG):
        ax[0].scatter(F["tokH_mean"][:, s], mbar[:, s], s=4, alpha=0.4, color=cmap(s / (N_SEG - 1)),
                      label=f"seg {s}")
    ax[0].set_xlabel("mean token entropy of D at segment start (bits)")
    ax[0].set_ylabel("Q(+1) - Q(+0)  (nats/token, mean of 2 streams)")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].legend(markerscale=3, fontsize=7, ncol=2)
    pos = np.arange(N_SEG)
    ax[1].violinplot([mbar[:, s][np.isfinite(mbar[:, s])] for s in range(N_SEG)], pos - 0.18, widths=0.35)
    ax[1].violinplot([noise[:, s][np.isfinite(noise[:, s])] for s in range(N_SEG)], pos + 0.18, widths=0.35)
    sd = [h["signal_sd"] for h in het if h["k"] == 1]
    ax[1].plot(pos, sd, "k^-", label="state signal SD (cross-stream)")
    ax[1].set_xticks(pos)
    ax[1].set_xlabel("segment (left: Q(+1)-Q(+0); right: noise-only (r0-r1)/2)")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_heterogeneity.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--states", default=str(STATE_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out)
    Q, E, F, anchors, pooled, null = load(out_dir, Path(args.states))
    het = heterogeneity(Q, E)
    case, segs = decide(het)
    pred_res, _ = predictors(Q, F, k=1)
    P = predicted_gains(Q, F)
    orc = oracle(Q, P)
    summary = {"n_samples": int(Q.shape[0]), "anchors": anchors, "pooled_reference": pooled,
               "null_replay": null, "heterogeneity": het, "decision_k1": {"case": case, "segments": segs},
               "predictors_k1": pred_res, "oracle": orc}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    try:
        figures(Q, F, het, out_dir)
    except Exception as e:  # figures are a convenience
        print(f"[fig] skipped: {e}")
    print(json.dumps({"pooled_reference": pooled, "decision_k1": summary["decision_k1"]}, indent=2))
    print("\nk  s  mean_M     signal_sd  noise_sd  retest_r  dE")
    for h in het:
        print(f"{h['k']}  {h['s']}  {h['mean_M']:+.4f}  {h['signal_sd']:.4f}    "
              f"{math.sqrt(h['noise_var']):.4f}    {h['retest_r']:+.3f}   {h['mean_dE']:+.4f}")
    print("\npredictors (k=1), correlation with the OTHER stream:")
    print(f"  ceiling {pred_res['_ceiling_corr_vs_other_stream']:.3f}")
    for nm, v in pred_res.items():
        if nm.startswith("_") or nm.startswith("leave"):
            continue
        print(f"  {nm:14s} pearson {v['pearson_vs_other_stream']:+.3f}  spearman "
              f"{v['spearman_vs_other_stream']:+.3f}  within-t spearman {v['within_t_spearman_mean']:+.3f}")
    print("  leave-one-checkpoint-out:", pred_res["leave_one_checkpoint_out_spearman"])
    print("\noracle (additive), mean gain in Q (nats/token) per sample:")
    for r in orc:
        print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()

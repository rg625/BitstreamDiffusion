"""experiments/compute_allocation/analyze.py on synthetic branch tables with known truth."""
import numpy as np

from experiments.compute_allocation import analyze as A
from experiments.compute_allocation.run import K_UNITS, N_SEG


def _synthetic(N, slope, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    feat = rng.normal(size=(N, N_SEG))
    F = {nm: rng.normal(size=(N, N_SEG)) for nm in A.STATE_FEATS + A.DYN_FEATS + A.PRED_FEATS + A.BIT_FEATS}
    F["H_mean"] = feat
    F["log_sigma"] = np.tile(np.linspace(4, -2, N_SEG), (N, 1))
    base = rng.normal(-4.0, 0.3, size=(N, N_SEG))
    Q = np.zeros((N, N_SEG, len(K_UNITS), 2))
    for kj, k in enumerate(K_UNITS):
        gain = np.log1p(k) * (0.01 + slope * feat)           # E[M_k | state]
        for r in range(2):
            Q[:, :, kj, r] = base + gain + (rng.normal(0, noise, size=(N, N_SEG)) if k else 0.0)
    E = np.full_like(Q, 4.3)
    return Q, E, F


def test_detects_planted_heterogeneity():
    Q, E, F = _synthetic(1000, slope=0.03, noise=0.02)
    het = A.heterogeneity(Q, E)
    true_var = (np.log1p(1) * 0.03) ** 2
    est = np.mean([h["signal_var"] for h in het if h["k"] == 1])
    assert abs(est - true_var) < 0.15 * true_var
    case, _ = A.decide(het)
    assert case == "B"
    res, _ = A.predictors(Q, F, k=1)
    assert res["B_state"]["within_t_spearman_mean"] > 0.3
    assert res["A_timestep"]["within_t_spearman_mean"] < 0.1
    orc = A.oracle(Q, A.predicted_gains(Q, F), budgets=(4,))
    assert orc[0]["state"] > orc[0]["temporal"] + 0.005


def test_no_heterogeneity_is_case_a():
    Q, E, F = _synthetic(400, slope=0.0)
    case, segs = A.decide(A.heterogeneity(Q, E))
    assert case in ("A", "weak") and len(segs) <= 1

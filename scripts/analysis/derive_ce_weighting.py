"""Derive the sigma-weighting for the binary_ce arm.

REFERENCE MODEL
---------------
CoBit's matched filter is mf = (x_t - 0.5)/sigma^2, which IS the exact posterior
log-odds of a single bit under x0 ~ Bernoulli(1/2), x_t = x0 + sigma*eps. The
network learns a residual on top of it. So the independent-bit posterior is not
an arbitrary reference -- it is the architecture's own inductive bias, and it is
the right place to read off the sigma-dependence of each objective.

With x0 = 1 (the x0 = 0 case is the mirror image and every statistic below is
symmetric under lambda -> -lambda):

    lambda = (x_t - 1/2)/sigma^2 = 1/(2 sigma^2) + eps/sigma  ~  N(mu, 2 mu),
    mu = 1/(2 sigma^2),      q = sigmoid(lambda).

Bayes risk per bit of each proper scoring rule:

    R_sm(sigma) = E[(x0 - q)^2] = E[q(1-q)]        (Brier)
    R_ce(sigma) = E[H(q)],  H in nats              (log loss)

Both are 1-D Gaussian expectations -> Gauss-Hermite quadrature, no free
parameters, no fitting.
"""
from __future__ import annotations

import json
import numpy as np

SIGMA_DATA = 0.399844765663147   # runs/.../sigma_data.json (overrides cfg's 0.5)
SIGMA_MIN, SIGMA_MAX = 0.002, 80.0


def w_edm(sigma, sigma_data=SIGMA_DATA):
    """The existing weight: (sigma^2 + sd^2)/(sigma^2 sd^2) = 1/c_out(sigma)^2."""
    s2 = np.asarray(sigma, dtype=np.float64) ** 2
    return (s2 + sigma_data ** 2) / (s2 * sigma_data ** 2)


# Integrate in LOG space on a fixed fine grid in lambda.
#
# Gauss-Hermite fails here. At small sigma, mu = 1/(2 sigma^2) is enormous
# (1.25e5 at sigma=0.002) and both integrands are concentrated near lambda = 0,
# which is ~250 standard deviations into the lower tail: nodes placed relative
# to the Gaussian never sample the region that carries the integral, and the
# risks underflow to 0 (or NaN) in double precision. They genuinely ARE ~e^-31250
# -- but their RATIO is perfectly well conditioned, and the ratio is what the
# weighting needs. So accumulate log(g) + log(f) and log-sum-exp.
_T = np.linspace(-100.0, 100.0, 400001)
_DT = float(_T[1] - _T[0])


def _log_normal_pdf(t, mu, var):
    return -0.5 * np.log(2 * np.pi * var) - (t - mu) ** 2 / (2 * var)


def _log_expect(log_g, mu, var):
    """log E[g(lambda)] for lambda ~ N(mu, var), g given by its log."""
    h = log_g(_T) + _log_normal_pdf(_T, mu, var)
    m = h.max()
    return float(m + np.log(np.sum(np.exp(h - m)) * _DT))


def _sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700))),
                    np.exp(np.clip(z, -700, 700)) / (1.0 + np.exp(np.clip(z, -700, 700))))


def _log_q1mq(t):
    """log[q(1-q)] = -2 log(2 cosh(t/2)), stable for large |t|."""
    a = np.abs(t) / 2.0
    return -2.0 * (a + np.log1p(np.exp(-2.0 * a)) + np.log(1.0))


def _log_H(t):
    """log H(sigmoid(t)) in nats, stable: H = log(1+e^-|t|) + |t| e^-|t|/(1+e^-|t|)."""
    a = np.abs(t)
    e = np.exp(-a)
    H = np.log1p(e) + a * e / (1.0 + e)
    return np.log(np.maximum(H, 1e-320))


def bayes_risks_log(sigma):
    """Returns (log R_sm, log R_ce). Logs, because at sigma=0.002 both are ~e^-31250."""
    mu = 1.0 / (2.0 * sigma ** 2)
    var = 2.0 * mu
    return _log_expect(_log_q1mq, mu, var), _log_expect(_log_H, mu, var)


def bayes_risks(sigma):
    ls, lc = bayes_risks_log(sigma)
    return float(np.exp(ls)), float(np.exp(lc))


def main():
    sigmas = np.array([0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4,
                       SIGMA_DATA, 0.75, 1.0, 3.0, 10.0, 40.0, 80.0])
    sigmas = np.unique(sigmas)
    rows = []
    for s_ in sigmas:
        ls, lc = bayes_risks_log(float(s_))
        ratio = float(np.exp(ls - lc))          # R_sm / R_ce, well conditioned
        rows.append(dict(sigma=float(s_), log_R_sm=ls, log_R_ce=lc,
                         R_sm_over_R_ce=ratio, w_edm=float(w_edm(s_))))

    print("Bayes risk per bit, matched-filter reference model "
          f"(sigma_data={SIGMA_DATA:.6f})")
    print("logs, because at sigma=0.002 both risks are ~e^-31250\n")
    print(f"{'sigma':>8} {'log R_sm':>13} {'log R_ce':>13} {'R_sm/R_ce':>11} "
          f"{'w_edm':>12}")
    for r in rows:
        print(f"{r['sigma']:>8g} {r['log_R_sm']:>13.4f} {r['log_R_ce']:>13.4f} "
              f"{r['R_sm_over_R_ce']:>11.5f} {r['w_edm']:>12.4e}")

    ra = np.array([r["R_sm_over_R_ce"] for r in rows])
    print(f"\nR_sm/R_ce spans {ra.min():.5f} .. {ra.max():.5f}  "
          f"-> total variation factor {ra.max()/ra.min():.4f}")
    print(f"  analytic limits: sigma->0  {3/np.pi**2:.5f} (= 1 / (pi^2/3))")
    print(f"                   sigma->inf {0.25/np.log(2):.5f} (= 0.25 / ln 2)")

    print("\n--- Criterion 1: match LOSS scale ---")
    print("  w_ce(sigma) = w_edm(sigma) * R_sm(sigma)/R_ce(sigma)")
    print("  -> a sigma-dependent factor spanning only "
          f"{ra.min():.3f}..{ra.max():.3f}; anchored at sigma_data it is")
    anchor = [r for r in rows if abs(r["sigma"] - SIGMA_DATA) < 1e-9][0]
    rel = ra / anchor["R_sm_over_R_ce"]
    print(f"     a relative reweighting of {rel.min():.4f}..{rel.max():.4f} "
          "across the whole sigma range.")

    print("\n--- Criterion 2: match GRADIENT scale ---")
    print("  dL_ce/d_ell = w (D-x0);  dL_sm/d_ell = w (D-x0) D(1-D)")
    print("  equal gradients require w_ce = w_sm * D(1-D), i.e. multiplying CE")
    print("  by the very suppression factor under test. SELF-DEFEATING: it")
    print("  turns CE back into SM. Rejected on principle, not on magnitude.")

    print("\n--- Criterion 3: preserve sigma-INVARIANCE (recommended) ---")
    print("  w(sigma)*R(sigma) = const is the design intent of the EDM weight.")
    print("  Check it for the EXISTING SM recipe: w_edm * R_sm should be flat.")
    print(f"\n{'sigma':>8} {'log(w_edm*R_sm)':>18} {'log(w_ce_inv*R_ce)':>20}")
    lw = []
    for r in rows:
        lw.append(np.log(r["w_edm"]) + r["log_R_sm"])
        print(f"{r['sigma']:>8g} {lw[-1]:>18.4f} {'(flat by construction)':>20}")
    lw = np.array(lw)
    print(f"\n  log(w_edm*R_sm) spans {lw.min():.4f} .. {lw.max():.4f}")
    print(f"  -> the EXISTING SM recipe is NOT sigma-invariant either; it varies")
    print(f"     by a factor of {np.exp(lw.max()-lw.min()):.3e} across sigma.")

    C = np.exp(np.log(anchor["w_edm"]) + anchor["log_R_ce"])
    out = []
    for r in rows:
        w_inv = float(np.exp(np.log(C) - r["log_R_ce"]))
        out.append(dict(sigma=r["sigma"], w_edm=r["w_edm"], w_ce_invariant=w_inv,
                        w_ce_loss_matched=r["w_edm"] * r["R_sm_over_R_ce"]))

    Path = __import__("pathlib").Path
    Path("results/objective").mkdir(parents=True, exist_ok=True)
    json.dump({"sigma_data": SIGMA_DATA, "rows": rows, "w_ce": out},
              open("results/objective/ce_weighting.json", "w"), indent=2)
    print("\nwrote results/objective/ce_weighting.json")


if __name__ == "__main__":
    main()

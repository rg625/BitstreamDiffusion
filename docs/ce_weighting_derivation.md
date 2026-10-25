# Deriving the sigma-weighting for the binary_ce arm

No arbitrary constants. Everything below is either exact algebra or a converged
1-D quadrature (`scripts/analysis/derive_ce_weighting.py`; the reported ratios
are stable to 6 decimals across a 6.7x change of integration window).

## 0. Reference model — why this one

CoBit's matched filter is `mf = (x_t - 0.5)/sigma^2`
(`diffusion/continuous/logit_postprocess.py:190`). That is **exactly** the
posterior log-odds of a single bit under `x0 ~ Bernoulli(1/2)`,
`x_t = x0 + sigma*eps`. The network learns a residual on top of it. So the
independent-bit posterior is not a convenient fiction — it is the architecture's
own built-in prior, and the right place to read off sigma-dependence.

Taking `x0 = 1` (the `x0 = 0` case is the mirror image, and every statistic used
here is symmetric under `lambda -> -lambda`):

```
lambda = (x_t - 1/2)/sigma^2 = 1/(2 sigma^2) + eps/sigma  ~  N(mu, 2 mu),
mu = 1/(2 sigma^2),   q = sigmoid(lambda).
```

Bayes risk per bit of each proper scoring rule:

```
R_sm(sigma) = E[(x0 - q)^2] = E[q(1-q)]     (Brier)
R_ce(sigma) = E[H(q)],  H in nats           (log loss)
```

Both risks vanish super-exponentially as sigma -> 0 (at sigma=0.002 both are
~e^-31250 — the matched filter already resolves the bit), so they must be
computed in log space. Their **ratio** is perfectly well conditioned.

## 1. The three criteria, kept distinct

### (a) Matching loss scale

Choose `w_ce` so that `E[w_ce * l_ce] = E[w_sm * l_sm]` at each sigma. At the
optimum this gives

```
w_ce(sigma) = w_edm(sigma) * R_sm(sigma) / R_ce(sigma)
```

Measured:

| sigma | R_sm/R_ce |
|---|---|
| 0.002 | 0.25000 |
| 0.05 | 0.25240 |
| 0.2 | 0.27539 |
| 0.3998 (=sigma_data) | 0.30493 |
| 1 | 0.34206 |
| 10 | 0.36042 |
| 80 | 0.36067 |

Closed-form limits, both confirmed numerically:
`sigma -> 0` gives **1/4**; `sigma -> inf` gives `1/(4 ln 2) = 0.36067`.

**The ratio spans 0.250 to 0.361 — a total variation of 1.44x across four
decades of sigma.** Anchored at `sigma_data`, that is a relative reweighting of
**0.82x to 1.18x**.

This criterion is also the weakest of the three: matching loss *values* makes
TensorBoard curves comparable and nothing else. Optimisation responds to
gradients.

### (b) Matching gradient scale — rejected on principle

```
dL_ce/d_ell = w (D - x0)
dL_sm/d_ell = w (D - x0) D(1-D)
```

Equal gradient magnitude requires `w_ce = w_sm * D(1-D)` — i.e. multiplying CE
by **precisely the suppression factor the experiment exists to test**. It turns
CE back into SM. This criterion is self-defeating regardless of its magnitude,
and it is rejected for that reason, not because the number is inconvenient.

Worth stating plainly: **the two objectives are *supposed* to have different
gradient scales.** That difference is the hypothesis.

### (c) Preserving sigma-invariance — the right criterion, with a surprise

The design intent of EDM's `lambda(sigma) = 1/c_out^2 = (sigma^2+sd^2)/(sigma^2 sd^2)`
is that no sigma band dominates training. Formally: `w(sigma) * R(sigma)` should
be flat in sigma.

Testing that on the **existing SM recipe**:

| sigma | log(w_edm * R_sm) |
|---|---|
| 0.002 | **-31244.3** |
| 0.05 | -47.5 |
| 0.2 | -1.91 |
| 0.4 | -0.04 |
| 3 | 0.44 |
| 80 | 0.45 |

**The existing recipe is not sigma-invariant — it misses by tens of thousands of
nats.** EDM's `lambda` is derived for continuous Gaussian data, where the
residual scale is exactly `c_out(sigma)`. For *binary* data the Bayes risk
collapses like the bit-error probability `~exp(-1/(8 sigma^2))`, which no
power-law weight can flatten.

## 2. What this actually implies

Under the real training draw (`sigma ~ lognormal(p_mean=-1.2, p_std=1.2)`,
which is what the production config uses — *not* log-uniform):

```
frac(sigma < 0.1)                     = 0.180
share of total weight mass they carry = 0.907
E[w_edm(sigma)]                       = 200.8   (p50 = 17.3, max = 2.5e5)
log R_sm(0.1)                         = -15.3   (there is ~nothing to learn)
```

**18% of draws carry 91% of the weight mass, at sigmas where the Bayes risk is
~2e-7.** While the model is calibrated this is harmless: a huge weight times an
almost-zero residual. Once the model is *wrong* at small sigma, the same weight
amplifies that error by up to 2.5e5.

This is a latent amplifier, and it explains the pilot better than any
objective-specific story does:

- It applies to **both arms**, which matches the observation that **SM diverged
  too** (step 19,540) — an objective-specific explanation cannot account for that.
- CE diverged **first** (step 6,580) because CE has no `D(1-D)` damping between
  the amplified residual and the weights, so the amplification reaches the
  parameters undiminished.

## 3. Recommendation

1. **Do not give CE a bespoke sigma-weighting.** The principled correction is
   within +-18% of `w_edm`, which cannot matter next to a 2.5e5 amplifier, and
   introducing it would add a second variable to a one-variable experiment.
   Keep `loss_weighting = edm` in both arms.

2. **Bound the low-sigma amplification identically in both arms.** This is one
   change, applied symmetrically, so `loss_type` remains the only difference.
   Options, in order of preference:
   - clamp the weight: `w(sigma) <- min(w(sigma), w_max)`, `w_max = 100`
     (the p90 of the current draw is 246, so this is a tail clip, not a
     redesign); or
   - raise the training `sigma_min` from 0.002 to ~0.05, on the grounds that
     `log R_sm(0.05) = -53` means those draws teach nothing anyway.

   Preference is the clamp: it changes no sigma the sampler visits, only how
   hard a mis-prediction there is punished.

3. **This is now a testable prediction, not a preference.** If the amplifier
   story is right, the clamp alone should let *both* arms train past 20k without
   diverging. The 5k smoke test discriminates: if SM still blows up with the
   clamp in place, the diagnosis is wrong and no CE result would be
   interpretable.

# Cross-regime comparison — which guidance findings survive?

Regimes are never pooled. Every entry is a **delta against its own regime's
baseline**, never raw accuracy and never across regimes.

| | Regime A (γ=0) | Regime A (γ=0.3) | **Regime B canonical** |
|---|---|---|---|
| sampler | DDIM | DDIM | DDIM, γ=0.41 *(≡ FKC-EM)* |
| steps | 512 | 256 | 1024 |
| checkpoint | CFG 500k | CFG 500k | CFG 500k |
| problems | 1319, 3 seeds | 1319, 3 seeds | **250, 1 seed** |
| **baseline** | 0.1385 | 0.2568 | **0.3160** |

## Deltas vs the same-regime baseline

| Mechanism | A, γ=0 | A, γ=0.3 | **B canonical** |
|---|---|---|---|
| CFG w=12 | **+0.0811** | −0.0045 n.s. | — |
| CFG w=2 | — | **+0.0190** | +0.0040 n.s. |
| CFG w=1 | — | −0.0010 n.s. | −0.0040 n.s. |
| CFG w=4 | — | +0.0174 | −0.0120 n.s. |
| CFG w=8 | — | — | −0.0400 n.s. |
| CFG w=0.5 | ≈0 (collapse) | — | **−0.2320** |
| AutoGuidance w=15 | **+0.0690** | −0.0169 | **−0.0920** |
| AutoGuidance w=4 | — | — | −0.0400 n.s. |
| SG-prev w=2 | **+0.0498** | −0.2120 | **−0.3160** *(0.0000 absolute)* |
| SG-prev w=0.125 | — | −0.0286 | −0.1600 |
| SG-exact w=1 | −0.0152 | — | **−0.1480** |

**At the canonical operating point, no mechanism helps.** CFG's best cell (w=2)
is +0.0040 with an interval of [−0.0360, +0.0440] — indistinguishable from zero.
Everything else is neutral or significantly harmful.

---

## Classification of every original finding

| # | Original finding | Verdict | Evidence |
|---|---|---|---|
| 1 | CFG improves over single-sample baseline | **REGIME-DEPENDENT** | +0.0811 at γ=0; +0.0040 n.s. at canonical |
| 2 | CFG beats equal-compute majority vote | **REGIME-DEPENDENT** | wins at γ=0 (+0.0325); loses at γ=0.3 (−0.0124). Not tested at canonical |
| 3 | CFG optimal scale ≈12 | **REGIME-DEPENDENT** | w\* moves 12 → 2 → ~1–2 and flattens to nothing as γ rises |
| 4 | AutoGuidance improves | **REVERSED** | +0.0690 at γ=0 → −0.0920 at canonical |
| 5 | SG-prev improves | **REVERSED** | +0.0498 at γ=0 → −0.3160 at canonical (absolute 0.0000) |
| 6 | SG-exact is harmful | **ROBUST** | harmful in every regime tested (−0.0152, −0.1480) |
| 7 | SG-prev is not merely Heun | **NOT TESTED** in B | established in A; Heun not implemented for the stochastic path |
| 8 | CFG+AG fails catastrophically | **NOT TESTED** in B | measured only at γ=0 |
| 9 | Stochasticity improves the baseline | **ROBUST** | +0.1200 in A; canonical baseline 0.3160 vs 0.1385 deterministic |
| 10 | Stochasticity lowers CFG's optimal scale | **ROBUST, extended** | 12 → 2 → ~1–2, and the peak flattens away entirely |
| 11 | Stochasticity makes AG harmful | **ROBUST** | −0.0169 at γ=0.3, −0.0920 at canonical; monotone in γ |
| 12 | Stochasticity makes SG harmful | **ROBUST** | −0.2120 at γ=0.3, −0.3160 at canonical |
| 13 | SG harm is ~linear in scale | **NOT REPRODUCED** | at canonical, 16× the scale gives only 2× the harm (−0.160 → −0.316): strongly sublinear/saturating. **Confounded** — Regime A measured this at 256 steps, Regime B at 1024 |
| 14 | DDIM 512→1024 does nothing at γ=0 | **ROBUST** | +0.0000 [−0.0023,+0.0023] |
| 15 | Heun does not explain SG-prev | **ROBUST within A** | not re-testable in B without implementing Heun+churn |

---

## Reading

The headline is **not** "guidance doesn't work". It is that **guidance's value is
concentrated where the sampler is weak.** Every mechanism was discovered and
tuned at γ=0, where the baseline scores 0.1385. As stochasticity is turned on the
baseline climbs to 0.3160 and each mechanism's contribution shrinks, then
reverses:

* **CFG** degrades gracefully — its optimum slides down (12 → 2 → ~1) and its
  benefit decays to zero. It is never significantly *harmful* at a sane scale.
* **AutoGuidance** and **Self-Guidance** actively reverse, and SG-prev at its
  Regime A optimum destroys the model outright (0.0000).

The one finding that survives everywhere is a **negative** one: SG-exact is
harmful in every regime tested.

## Caveats — deliberately not smoothed over

* Regime B here is **250 problems, one seed**. Intervals are ±0.04–0.05, so the
  CFG cells are genuinely unresolved, not shown to be zero. The *harmful*
  results are large enough to survive this.
* The 250-problem prefix is easier than the full set: baseline 0.3160 here vs
  0.2957 on 1319. Deltas are the comparable quantity, not levels.
* Regime B changes γ **and** step count at once relative to Regime A's γ=0.3
  cells. Finding 13's non-reproduction is confounded by that.
* Nothing here tests combinations, voting, or geometry under the canonical
  regime.

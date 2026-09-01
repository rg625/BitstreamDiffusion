# Open questions

Status vocabulary: **ANSWERED · PARTIALLY ANSWERED · OPEN · REJECTED · INCONCLUSIVE**

| # | Question | Status | Cost so far |
|---|---|---|---|
| Q1 | Why does stochastic churn work? | **OPEN** | — |
| Q2 | Why does SG-prev become systematically wrong under churn? | **PARTIALLY ANSWERED** | 4.1 h |
| Q3 | How does churn change CFG/AG/SG geometry? | **OPEN** (blocked: instrumentation) | — |
| Q4 | Does guidance compose with voting? | **ANSWERED** | 0 h |
| Q5 | Can guidance methods be combined under churn? | **OPEN** | — |
| Q6 | Is temperature an independent control dimension? | **OPEN** | — |
| Q7 | Is optimal guidance scale state/timestep dependent? | **PARTIALLY ANSWERED** | — |
| Q8 | Can dynamic guidance beat the best static policy? | **OPEN** | — |
| Q9 | Can guidance compute be reduced? | **OPEN** | — |
| Q10 | Correct RL formulation for CoBit? | **OPEN** (gated on Q1–Q8) | — |

---

## Q4 — Does guidance compose with voting? · **ANSWERED**

**Answer: yes, additively — and it does not change the compute verdict.**

All at γ=0.3, 1319 problems, 3 seeds, paired. Cost: **zero** — three seeds of
CFG w=2 already existed and `per_problem.answer` was already recorded.

| arm | NFE | samples | exact match |
|---|---|---|---|
| baseline single | 256 | 1 | 0.2568 |
| baseline maj@2 | 512 | 2 | 0.2881 |
| baseline maj@3 | 768 | 3 | 0.3146 |
| CFG w=2 single | 512 | 1 | 0.2757 |
| **CFG w=2 maj@2** | 1024 | 2 | **0.3060** |
| CFG w=2 maj@3 | 1536 | 3 | 0.3389 |

**Composition is additive.** Voting adds +0.0313 [+0.0243,+0.0389] to the
baseline and +0.0303 [+0.0230,+0.0374] to CFG — statistically the same
increment. Guidance and voting neither reinforce nor interfere; their benefits
are independent.

**The verdict depends entirely on what is held fixed** — which is why these must
not be collapsed into one "compute" number:

| held fixed | comparison | result |
|---|---|---|
| **number of samples** (latency-bound) | CFG maj@2 − baseline maj@2 | **+0.0179 [+0.0023,+0.0339]** — guidance wins |
| **forward passes** | CFG maj@2 (1024) − baseline maj@3 (768) | −0.0086 [−0.0265,+0.0088] — inconclusive, and the baseline used 25 % *less* compute |

**Controls.** Paired per problem; same seeds; majority ties broken toward the
first seed (never oracle-assisted); pass@k reported separately as an oracle
bound and never used as the deployable number.
**Caveat.** The exact 1024-vs-1024 comparison needs baseline maj@4, i.e. one
extra baseline seed (~0.15 GPU-h). The trend across maj@2 → maj@3 makes the
outcome fairly predictable but it is not yet measured.
**Implication.** For a latency-bound single-stream deployment, guidance is worth
it. For a throughput-bound one, additional samples remain the better buy.

---

## Q2 — Why does SG-prev fail under churn? · **PARTIALLY ANSWERED**

**Established.** The failure is not magnitude. Scaling down 33× still loses
(−0.0131 / −0.0286 / −0.0551 at w = 0.06 / 0.125 / 0.25), and harm is **linear
in w** — the signature of a systematically wrong direction, not amplified noise.
Two candidate mechanisms are already excluded: the spacing normaliser is
unaffected by churn (`sg_delta_used` median 0.0150 at every γ) and log-σ stays
monotone (0 % increasing steps), so SG is neither mis-normalised nor skipped.

**Still open.** *Why* the direction is wrong. Current hypothesis: churn re-noises
`x_t` each step, so `D_prev` is computed at a strictly noisier state and
`D_cur − D_prev` acquires a consistent backward component. **Untested.**
**Cheapest decisive test.** The counterfactual in §XI — recompute `D_prev` at
the *post-churn* state rather than the stored pre-churn one. If the harm
vanishes, re-noising is confirmed as the cause.

---

## Q7 — Is optimal guidance scale state-dependent? · **PARTIALLY ANSWERED**

Established that it is **regime**-dependent: w\* ≈ 12 at γ=0, w\* ≈ 2–4 at
γ=0.3. Whether it is additionally *timestep* or *state* dependent is untested
and requires the oracle analysis in §XVIII.

---

## Blocked on instrumentation

`Q1C` (answer commitment) and `Q3` (guidance geometry) **cannot be run with the
current logging.** See `docs/research_status.md` for the gap and the fix.

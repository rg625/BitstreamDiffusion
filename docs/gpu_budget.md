# GPU budget — guidance research programme

Total: **~1000 A100-hours**. Living file; update after every array.

Rates measured from completed CSD3 arrays (Ampere A100-80GB, batch 64):

| cell shape | wall clock | GPU-h |
|---|---|---|
| 250 problems, 256 steps, 1 branch | ~4 min | 0.07 |
| 250 problems, 256 steps, 2 branches | ~8 min | 0.13 |
| 1319 problems, 512 steps, 1 branch | ~17 min | 0.28 |
| 1319 problems, 512 steps, 2 branches | ~33 min | 0.55 |

## Ledger

| # | Tranche | Cells | Est. GPU-h | Actual | Status | Decision it enables |
|---|---|---|---|---|---|---|
| 0 | Prior study (PDF) | 242 | — | ~90 | done | the starting point |
| 0b | `solver_control` Heun arm | 6 | 3 | 3.4 | done | **SG-prev is not 2nd-order integration** |
| 1a | `repro` | 4 | 1.7 | — | ready | grading unchanged after the eval-loop edit? |
| 1b | `stoch_screen` | 20 | 2.0 | — | ready | which γ band is worth confirming |

**Spent so far: ~93 h. Committed in tranche 1: ~4 h. Remaining: ~903 h.**

## Staged allocation (guideline, reallocable)

| Phase | Target | Content |
|---|---|---|
| A | 450 h | reproduction, guidance anatomy (CFG/AG/SG), solver analysis, vector geometry |
| B | 250 h | stochasticity × guidance, temperature, sampler interactions |
| C | 150 h | dynamic guidance, trajectory control, adaptive compute |
| D | 150 h | RL / reserve / follow-ups |

**Reallocation flagged.** F4 shows every guidance number in the study was
measured at γ=0, the worst sampler setting available. Phase A's remaining
guidance-anatomy budget should be re-centred on the best stochastic operating
point once `stoch_confirm` fixes it, and Phase B pulled earlier. Characterising
CFG/AG/SG geometry at γ=0 would describe a regime nobody should deploy.


Phase A is already partly paid: the prior study's 90 h plus the solver arm answer
several Phase-A questions outright. Do not spend the Phase-D reserve without
recording here why.

## Rules

* Nothing above ~5 GPU-h is submitted without a hypothesis, a cost estimate and
  a stated branch condition in `docs/research_protocol.md`.
* Screening runs at 250 problems / 1 seed; confirmation at 1319 / ≥3 seeds.
  Never confirm what screening has not first localised.
* Every cell is idempotent, so a failed array is resubmitted, never re-planned.

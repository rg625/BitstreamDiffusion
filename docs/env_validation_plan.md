# Environment validation: is Python 3.9 the cause of the divergences?

## The claim under test

Seven training runs in this environment diverged before ~12k steps — including
the production-era commit `036a2b5` itself (break at 11,278). The same code ran
**500,000 steps with zero excursions** in the collaborator's environment.

Two things prove that environment differed, neither of which is disputable:

1. The historical code imports `mauve` at module scope, so `import
   trainers.trainer` fails without it. Production ran that code; **their env had
   the package, ours does not.**
2. `diffusion/discrete/losses.py` has used PEP 604 (`torch.Tensor | None`) in a
   function signature **since the initial release (2026-05-10)**, which cannot
   be imported on Python 3.9. `from __future__ import annotations` was not added
   until 2026-09-05. **Production therefore ran on Python >= 3.10; we run 3.9.**

A different interpreter means a different torch wheel and different compiled
kernels even at an identical version string (`2.8.0+cu128`, installed
2026-03-18, i.e. before production — so the *version* never changed).

## The experiment

Build `sedd310` (Python 3.10, torch 2.8.0+cu128) alongside the existing `sedd`,
which is left untouched so the running jobs are unaffected. Then rerun the
**exact configuration that broke**, changing only the interpreter:

- `configs/tasks/tinygsm_bits_objective.py`, `OBJ_LOSS=binary_sm`
- production-matched: `cond.p_uncond=0.1`, clamp OFF, `loss_weighting=edm`
- seed 42 — the same seed whose py3.9 run broke at **6,356**
- 20,000 steps, divergence guard armed at factor 10
- identical corpus, identical code (HEAD)

**One variable: the Python environment.**

## Reading the result

| outcome | conclusion |
|---|---|
| **survives 20k** | The interpreter/toolchain is the cause. Every stability number in this study is environment-specific and must be remeasured — including CE's 1.80x advantage, which would then be an artefact of a broken stack rather than a property of the objectives. Branch B becomes answerable for the first time. |
| **breaks again** | The interpreter is exonerated. The remaining candidates are the collaborator's uncommitted working tree (their directory is permission-denied to us) or something in the corpus rebuild that the `sigma_data` fingerprint match to ~8 s.f. did not catch. |

20k comfortably exceeds every observed break (max 11,278), so a survival is
informative rather than lucky. Budget: ~9.8 GPU-h, ~2.7 h wall on 4 GPUs.

**Pre-registered caveat.** One seed. A survival at 20k is strong but not proof;
the py3.9 breaks ranged 4,481–11,278, so a single 20k survival is roughly a
1-in-8 coincidence under the null. If it survives, the honest next step is two
more seeds before rewriting any conclusions.

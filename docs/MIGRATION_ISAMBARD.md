# Moving CoBit to Isambard-AI (GH200)

Written for: whoever runs the migration — assumes the CoBit project but not the
Isambard account.

Everything below is ordered so that nothing expensive happens before the cheap
checks have passed. The grant is 10,000 GPU-hours; the four gates in step 4 cost
about two of them and are the difference between a result and a confidently
wrong number.

---

## 1. The one thing that does not transfer

**Isambard-AI is aarch64 (NVIDIA Grace) with Hopper GPUs. CSD3 is x86_64 with
Ampere.** The `sedd310` conda environment is x86_64 with `torch 2.8.0+cu128`
compiled for sm_80. Every compiled wheel in it is the wrong architecture.
Copying it yields an environment that imports cleanly and then fails at the
first CUDA call — the slowest possible way to discover the problem.

The environment gets **rebuilt**, not copied: `scripts/site/build_env_isambard.sh`.

| | CSD3 | Isambard-AI |
|---|---|---|
| CPU | x86_64 | aarch64 (Grace) |
| GPU | A100 80 GB, sm_80 | GH200 96 GB, sm_90 |
| torch wheel | cu128 x86_64 | cu128 **aarch64** (versioned index, not PyPI) |
| GPUs/node | 4 | 4 |

The one trap inside the rebuild: on aarch64, `pip install torch` from PyPI gives
a **CPU-only** build. It must come from `--index-url
https://download.pytorch.org/whl/cu128`. The build script's verification catches
this, which is why it runs before anything else.

---

## 2. Code

```bash
git clone git@github.com:rg625/BitstreamDiffusion.git
cd BitstreamDiffusion && git checkout tasks/fkc-temperature
```

Everything through commit `d3cc81c` is on that branch. `datasets/`, `runs/` and
`tinigsm_gsm8k/` are gitignored — they are step 3.

## 3. Data and checkpoints (24.5 GB)

```bash
# on CSD3
bash scripts/site/transfer.sh                                   # dry run, lists what and why
bash scripts/site/transfer.sh --dest user@isambard:/path/BitstreamDiffusion --go

# on Isambard, before using any of it
sha256sum -c scripts/site/transfer_checksums.txt
```

rsync is resumable; re-run the same command after an interruption. The payload
is the tokenised corpus (11.2 GB — rebuilding it needs the HuggingFace download
and about two hours of tokenising), the production anchor, the four ordering
arms' `last.pt`, the in-flight token run, and `results/`.

`sigma_data.json` travels with every checkpoint deliberately. Without it the
evaluator falls back to a config default of 0.5 against a true 0.3998, which
does not crash — it just quietly produces the wrong numbers.

## 4. Four gates before spending grant hours

| # | Gate | Command | Pass condition |
|---|------|---------|----------------|
| 1 | Environment | `bash scripts/site/build_env_isambard.sh` | cuda True, sm_90, bf16 supported |
| 2 | Test suite | `PYTHONPATH=$PWD python -m pytest tests/ -q` | 464 passed |
| 3 | **Decode path** | evaluate the production anchor, 3 seeds, karras, 256 steps | accuracy **0.12–0.20** |
| 4 | Distributed training | `submit.sh scripts/hpc/arch/deadlock_repro.slurm` | 400 steps, 8 epoch boundaries, `rc=0` |

Gate 3 is the one that matters. A sampler bug on CSD3 once returned
probabilities from σ_{N−1} instead of a final denoise and scored **0.000 against
a true 0.164**; it was caught only because an anchor was in the batch. On a new
architecture, with a different BLAS, a different attention kernel and a
different reduction order, that class of failure is *more* likely, not less.
`experiments/ordering/analyze_train.py` refuses to present a clean table without
an in-band anchor, so run it through that rather than reading the raw JSON.

Gate 4 also tells you something new: the epoch-boundary deadlock that costs a
restart every 5 epochs on CSD3 may or may not reproduce on a different NCCL and
interconnect. Either answer is useful.

## 5. The rule that governs what you may compare

**Isambard results cannot be pooled with CSD3 results in a paired comparison.**
Different architecture, different libraries, different reduction order; the
project's standing rule is that regimes are never pooled, and this is a regime
change in every sense. Concretely:

- `control` and `l2r` finished on **CSD3**. If `r2l` and `random` finish on
  **Isambard**, the four-arm comparison is cross-machine and the deltas are not
  interpretable. They are at 91% and need ~61 GPU-hours — **finish them on
  CSD3**, where their own control already lives.
- The V-way token run's matched control is `ord_fs500k_none_s42`, also CSD3.
  Continuing that run on Isambard would compare a half-Isambard arm against a
  CSD3 control.

The clean way to use the grant on the logits question is to run **both arms from
scratch on Isambard** — the V-way token model and its binary control, matched in
steps, batch, LR and seed, both on GH200. Roughly 535 GPU-hours at an assumed
2× speedup over A100, about 5% of the grant, and it answers the question without
a cross-machine confound anywhere in it.

## 6. Settings that should change on GH200

- **`TOK_ACCUM` can probably drop.** Gradient accumulation exists only because
  the V-way model OOMs at 128 examples/GPU on an 80 GB A100. With 96 GB, batch
  512 may fit at accum 2 or even 1, which removes 3 of every 4 backward passes'
  worth of redundant all-reduces. Re-smoke with
  `TOK_STEPS=300 TOK_TAG=smoke_gh` before trusting it; do not assume.
- **Wall limits differ.** `COBIT_WALL_MAX` in `scripts/site/site.sh` is set to
  24 h as a guess. Correct it from the grant documentation — `submit.sh` refuses
  an over-cap request rather than letting SLURM silently truncate it.
- **`COBIT_ACCOUNT` and `COBIT_PARTITION` are not guessable.** They come from
  the grant email. `site.sh` refuses to run with an unset account, because a
  wrong one is a job that pends forever without saying why.

## 7. Submitting

Every launcher carries CSD3 `#SBATCH` directives. Do not edit them — submit
through the wrapper, which passes this site's flags on the command line, where
SLURM's precedence puts them above the in-script directives:

```bash
export COBIT_SITE=isambard COBIT_ACCOUNT=<project> COBIT_PYTHON=<env>/bin/python
bash scripts/site/submit.sh scripts/hpc/arch/token_train.slurm \
     --time=24:00:00 --export=ALL,TOK_BATCH=512,TOK_ACCUM=2,TOK_LR=1e-4,TOK_TAG=gh
```

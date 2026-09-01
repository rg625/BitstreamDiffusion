#experiments/guidance/grids.py
"""Declarative experiment grids for the guidance study.

One place defines every sweep, so a SLURM array task is just "cell N of grid
G". That gives three things the ad-hoc per-sweep scripts in `scripts/tasks/`
could not:

  * the grid can be enumerated and reviewed *locally* before anything is
    submitted (`python -m experiments.guidance.grids show cfg_coarse`);
  * array size and cell identity are derived from the same source the job
    reads, so an off-by-one cannot silently shift the whole sweep;
  * every cell carries a stable `name`, which becomes the result filename tag,
    so reruns are idempotent and results never collide.

Staging follows the plan: coarse exploration on a reduced evaluation subset
first, then refinement around whatever the coarse pass finds, then confirmation
at full size with several seeds. Nothing here launches anything.
"""
from __future__ import annotations

import argparse
import itertools
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# -----------------------------------------------------------------------------
# Defaults for the TinyGSM -> GSM8K testbed
# -----------------------------------------------------------------------------
# This is the only CoBit run that is conditional (cfg.cond.enabled) AND trained
# with conditioning dropout (p_uncond=0.1), so it is the only checkpoint family
# on which classifier-free guidance is even defined.
CONFIG = "configs/tasks/tinygsm_bits_cfg.py"
RUN_DIR = "tinigsm_gsm8k/runs/cobit_raw_binary_bits_cfg"
# The CFG run trained to 500k; `last.pt` (global_step=500000) is the strongest
# model and therefore AutoGuidance's "good". Verified by loading each file.
GOOD_CKPT = f"{RUN_DIR}/checkpoints/last.pt"

# Candidate "bad" models for AutoGuidance, weakest first. Only those present on
# disk are used; `available_bad_checkpoints()` filters at build time so a sweep
# never silently evaluates a missing file.
BAD_CKPT_CANDIDATES = [
    f"{RUN_DIR}/checkpoints/step=000025000.pt",
    f"{RUN_DIR}/checkpoints/step=000050000.pt",
    f"{RUN_DIR}/checkpoints/step=000100000.pt",
    f"{RUN_DIR}/checkpoints/step=000150000.pt",
    f"{RUN_DIR}/checkpoints/step=000200000.pt",
    f"{RUN_DIR}/checkpoints/step=000250000.pt",
    f"{RUN_DIR}/checkpoints/step=000350000.pt",
    f"{RUN_DIR}/checkpoints/step=000425000.pt",
]

# Exploration runs on a fixed 250-problem prefix of the GSM8K test set; the
# confirmation stage uses all 1319. Using a *prefix* (not a random subset) keeps
# every method on identical problems, which is what makes the comparisons paired.
EXPLORE_LIMIT = 250
FULL_LIMIT = 1319

# Sampler defaults. `deterministic` (gamma=0) is the primary exploration regime:
# the pre-existing 250k sweep showed CFG's headroom there is large (9.3% -> 14%),
# whereas EDM churn already recovers much of it, which would mask guidance
# effects. The churn interaction is a separate, later study (Phase 15).
DET = dict(sampler="deterministic", gamma=0.0)
STOCH = dict(sampler="stochastic", gamma=0.41)


@dataclass
class Cell:
    """One evaluation run."""
    name: str
    checkpoint: str = GOOD_CKPT
    config: str = CONFIG
    sampler: str = "deterministic"
    sampler_kind: str = "ddim"
    schedule: str = "entropic"
    gamma: float = 0.0
    steps: int = 256
    limit: int = EXPLORE_LIMIT
    seed: int = 42
    batch_size: int = 64
    # guidance
    guidance_scale: float = 0.0
    ag_scale: float = 0.0
    bad_checkpoint: Optional[str] = None
    bad_ema: int = 1
    sg_scale: float = 0.0
    sg_variant: str = "prev"
    sg_delta: float = 0.5
    sg_mf_mode: str = "hold"
    collect_diagnostics: bool = True
    extra: Dict[str, str] = field(default_factory=dict)

    def cli(self, out_dir: str) -> List[str]:
        """The exact argv this cell runs."""
        a = [
            "--config", self.config,
            "--checkpoint", self.checkpoint,
            "--sampler", self.sampler,
            "--sampler_kind", self.sampler_kind,
            "--schedule", self.schedule,
            "--gamma", str(self.gamma),
            "--steps", str(self.steps),
            "--limit", str(self.limit),
            "--seed", str(self.seed),
            "--batch_size", str(self.batch_size),
            "--guidance_scale", str(self.guidance_scale),
            "--out_dir", out_dir,
            "--tag", self.name,
        ]
        if self.ag_scale > 0:
            a += ["--ag_scale", str(self.ag_scale),
                  "--bad_checkpoint", str(self.bad_checkpoint),
                  "--bad_ema", str(self.bad_ema)]
        if self.sg_scale != 0:
            a += ["--sg_scale", str(self.sg_scale),
                  "--sg_variant", self.sg_variant,
                  "--sg_delta", str(self.sg_delta),
                  "--sg_mf_mode", self.sg_mf_mode]
        if self.collect_diagnostics:
            a += ["--collect_diagnostics"]
        for k, v in self.extra.items():
            a += [f"--{k}", str(v)]
        return a


def _step_of(ckpt: str) -> str:
    return "".join(ch for ch in ckpt.rsplit("/", 1)[-1] if ch.isdigit()) or "na"


def available_bad_checkpoints(root: str = ".") -> List[str]:
    """Bad-model candidates that actually exist on disk, weakest first."""
    from pathlib import Path
    return [c for c in BAD_CKPT_CANDIDATES if (Path(root) / c).exists()]


# -----------------------------------------------------------------------------
# Grids
# -----------------------------------------------------------------------------

def grid_cfg_coarse() -> List[Cell]:
    """Phase 10: coarse CFG scale sweep. Deliberately spans past the expected
    optimum so the degradation onset is measured, not assumed."""
    ws = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0]
    return [Cell(name=f"cfg_w{w:g}", guidance_scale=w, **DET) for w in ws]


def grid_cfg_high() -> List[Cell]:
    """Extension of the coarse CFG sweep past w=7.

    The coarse pass rose monotonically to w=7 (0.164 -> 0.224 at n=250,
    NFE=256) without turning over, so the optimum is not yet bracketed and no
    "best scale" can be claimed. This extends the range until accuracy actually
    degrades. Note the earlier 250k-checkpoint sweep peaked near w=6 and
    declined by w=12, so the turnover is expected somewhere in here -- but the
    500k model may behave differently, which is the point of measuring.
    """
    ws = [8.0, 10.0, 12.0, 15.0, 20.0, 30.0]
    return [Cell(name=f"cfghigh_w{w:g}", guidance_scale=w, **DET) for w in ws]


def grid_cfg_stochastic() -> List[Cell]:
    """The same sweep under EDM churn, to separate 'guidance helps' from
    'guidance substitutes for stochasticity'."""
    ws = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    return [Cell(name=f"cfgstoch_w{w:g}", guidance_scale=w, **STOCH) for w in ws]


def grid_cfg_confirm() -> List[Cell]:
    """Phase 16: confirm the CFG finding at full evaluation size, several seeds.

    The exploratory sweep (n=250, NFE=256) put the optimum at w~20 with a broad
    flat top from w~7 upward. This promotes a shortlist spanning that plateau,
    plus the baseline and one clearly sub-optimal point, to the full 1319-problem
    test set at NFE=512 across three seeds.

    Scales are held to the shortlist rather than re-swept: re-optimising on the
    confirmation set would be fitting the scale to the data it is then reported
    on. Seeds vary the initial noise; within a seed every method sees identical
    problems AND identical noise, so the comparison stays paired on both.
    """
    out = []
    for w in (0.0, 4.0, 7.0, 12.0, 20.0):
        for seed in (42, 43, 44):
            out.append(Cell(name=f"cfgconf_w{w:g}_s{seed}", guidance_scale=w,
                            steps=512, limit=FULL_LIMIT, seed=seed, **DET))
    return out


def grid_sg_confirm() -> List[Cell]:
    """Phase 16: confirm the SG-prev finding at full evaluation size.

    The exploratory grid put SG-prev at w=1 between +0.028 and +0.036 across
    NFE 64..512 (significant at 3 of 4, same sign and magnitude at all 4) at
    ZERO extra model evaluations, while SG-exact was consistently negative at
    2x the cost. That is the study's most useful claim, so it gets the same
    full-size treatment as CFG.

    SG-exact is carried along at the same scale, not dropped: a confirmation
    that only re-runs the winner cannot distinguish "SG-prev works" from
    "this evaluation set happens to favour it".
    """
    out = []
    for variant, w in (("prev", 0.0), ("prev", 1.0), ("prev", 2.0), ("exact", 1.0)):
        for seed in (42, 43, 44):
            tag = "base" if w == 0 else f"{variant}{w:g}"
            out.append(Cell(name=f"sgconf_{tag}_s{seed}", sg_scale=w, sg_variant=variant,
                            steps=512, limit=FULL_LIMIT, seed=seed, **DET))
    return out


def grid_ag(root: str = ".") -> List[Cell]:
    """Phase 11: the badness x scale surface.

    Both axes matter: Karras et al.'s claim is that there is an *optimal*
    degree of badness, so the earliest checkpoint is not assumed best.
    """
    ws = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0]
    out = []
    for bad in available_bad_checkpoints(root):
        for w in ws:
            if w == 0.0:
                continue                     # w=0 is the shared baseline cell
            out.append(Cell(name=f"ag_b{_step_of(bad)}_w{w:g}",
                            ag_scale=w, bad_checkpoint=bad, **DET))
    out.insert(0, Cell(name="ag_baseline", **DET))
    return out


def grid_ag_high(root: str = ".") -> List[Cell]:
    """Extension of the AutoGuidance scale range, mirroring cfg_high.

    The coarse AG surface was still rising at w_ag=5 for every bad checkpoint,
    so the scale is not bracketed and no optimum can be claimed -- the same
    situation the CFG sweep was in before cfg_high, where the true optimum
    turned out to be w~20, far outside the initial range. Nothing about
    AutoGuidance says its useful scale should be smaller, so it gets the same
    treatment rather than being written off at w=5.

    Run for every available bad checkpoint: the badness axis is the point, and
    the coarse pass could not separate them (all paired CIs covered zero).
    """
    ws = [6.0, 8.0, 10.0, 15.0, 20.0]
    out = []
    for bad in available_bad_checkpoints(root):
        for w in ws:
            out.append(Cell(name=f"aghigh_b{_step_of(bad)}_w{w:g}",
                            ag_scale=w, bad_checkpoint=bad, **DET))
    return out


def grid_ag_confirm(root: str = ".") -> List[Cell]:
    """Phase 16: resolve whether AutoGuidance actually helps.

    The extended sweep produced two individually significant cells (bad=350k
    w=15, +0.056, p=0.029; bad=250k w=6, +0.044, p=0.039). That is NOT the same
    standard of evidence as CFG or SG-prev: across ~30 AG cells, two hits at
    p<0.05 is roughly the chance expectation, whereas CFG showed a large effect
    consistent across a dozen scales and SG-prev the same effect at four
    independent NFE settings. AG's evidence rests on isolated cells and could
    easily be selection.

    So the two candidates are re-measured on the full test set across three
    seeds, against a shared baseline. The scales are fixed from the exploratory
    stage and not re-tuned here.
    """
    bads = {_step_of(b): b for b in available_bad_checkpoints(root)}
    out = []
    for seed in (42, 43, 44):
        out.append(Cell(name=f"agconf_base_s{seed}", steps=512,
                        limit=FULL_LIMIT, seed=seed, **DET))
    for step, w in (("000350000", 15.0), ("000250000", 6.0)):
        if step not in bads:
            continue
        for seed in (42, 43, 44):
            out.append(Cell(name=f"agconf_b{step}_w{w:g}_s{seed}", ag_scale=w,
                            bad_checkpoint=bads[step], steps=512,
                            limit=FULL_LIMIT, seed=seed, **DET))
    return out


def grid_sg() -> List[Cell]:
    """Phase 12: SG-prev vs SG-exact across strength and NFE.

    NFE is swept because the two variants should converge as the grid refines
    (the cached prediction's noise-level offset shrinks), and because SG-prev's
    selling point is a gain at *zero* extra evaluations.
    """
    out = []
    for nfe in (64, 128, 256, 512):
        out.append(Cell(name=f"sg_base_n{nfe}", steps=nfe, **DET))
        for variant in ("prev", "exact"):
            for w in (0.25, 0.5, 1.0, 2.0):
                out.append(Cell(name=f"sg_{variant}_w{w:g}_n{nfe}", steps=nfe,
                                sg_scale=w, sg_variant=variant, **DET))
    return out


def grid_sg_delta() -> List[Cell]:
    """How sensitive is self-guidance to the log-sigma offset itself?"""
    out = []
    for d in (0.1, 0.25, 0.5, 1.0, 2.0):
        for variant in ("prev", "exact"):
            out.append(Cell(name=f"sgdelta_{variant}_d{d:g}", sg_scale=1.0,
                            sg_variant=variant, sg_delta=d, **DET))
    return out


def grid_sg_mf() -> List[Cell]:
    """Ablation of the matched-filter decision: does holding the analytic term
    at the true sigma actually matter in practice, or only in principle?"""
    out = []
    for mode in ("hold", "vary"):
        for w in (0.5, 1.0, 2.0):
            out.append(Cell(name=f"sgmf_{mode}_w{w:g}", sg_scale=w,
                            sg_variant="exact", sg_mf_mode=mode, **DET))
    return out


def grid_factorial(cfg_w: float, ag_w: float, sg_w: float,
                   bad: Optional[str] = None) -> List[Cell]:
    """Phase 13: the 12-cell factorial at one chosen operating point per axis.

    Run only AFTER the single-axis sweeps pick the per-axis optimum, so the
    interaction estimates sit where each method is actually useful.
    """
    out = []
    for use_cfg, use_ag, sg in itertools.product([False, True], [False, True],
                                                 [None, "prev", "exact"]):
        parts = []
        if use_cfg:
            parts.append("cfg")
        if use_ag:
            parts.append("ag")
        if sg:
            parts.append(f"sg{sg}")
        name = "fact_" + ("baseline" if not parts else "_".join(parts))
        out.append(Cell(
            name=name,
            guidance_scale=cfg_w if use_cfg else 0.0,
            ag_scale=ag_w if use_ag else 0.0,
            bad_checkpoint=bad if use_ag else None,
            sg_scale=sg_w if sg else 0.0,
            sg_variant=sg or "prev",
            **DET,
        ))
    return out


def grid_nfe(cfg_w: float, ag_w: float, sg_w: float,
             bad: Optional[str] = None) -> List[Cell]:
    """Phase 14: quality vs compute. Guidance methods cost different numbers of
    model evaluations per step, so a fixed-NFE comparison is not a fixed-compute
    comparison; sweeping the step count is what makes the Pareto front visible."""
    out = []
    for nfe in (8, 16, 32, 64, 128, 256, 512):
        out.append(Cell(name=f"nfe_baseline_n{nfe}", steps=nfe, **DET))
        out.append(Cell(name=f"nfe_cfg_n{nfe}", steps=nfe, guidance_scale=cfg_w, **DET))
        out.append(Cell(name=f"nfe_sgprev_n{nfe}", steps=nfe, sg_scale=sg_w,
                        sg_variant="prev", **DET))
        if bad:
            out.append(Cell(name=f"nfe_ag_n{nfe}", steps=nfe, ag_scale=ag_w,
                            bad_checkpoint=bad, **DET))
    return out


def grid_confirm(cells: List[Cell], seeds=(42, 43, 44, 45, 46)) -> List[Cell]:
    """Phase 16: promote a shortlist to full evaluation size across seeds.

    Same problems, same initial-noise seeds across methods, so every comparison
    stays paired and the seed spread is a real uncertainty estimate rather than
    a selection opportunity.
    """
    out = []
    for c in cells:
        for s in seeds:
            d = asdict(c)
            d.pop("extra", None)
            d["name"] = f"{c.name}_seed{s}"
            d["seed"] = s
            d["limit"] = FULL_LIMIT
            d["steps"] = max(c.steps, 512)
            out.append(Cell(extra=dict(c.extra), **d))
    return out


def _require_operating_point(root: str = ".") -> Dict[str, object]:
    """As `_operating_point`, but refuses to fall back to the defaults.

    The confirmation-grade factorial is ~18 GPU-hours. Silently building it at
    the placeholder operating point (w=3 / 1.5 / 0.5, bad=the first checkpoint
    on disk) because a GUID_* variable was missing from --export would burn all
    of it on the wrong grid, and the result JSONs would look perfectly normal
    afterwards. The single-axis confirmations have chosen these values, so at
    this stage an unset variable is a mistake, not a request for a default.
    """
    missing = [k for k in ("GUID_CFG_W", "GUID_AG_W", "GUID_SG_W", "GUID_BAD_STEP")
               if not os.environ.get(k)]
    if missing:
        raise SystemExit(
            "factorial_confirm needs an explicit operating point; missing: "
            + ", ".join(missing)
            + "\nThe confirmations chose: GUID_CFG_W=12 GUID_AG_W=15 "
              "GUID_SG_W=2 GUID_BAD_STEP=350000\n"
              "Pass them in --export too, not just your shell "
              "(see scripts/hpc/guidance/RUNBOOK.md section 6)."
        )
    return _operating_point(root)


def _operating_point(root: str = ".") -> Dict[str, object]:
    """Per-axis operating points for the factorial and NFE grids.

    These are only meaningful AFTER the single-axis sweeps have found each
    method's useful setting, so they are read from the environment rather than
    hard-coded: the sweep results choose them, not this file.

        GUID_CFG_W=3 GUID_AG_W=1.5 GUID_SG_W=0.5 \
        GUID_BAD_STEP=250000 python -m experiments.guidance.grids show factorial
    """
    bad_step = os.environ.get("GUID_BAD_STEP")
    bad = None
    if bad_step:
        cand = f"{RUN_DIR}/checkpoints/step={int(bad_step):09d}.pt"
        bad = cand if (Path(root) / cand).exists() else None
        if bad is None:
            raise SystemExit(f"GUID_BAD_STEP={bad_step} -> {cand} does not exist")
    elif available_bad_checkpoints(root):
        bad = available_bad_checkpoints(root)[0]
    return {
        "cfg_w": float(os.environ.get("GUID_CFG_W", 3.0)),
        "ag_w": float(os.environ.get("GUID_AG_W", 1.5)),
        "sg_w": float(os.environ.get("GUID_SG_W", 0.5)),
        "bad": bad,
    }


def grid_repro() -> List[Cell]:
    """Section 4: reproduce the PDF's headline cells with the CURRENT code.

    Mandatory before extending, and not a formality here: the evaluation loop
    changed after those numbers were produced. `evaluate_samples(text, gold)`
    was replaced by `predict_answer` + `_numbers_equal` so the executed answer
    could be recorded for maj@k. That composition is what evaluate_samples
    already was, so grading *should* be bit-identical -- but "should" is
    exactly what a reproduction run is for, and a silent change in grading
    would invalidate every comparison in the study.

    Same checkpoint, steps, limit and seed as the confirmation cells, written
    to a separate out_dir so the per-problem vectors can be diffed against the
    originals element by element rather than only in aggregate.
    """
    base = dict(DET)
    base.update(steps=512, limit=FULL_LIMIT, seed=42)
    return [
        Cell(name="repro_baseline_s42", **base),
        Cell(name="repro_cfg12_s42", guidance_scale=12.0, **base),
        Cell(name="repro_sgprev2_s42", sg_scale=2.0, sg_variant="prev", **base),
        Cell(name="repro_ag15_s42", ag_scale=15.0,
             bad_checkpoint=f"{RUN_DIR}/checkpoints/step=000350000.pt", **base),
    ]


def grid_stoch_screen() -> List[Cell]:
    """Section 12: the axis the PDF never touched -- guidance under churn.

    Every number in the study is deterministic (gamma=0), so "guidance helps"
    and "guidance substitutes for the stochasticity we never used" are not yet
    separable. This is a coarse screen, not a confirmation: 250 problems, one
    seed, 256 steps, to find which gamma region is worth spending on.

    gamma enters as s_churn = gamma*(NFE-1) and DDIM caps it at sqrt(2)-1, so
    0.41 is the top of the usable band, not an arbitrary endpoint.

    One confound is already instrumented: churn can push sigma back UP between
    steps, which makes SG-prev's backward derivative invalid, so it is skipped
    on those steps and the run reports sg_skipped. Read that before reading the
    SG row -- a null result there may mean "SG never fired", not "SG failed".
    """
    out = []
    for gamma in (0.0, 0.1, 0.2, 0.3, 0.41):
        mode = "deterministic" if gamma == 0.0 else "stochastic"
        common = dict(sampler=mode, gamma=gamma, steps=256, limit=EXPLORE_LIMIT)
        g = f"{gamma:g}".replace(".", "p")
        out += [
            Cell(name=f"stoch_g{g}_base", **common),
            Cell(name=f"stoch_g{g}_cfg12", guidance_scale=12.0, **common),
            Cell(name=f"stoch_g{g}_ag15", ag_scale=15.0,
                 bad_checkpoint=f"{RUN_DIR}/checkpoints/step=000350000.pt", **common),
            Cell(name=f"stoch_g{g}_sgprev2", sg_scale=2.0, sg_variant="prev", **common),
        ]
    return out


def grid_stoch_confirm() -> List[Cell]:
    """Confirm the screen's two large effects at full size.

    The 250-problem screen found two things that dwarf everything in the PDF:
    churn alone takes the baseline 0.164 -> 0.276 at zero extra NFE, which is
    larger than any guidance effect ever measured here; and SG-prev *inverts*
    under churn (+0.028 -> -0.244). Both are far outside the +/-0.04 screening
    noise, but the study's headline recommendation now depends on them, so they
    get 1319 problems and three seeds.

    Also decides the interaction the PDF could not: guidance's margin over the
    SAME-gamma baseline shrinks as gamma rises (CFG +0.064 -> ~+0.01), which is
    what "guidance was substituting for stochasticity we never used" predicts.
    Three gammas is enough to test monotonicity without paying for a surface.

    256 steps, not 512: the screen ran there, and at 1319 problems this keeps
    the whole grid near 8 GPU-h.
    """
    out = []
    for gamma in (0.0, 0.2, 0.3):
        mode = "deterministic" if gamma == 0.0 else "stochastic"
        common = dict(sampler=mode, gamma=gamma, steps=256, limit=FULL_LIMIT)
        g = f"{gamma:g}".replace(".", "p")
        arms = [
            (f"stc_g{g}_base", {}),
            (f"stc_g{g}_cfg12", dict(guidance_scale=12.0)),
            (f"stc_g{g}_ag15", dict(ag_scale=15.0,
                bad_checkpoint=f"{RUN_DIR}/checkpoints/step=000350000.pt")),
            (f"stc_g{g}_sgprev2", dict(sg_scale=2.0, sg_variant="prev")),
        ]
        for name, kw in arms:
            for seed in (42, 43, 44):
                out.append(Cell(name=f"{name}_s{seed}", seed=seed, **kw, **common))
    return out


def grid_churn_anatomy() -> List[Cell]:
    """Two questions left open by stoch_confirm, at gamma=0.3.

    ARM 1 -- does CFG pay at ANY scale under churn?
    stoch_confirm shows CFG w=12 is n.s. against the same-gamma baseline once
    churn is on. But w=12 was tuned at gamma=0, so "guidance stops helping" and
    "guidance is mis-tuned for this regime" are still confounded. Sweeping the
    scale *at the good gamma* separates them. If no scale beats the plain
    baseline, guidance is genuinely superseded here; if a small scale wins, the
    story is re-tuning, not redundancy.

    ARM 2 -- is SG-prev's collapse the noise amplification F6 identified?
    sg_dir_rms grows ~16x from gamma=0 to 0.41, so if the direction is right and
    only its magnitude is wrong, dividing the scale by roughly that factor
    should restore it. w=2/16 ~ 0.125, bracketed either side. A flat null across
    all three would falsify the magnitude explanation and point at the direction
    itself being corrupted by injected noise.
    """
    G = 0.3
    common = dict(sampler="stochastic", gamma=G, steps=256, limit=FULL_LIMIT)
    out = []
    for w in (1.0, 2.0, 4.0, 7.0):
        for seed in (42, 43, 44):
            out.append(Cell(name=f"chn_cfg{w:g}_s{seed}", guidance_scale=w,
                            seed=seed, **common))
    for w in (0.06, 0.125, 0.25):
        for seed in (42, 43, 44):
            out.append(Cell(name=f"chn_sg{w:g}_s{seed}", sg_scale=w,
                            sg_variant="prev", seed=seed, **common))
    return out


def grid_solver_control() -> List[Cell]:
    """Is SG-prev guidance, or just a better ODE solver?

    SG-prev's direction is (delta_ref / realised_log_sigma_spacing) * (D_cur -
    D_prev): a consecutive-step derivative estimate, amplified ~96x at 512
    steps, applied along the trajectory. That is structurally what a 2nd-order
    solver or a momentum term does, built from the same two quantities -- so
    "Self-Guidance helps bitstream diffusion" and "this model was under-served
    by a 1st-order solver" predict the same +4.8 points.

    The discriminating comparison is at MATCHED model evaluations:
        DDIM  512 steps            =  512 NFE   (baseline)
        DDIM  512 steps + SG-prev  =  512 NFE   (the claim)
        Heun  256 steps            =  512 NFE   (2 evals/step: the rival)
    plus DDIM 1024 (1024 NFE) to separate "better direction" from "finer grid",
    and Heun 512 (1024 NFE) as the 2nd-order arm at CFG's budget.

    If Heun-256 matches SG-prev at 512 NFE, the effect is a solver artefact and
    should be reported as one.
    """
    out = []
    for name, kind, steps in (("solv_ddim512", "ddim", 512),
                              ("solv_sgprev512", "ddim", 512),
                              ("solv_ddim1024", "ddim", 1024),
                              ("solv_heun256", "heun", 256),
                              ("solv_heun512", "heun", 512)):
        c = dict(DET)
        c.update(steps=steps, sampler_kind=kind, limit=FULL_LIMIT)
        if name == "solv_sgprev512":
            c.update(sg_scale=2.0, sg_variant="prev")
        # Heun has no GuidedDenoiser path, so it cannot emit the trace.
        out.append(Cell(name=name, collect_diagnostics=(kind == "ddim"), **c))
    return [c for cell in out for c in _seeds(cell, (42, 43, 44))]


def grid_compute_control() -> List[Cell]:
    """Deployable multi-sample baselines at guidance's compute.

    pass@2 is computable offline from the seeds already run, but it needs an
    oracle to pick the right sample. maj@k is what someone could actually ship,
    and it needs the executed answers -- now recorded per problem. Three extra
    baseline seeds give maj@3/maj@5 headroom against CFG's 2x and a 4x arm.
    """
    base = dict(DET)
    base.update(steps=512, limit=FULL_LIMIT)
    cells = [Cell(name="cmp_baseline", **base)]
    return [c for cell in cells for c in _seeds(cell, (45, 46, 47, 48))]


def grid_null_ablation() -> List[Cell]:
    """How much does CFG depend on the null the model was TRAINED with?

    The run trained its unconditional branch with null_strategy="half" (0.5
    bits), so this is not a free hyper-parameter: swapping it at evaluation
    time is a train/eval mismatch probe, not a search for a better null.
    Choosing a different null properly would need retraining. Reported as
    sensitivity, and cheap: if CFG collapses under a mismatched null, the
    +8 points are a statement about that specific null.
    """
    out = []
    for strat in ("half", "data_center", "zeros"):
        out.append(Cell(name=f"null_{strat}_w12", guidance_scale=12.0,
                        extra={"null_strategy": strat}, **DET))
    return out


def grid_ag_ema() -> List[Cell]:
    """AutoGuidance badness axis: is the EMA or the raw weight the better bad?

    Karras et al. degrade the good model along capacity and training time; EMA
    is a third axis this repository has for free, and the raw weights of a
    checkpoint are a *differently* bad model from its EMA at the same step, not
    merely a noisier one.
    """
    out = []
    for step in ("000250000", "000350000"):
        ck = f"{RUN_DIR}/checkpoints/step={step}.pt"
        for ema in (1, 0):
            out.append(Cell(name=f"agema_{step}_ema{ema}", ag_scale=15.0,
                            bad_checkpoint=ck, bad_ema=ema, **DET))
    return out


def _seeds(cell: Cell, seeds) -> List[Cell]:
    """Replicate one cell across seeds, keeping everything else fixed."""
    out = []
    for s in seeds:
        d = asdict(cell)
        d.pop("extra", None)
        d["name"] = f"{cell.name}_seed{s}"
        d["seed"] = s
        out.append(Cell(extra=dict(cell.extra), **d))
    return out


def grid_factorial_confirm(root: str = ".") -> List[Cell]:
    """Phase 13 at confirmation grade: the 12-cell factorial, full test set, 3 seeds.

    The n=250 exploration factorial cannot resolve interactions: its paired
    deltas carry a +/-0.04 CI while a two-way interaction is second-order and
    smaller than either main effect. Promoting the same 12 cells to the full
    1319 problems and three seeds -- the settings the single-axis confirmations
    already used, so the per-axis operating points transfer unchanged -- is what
    makes an interaction estimate meaningful rather than decorative.
    """
    return grid_confirm(grid_factorial(**_require_operating_point(root)),
                        seeds=(42, 43, 44))


GRIDS = {
    "cfg_coarse": grid_cfg_coarse,
    "cfg_high": grid_cfg_high,
    "cfg_confirm": grid_cfg_confirm,
    "sg_confirm": grid_sg_confirm,
    "ag_confirm": grid_ag_confirm,
    "cfg_stochastic": grid_cfg_stochastic,
    "ag": grid_ag,
    "ag_high": grid_ag_high,
    "sg": grid_sg,
    "sg_delta": grid_sg_delta,
    "sg_mf": grid_sg_mf,
    "factorial": lambda root=".": grid_factorial(**_operating_point(root)),
    "factorial_confirm": grid_factorial_confirm,
    "solver_control": grid_solver_control,
    "repro": grid_repro,
    "stoch_screen": grid_stoch_screen,
    "stoch_confirm": grid_stoch_confirm,
    "churn_anatomy": grid_churn_anatomy,
    "compute_control": grid_compute_control,
    "null_ablation": grid_null_ablation,
    "ag_ema": grid_ag_ema,
    "nfe": lambda root=".": grid_nfe(**_operating_point(root)),
}
# Grids that need to inspect the filesystem for available checkpoints.
_ROOT_AWARE = {"ag", "ag_high", "ag_confirm", "factorial", "factorial_confirm",
               "nfe"}


def build(name: str, root: str = ".") -> List[Cell]:
    if name not in GRIDS:
        raise SystemExit(f"unknown grid {name!r}; have {sorted(GRIDS)}")
    fn = GRIDS[name]
    return fn(root) if name in _ROOT_AWARE else fn()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_show = sub.add_parser("show", help="print a grid for review")
    p_show.add_argument("grid")
    p_show.add_argument("--root", default=".")

    p_size = sub.add_parser("size", help="print the number of cells (array upper bound)")
    p_size.add_argument("grid")
    p_size.add_argument("--root", default=".")

    p_cli = sub.add_parser("cli", help="print one cell's argv (used by the job script)")
    p_cli.add_argument("grid")
    p_cli.add_argument("index", type=int)
    p_cli.add_argument("--out_dir", required=True)
    p_cli.add_argument("--root", default=".")

    p_all = sub.add_parser("list", help="list every grid and its size")
    p_all.add_argument("--root", default=".")

    args = ap.parse_args()

    if args.cmd == "list":
        for g in sorted(GRIDS):
            try:
                print(f"{g:18} {len(build(g, args.root)):4d} cells")
            except Exception as e:
                print(f"{g:18} ERROR {e}")
        return

    cells = build(args.grid, args.root)

    if args.cmd == "size":
        print(len(cells))
    elif args.cmd == "show":
        for i, c in enumerate(cells):
            print(f"[{i:3d}] {c.name:34} w={c.guidance_scale:<5g} ag={c.ag_scale:<5g} "
                  f"sg={c.sg_scale:<5g}{c.sg_variant if c.sg_scale else '':<6} "
                  f"nfe={c.steps:<5d} n={c.limit:<5d} bad={_step_of(c.bad_checkpoint or '')}")
    elif args.cmd == "cli":
        if not (0 <= args.index < len(cells)):
            raise SystemExit(f"index {args.index} out of range for grid "
                             f"{args.grid!r} ({len(cells)} cells)")
        print(" ".join(cells[args.index].cli(args.out_dir)))


if __name__ == "__main__":
    main()

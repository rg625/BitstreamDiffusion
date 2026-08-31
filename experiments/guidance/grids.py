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


def grid_cfg_stochastic() -> List[Cell]:
    """The same sweep under EDM churn, to separate 'guidance helps' from
    'guidance substitutes for stochasticity'."""
    ws = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    return [Cell(name=f"cfgstoch_w{w:g}", guidance_scale=w, **STOCH) for w in ws]


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


GRIDS = {
    "cfg_coarse": grid_cfg_coarse,
    "cfg_stochastic": grid_cfg_stochastic,
    "ag": grid_ag,
    "sg": grid_sg,
    "sg_delta": grid_sg_delta,
    "sg_mf": grid_sg_mf,
    "factorial": lambda root=".": grid_factorial(**_operating_point(root)),
    "nfe": lambda root=".": grid_nfe(**_operating_point(root)),
}
# Grids that need to inspect the filesystem for available checkpoints.
_ROOT_AWARE = {"ag", "factorial", "nfe"}


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

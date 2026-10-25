"""Temporal-ordering experiment grid.

DESIGN
------
Hypothesis: denoising the generated suffix in an order (left-to-right, or a
random permutation) rather than simultaneously improves GSM8K accuracy.

Control and intervention differ by EXACTLY ONE NUMBER, `order_w`. Both arms use
the same sampler class (OrderedSampler), the same sigma schedule, the same
initialisation, the same checkpoint, the same steps and the same seed. At
order_w=0 the sampler is a plain uniform-sigma deterministic Euler run, which is
today's model; this is verified bit-identically by tests, not assumed.

This is a SAMPLING-TIME intervention on the healthy production checkpoint. It is
deliberately not a training-time one: every training run in this environment
diverges before ~12k steps, so a trained ordering arm could not be evaluated.

Ordering is applied to the SUFFIX only; prompt positions keep the global sigma.
"""
from __future__ import annotations

CONFIG = "configs/tasks/tinygsm_bits_cfg.py"
RUN_DIR = "tinigsm_gsm8k/runs/cobit_raw_binary_bits_cfg"
CKPT = f"{RUN_DIR}/checkpoints/last.pt"      # 500k, EMA, the healthy model
SCHEDULE = "entropic"


class Cell:
    def __init__(self, name, order_w, order_mode, steps, limit, seed, order_seed=None):
        self.name, self.order_w, self.order_mode = name, order_w, order_mode
        self.steps, self.limit, self.seed = steps, limit, seed
        self.order_seed = order_seed

    def cli(self, out_dir):
        a = [
            "--config", CONFIG, "--checkpoint", CKPT, "--ema", "1",
            "--schedule", SCHEDULE, "--sampler_kind", "ordered",
            "--steps", str(self.steps), "--limit", str(self.limit),
            "--seed", str(self.seed), "--out_dir", out_dir,
            "--order_w", str(self.order_w), "--order_mode", self.order_mode,
            "--regime", "ordering",
            "--tag", self.name,
        ]
        if self.order_seed is not None:
            a += ["--order_seed", str(self.order_seed)]
        return a


def screen(limit=250, seed=0, steps=256):
    """Stage 1: is any w worth confirming? One seed, 250 problems."""
    cells = [Cell(f"screen_ctrl_s{steps}", 0.0, "none", steps, limit, seed)]
    for w in (0.1, 0.25, 0.5, 1.0):
        cells.append(Cell(f"screen_l2r_w{w}_s{steps}", w, "l2r", steps, limit, seed))
    for w in (0.25, 0.5):
        cells.append(Cell(f"screen_rand_w{w}_s{steps}", w, "random", steps, limit, seed,
                          order_seed=1234))
    return cells


def screen_steps(limit=250, seed=0):
    """Does ordering need a longer trajectory to pay off? Confounder check:
    with w>0 each token only denoises over part of the schedule, so too few
    steps could hide a real effect."""
    out = []
    for steps in (512,):
        out.append(Cell(f"screen_ctrl_s{steps}", 0.0, "none", steps, limit, seed))
        out.append(Cell(f"screen_l2r_w0.25_s{steps}", 0.25, "l2r", steps, limit, seed))
    return out


def confirm(arms, limit=1319, steps=256, seeds=(0, 1, 2)):
    """Stage 2: full 1319 problems, multiple seeds, paired against the control."""
    cells = []
    for seed in seeds:
        cells.append(Cell(f"confirm_ctrl_s{seed}", 0.0, "none", steps, limit, seed))
        for w, mode in arms:
            cells.append(Cell(f"confirm_{mode}_w{w}_s{seed}", w, mode, steps, limit,
                              seed, order_seed=1234 + seed))
    return cells

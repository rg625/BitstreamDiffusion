# configs/tasks/tinygsm_bits_ordering.py
#
# TRAINING-TIME temporal ordering on TinyGSM. One variable: cfg.train.ordering.
#
#   ORD_MODE = none | l2r | r2l | random      (none + ORD_W=0 is the control)
#   ORD_W    = ordering strength w            (0 => bit-identical to the control)
#   ORD_SEED, ORD_STEPS, ORD_TAG
#
# WHY FINE-TUNE FROM THE 500k CHECKPOINT RATHER THAN TRAIN FROM SCRATCH
# --------------------------------------------------------------------
# From-scratch is not evaluable in this environment, and that is measured, not
# assumed: every training run here diverges before ~12k steps, and GSM8K
# accuracy is exactly 0.0000 at both 5k and 10k (the first non-zero point we
# have is 0.2024 at 200k). A from-scratch ordering arm would therefore produce
# four broken models scoring 0 and answer nothing -- the same trap the
# CE-vs-SM branch fell into.
#
# All four arms instead start from the SAME healthy production checkpoint and
# get the SAME budget, so the comparison is matched and accuracy is measurable
# from step one (~0.16). This answers "does CONTINUED training with ordering
# improve TinyGSM?", which is a real question; it does NOT answer "does
# ordering-from-scratch help", and that limitation is reported.
import importlib.util
import os


def get_config():
    base = os.path.join(os.path.dirname(__file__), "tinygsm_bits_cfg.py")
    spec = importlib.util.spec_from_file_location("tinygsm_bits_cfg_base", base)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.get_config()          # production recipe: p_uncond=0.1, edm, clamp off

    mode = os.environ.get("ORD_MODE", "none")
    if mode not in ("none", "l2r", "r2l", "random"):
        raise SystemExit(f"ORD_MODE must be none|l2r|r2l|random, got {mode!r}")
    w = float(os.environ.get("ORD_W", 0.0))
    if mode == "none" and w != 0.0:
        raise SystemExit("ORD_MODE=none is the control and requires ORD_W=0")
    if mode != "none" and w == 0.0:
        raise SystemExit(f"ORD_MODE={mode} with ORD_W=0 is the control in disguise")

    o = type(cfg.train.checkpointing)()
    o.enabled = (w != 0.0)
    o.mode = mode
    o.w = w
    cfg.train.ordering = o

    # Every arm starts from the same healthy weights, fresh optimiser state.
    # ORD_INIT=none trains FROM SCRATCH, which is the real experiment: ordering
    # is baked into the whole trajectory rather than bolted onto a model that
    # converged without it. Fine-tuning was only ever a workaround for the
    # divergence blocker, and that blocker is now fixed.
    _init = os.environ.get(
        "ORD_INIT", "tinigsm_gsm8k/runs/cobit_raw_binary_bits_cfg/checkpoints/last.pt")
    if str(_init).lower() not in ("none", "", "scratch"):
        cfg.train.init_from = _init

    # Fine-tuning learning rate. The from-scratch value (3e-4) applied to
    # ALREADY-CONVERGED weights kicked every arm out of its minimum: the CONTROL
    # diverged first, at step 2,323, which is a defect of the setup and not a
    # property of ordering. A converged model needs a smaller step.
    cfg.optim.lr = float(os.environ.get("ORD_LR", cfg.optim.lr))

    cfg.train.seed = int(os.environ.get("ORD_SEED", 42))
    cfg.optim.total_steps = int(os.environ.get("ORD_STEPS", 30000))
    cfg.train.checkpointing.interval.every_steps = int(
        os.environ.get("ORD_CKPT_EVERY", 10000))
    cfg.train.checkpointing.interval.keep_last = None

    # Divergence guard, factor 10: bracketed between production's own lifetime
    # max of 6.29x (its sigma-schedule handover) and the smallest real
    # divergence measured at 17.2x.
    g = type(cfg.train.checkpointing)()
    g.enabled = True
    g.factor = 10.0
    g.patience = 200
    g.min_steps = 1000          # fine-tuning from converged weights: no warmup phase
    g.ema_decay = 0.99
    g.ring_steps = 4000
    cfg.train.divergence_guard = g

    # EPOCH-BOUNDARY DEADLOCK MITIGATION.
    # All four arms have hung on an ALLREDUCE at the start of an epoch, with a
    # 120-minute process-group timeout already in force -- so it is a genuine
    # desynchronisation, not a timeout that is merely too short. At each epoch
    # end rank 0 writes best/epoch checkpoints (2-3.4 GB) to a 92%-full Lustre
    # while the other ranks move on and block in a collective.
    # Interval checkpoints (every 100k) and last.pt (every 5k) already cover
    # resume and analysis, so the per-epoch saves are pure redundancy: drop
    # them and the heavy rank-0 work at the boundary goes with them.
    cfg.train.checkpointing.save_top_k = 0
    cfg.train.entropy_plot_every_k_epochs = 100000   # effectively never

    cfg.train.objective_probe = type(cfg.train.checkpointing)()
    cfg.train.objective_probe.enabled = True
    cfg.train.objective_probe.every_steps = int(os.environ.get("ORD_PROBE_EVERY", 250))

    tag = os.environ.get("ORD_TAG", "").strip()
    parts = [p for p in (tag, mode if mode == "none" else f"{mode}_w{w}",
                         f"s{cfg.train.seed}") if p]
    cfg.experiment = "tasks/tinygsm/ord_" + "_".join(parts)
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/last.pt"
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/gsm8k_eval"
    return cfg

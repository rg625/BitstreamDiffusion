# configs/tasks/tinygsm_bits_cfg.py -> deliberately under-trained "bad" model
#
# AutoGuidance (Karras et al.) guides with a *bad version of the same model*.
# The confound-free way to produce one is to change nothing but the amount of
# training, so this config is byte-for-byte `tinygsm_bits_cfg.py` except for the
# step budget and the run directory:
#
#   * SAME architecture  -> the AG direction cannot pick up a capacity change
#   * SAME p_uncond=0.1  -> the bad model has an unconditional mode too, which
#                           the nested CFG+AutoGuidance form requires
#   * SAME data + cache  -> no distribution shift between good and bad
#   * FEWER steps        -> the only axis that varies
#
# Checkpoints are kept at every 25k steps (keep_last=0), so one run yields the
# whole badness ladder {25k, 50k, 75k, 100k} rather than a single point. That
# matters: Karras et al.'s claim is that there is an *optimal* degree of
# badness, which cannot be tested with one bad model.
#
# NOTE: if the original run's periodic checkpoints still exist anywhere (it was
# configured with every_steps=25_000, and eval directories survive for steps
# 100k/150k/200k/225k/300k/350k), copying those is strictly better than this
# run: a wider, already-measured badness range at zero GPU cost. Train only if
# they are genuinely gone.
import importlib.util
import os


def get_config():
    base_path = os.path.join(os.path.dirname(__file__), "tinygsm_bits_cfg.py")
    spec = importlib.util.spec_from_file_location("tinygsm_bits_cfg_base", base_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.get_config()

    # --- the only substantive change: a short budget -----------------------
    cfg.optim.total_steps = int(os.environ.get("BAD_TOTAL_STEPS", 100_000))
    # Warmup must stay as in the good run, or the bad model is bad for the
    # wrong reason (a different optimisation trajectory rather than less of it).
    cfg.optim.warmup = 2_500

    # Keep every periodic checkpoint: the badness ladder is the point.
    cfg.train.checkpointing.interval.enabled = True
    cfg.train.checkpointing.interval.every_steps = 25_000
    cfg.train.checkpointing.interval.keep_last = 0

    # --- separate run dir so nothing can clobber the good run --------------
    cfg.experiment = "tasks/tinygsm/cobit_raw_binary_bits_cfg_bad"
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/step=000100000.pt"
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/gsm8k_eval"
    return cfg

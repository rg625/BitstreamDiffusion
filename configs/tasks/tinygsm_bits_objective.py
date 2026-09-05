# configs/tasks/tinygsm_bits_objective.py
#
# binary_sm vs binary_ce pilot. ONE variable: cfg.train.loss_type.
#
# The architecture already emits logits [B,8192]; "return logits" is not an
# experiment. What differs is the objective applied to those logits:
#
#   binary_sm : w(sigma) * ||sigmoid(ell) - x0||^2      (production recipe)
#   binary_ce : w(sigma) * BCEWithLogits(ell, x0)
#
#   dL_sm/d_ell = w * (D - x0) * D(1-D)
#   dL_ce/d_ell = w * (D - x0)
#
# Both are proper scoring rules for the Bernoulli mean, so they share an optimum
# and this is an optimisation question, not a capacity one. The measured
# suppression factor D(1-D) is ~0 for the median bit at the production
# checkpoint, so binary_sm carries 3-14% of CE's gradient magnitude.
#
# PRIMARY endpoint is the gradient-survival trajectory during training, NOT the
# 100k accuracy: the hypothesis is that CE prevents the saturation trap forming,
# which is a claim about the training curve.
#
# ARM is read from the environment so a single file serves both, guaranteeing
# that nothing else can drift between them:
#   OBJ_LOSS=binary_sm | binary_ce
import importlib.util
import os


def get_config():
    base_path = os.path.join(os.path.dirname(__file__), "tinygsm_bits.py")
    spec = importlib.util.spec_from_file_location("tinygsm_bits_base", base_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.get_config()

    loss = os.environ.get("OBJ_LOSS", "binary_sm")
    if loss not in ("binary_sm", "binary_ce"):
        raise SystemExit(f"OBJ_LOSS must be binary_sm or binary_ce, got {loss!r}")
    cfg.train.loss_type = loss

    cfg.optim.total_steps = int(os.environ.get("OBJ_STEPS", 100_000))
    # NOTE: the real key is train.checkpointing.interval.every_steps. An earlier
    # draft of this file set `cfg.train.checkpoint_every_steps`, which nothing
    # reads -- it would have silently produced no extra checkpoints.
    cfg.train.checkpointing.interval.enabled = True
    cfg.train.checkpointing.interval.every_steps = int(os.environ.get("OBJ_CKPT_EVERY", 10_000))
    cfg.train.checkpointing.interval.keep_last = None       # keep ALL milestones
    # Mechanistic telemetry: this is the PRIMARY endpoint, so log it often.
    cfg.train.objective_probe = type(cfg.train.checkpointing)()
    cfg.train.objective_probe.enabled = True
    cfg.train.objective_probe.every_steps = int(os.environ.get("OBJ_PROBE_EVERY", 500))

    cfg.experiment = f"tasks/tinygsm/obj_{loss}"
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/last.pt"
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/gsm8k_eval"
    return cfg

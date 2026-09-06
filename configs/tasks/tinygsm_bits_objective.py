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
    # Base on the CFG config: it is the exact recipe of the run that trained
    # healthily to 500k (cond.p_uncond=0.1). The previous base, tinygsm_bits.py,
    # gave p_uncond=0.0 -- the only substantive difference from production, and
    # therefore the one thing that made the SM control unvalidated. See
    # docs/objective_pilot_postmortem.md section 3.
    base_path = os.path.join(os.path.dirname(__file__), "tinygsm_bits_cfg.py")
    spec = importlib.util.spec_from_file_location("tinygsm_bits_base", base_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.get_config()

    loss = os.environ.get("OBJ_LOSS", "binary_sm")
    if loss not in ("binary_sm", "binary_ce"):
        raise SystemExit(f"OBJ_LOSS must be binary_sm or binary_ce, got {loss!r}")
    cfg.train.loss_type = loss

    # Bound the low-sigma amplifier, IDENTICALLY in both arms so loss_type stays
    # the only variable. Under the production sigma draw the 18% of samples with
    # sigma<0.1 carry 91% of all weight mass, at sigmas where the Bayes risk is
    # ~2e-7; unbounded, w(sigma) reaches 2.5e5. Both arms of the first pilot
    # diverged (CE at step 6,580, SM at 19,540), which no objective-specific
    # story explains. p90 of the current draw is 246, so this clips a tail.
    # Derivation: docs/ce_weighting_derivation.md
    cfg.train.loss_weight_max = float(os.environ.get("OBJ_WMAX", 100.0))

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

    # Abort a diverged run rather than let it sit frozen. The first pilot spent
    # ~55 GPU-h with bit-identical weights after diverging.
    cfg.train.divergence_guard = type(cfg.train.checkpointing)()
    cfg.train.divergence_guard.enabled = True
    cfg.train.divergence_guard.factor = 20.0
    cfg.train.divergence_guard.patience = 200
    cfg.train.divergence_guard.min_steps = 2500      # = optim.warmup
    cfg.train.divergence_guard.ema_decay = 0.99

    # OBJ_TAG keeps a smoke run from writing into the pilot's directory. Without
    # it a 5k smoke would overwrite the 100k arm's checkpoints under the same
    # experiment name -- the same class of silent collision as the FKC filename
    # bug. Empty tag => the canonical pilot path, unchanged.
    tag = os.environ.get("OBJ_TAG", "").strip()
    suffix = f"_{tag}" if tag else ""
    cfg.experiment = f"tasks/tinygsm/obj_{loss}{suffix}"
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/last.pt"
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/gsm8k_eval"
    return cfg

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

    # Low-sigma weight clamp: OPT-IN, and OFF by default.
    #
    # I proposed this as the fix for the first pilot's divergence and predicted
    # both arms would then train past 20k. The 5k smoke REFUTED that: with
    # OBJ_WMAX=100 the SM arm broke at step 3,900 -- earlier than the unclamped
    # run's 19,540 -- while CE was stable. Since that SM arm differed from the
    # healthy 500k production run by this key alone, the clamp is now a suspect
    # rather than a fix, and it must be tested as a variable, not assumed.
    #
    # Default off => the SM control reproduces production exactly.
    # Derivation of why a CE-specific weighting is NOT needed:
    #   docs/ce_weighting_derivation.md
    _wmax = os.environ.get("OBJ_WMAX", "").strip().lower()
    if _wmax not in ("", "none", "off", "0"):
        cfg.train.loss_weight_max = float(_wmax)

    cfg.optim.total_steps = int(os.environ.get("OBJ_STEPS", 20_000))
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
    # factor=10, bracketed against the KNOWN-GOOD production run and against
    # six measured divergences.
    #
    #   production, 500k steps, trained successfully: max EMA/best = 6.29,
    #     sitting at a stable ~5.4x plateau for its last 450k steps. The rise
    #     begins exactly at entropy_warmup_steps=40000 and ramps over
    #     entropy_transition_steps=10000: it is the sigma-schedule handover
    #     changing the loss scale, not a divergence.
    #   our six 20k runs, all genuinely diverging: 17.2x to 2942x.
    #
    # 10 sits between them with ~1.6x margin below and ~1.7x above. The previous
    # default of 4 WOULD HAVE ABORTED PRODUCTION at ~step 48,000.
    cfg.train.divergence_guard.factor = 10.0
    cfg.train.divergence_guard.patience = 200
    cfg.train.divergence_guard.min_steps = 2500      # = optim.warmup
    cfg.train.divergence_guard.ema_decay = 0.99

    # OBJ_TAG keeps a smoke run from writing into the pilot's directory. Without
    # it a 5k smoke would overwrite the 100k arm's checkpoints under the same
    # experiment name -- the same class of silent collision as the FKC filename
    # bug. Empty tag => the canonical pilot path, unchanged.
    # Seed is the replication axis of the stability study. The whole current
    # result rests on 2 SM runs breaking and 2 CE runs not, one seed each; a
    # single divergence event is not evidence of a systematic difference.
    # Learning rate. The production value (3e-4) diverges in this environment
    # in every from-scratch run measured (8/8, including the production-era
    # commit). Fine-tuning at 3e-5 was stable through 15k where 3e-4 broke at
    # 2,323, so lr is a live candidate for the instability and is overridable
    # here to test it from scratch.
    cfg.optim.lr = float(os.environ.get("OBJ_LR", cfg.optim.lr))
    # Batch override: the token-softmax arm cannot run at 512 (its [B,S,V]
    # logits are 12.9 GB/GPU), so its binary CONTROL must run at the same batch
    # or representation would be confounded with batch size.
    cfg.train.batch_size = int(os.environ.get("OBJ_BATCH", cfg.train.batch_size))

    cfg.train.seed = int(os.environ.get("OBJ_SEED", 42))

    tag = os.environ.get("OBJ_TAG", "").strip()
    parts = [p for p in (tag, f"s{cfg.train.seed}") if p]
    cfg.experiment = f"tasks/tinygsm/obj_{loss}_" + "_".join(parts)
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/last.pt"
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/gsm8k_eval"
    return cfg

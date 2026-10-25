# configs/tasks/tinygsm_bits_cobit_m.py
#
# CoBit-M (462M) on TinyGSM -> GSM8K, unguided.
#
#   data, conditioning, validation, diffusion, loss, entropy schedule, EMA,
#   batch, seed       <- configs/tasks/tinygsm_bits.py (unguided CoBit-S TinyGSM)
#   trunk, optimiser  <- configs/owt/rate_bits_edm_weight_medium_24x1024.py
#                        (CoBit-M OWT), read from that file, not copied by hand
#
# Flattened, the two parents differ only in the trunk (24 x 1024, 16 heads,
# dim_ff 4096), the optimiser (lr 2e-4, beta2 0.99, weight decay 0.01, cosine,
# 5k warmup) and checkpoint cadence; everything else is already identical.
#
# Schedule and stopping:
#   optim.total_steps = 500k  -- the cosine is laid out over 500k
#   train.stop_at_step = 250k -- the run stops there (LR ~1.0e-4, mid-cosine)
#   resume with COBIT_M_STOP_AT_STEP=500000 to finish the same cosine.
# Never set TINYGSM_TOTAL_STEPS for this run: train.py applies it to
# optim.total_steps after the resume merge, which would re-lay the cosine.
#
# Data order: train.data_order_seed = 0, the shuffle seed every pre-T4
# multi-GPU run used (DistributedSampler's default), so batches match the June
# CoBit-S runs up to their first resume. Weights/noise still use seed 42.
#
# Checkpoints: step=*.pt only at COBIT_M_CKPT_STEPS (default 100k, 200k, 250k);
# rolling last.pt every 2k (CoBit-M OWT); no best-val copies (save_top_k=0).
#
# Env:
#   COBIT_M_STOP_AT_STEP  default 250000
#   COBIT_M_CKPT_STEPS    default "100000,200000,250000"
#   COBIT_M_ACCUM         gradient accumulation, default 1 (layout only)
#   COBIT_M_TAG           run-dir tag, default "s42" (smoke runs use their own)
import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_OWT_M = os.path.join(_HERE, "..", "owt", "rate_bits_edm_weight_medium_24x1024.py")
_TINYGSM_S = os.path.join(_HERE, "tinygsm_bits.py")

TRUNK_KEYS = ("embed_dim", "dim_ff", "n_blocks", "n_heads")
OPTIM_KEYS = ("optimizer", "lr", "weight_decay", "beta1", "beta2", "eps",
              "grad_clip", "scheduler", "warmup")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


def get_config():
    cfg = _load(_TINYGSM_S, "tinygsm_bits_base")
    owt = _load(_OWT_M, "owt_cobit_m")

    tag = os.environ.get("COBIT_M_TAG", "s42")
    cfg.experiment = f"tasks/tinygsm/cobit_m_bits_{tag}"

    for k in TRUNK_KEYS:
        setattr(cfg.model, k, getattr(owt.model, k))
    for k in OPTIM_KEYS:
        setattr(cfg.optim, k, getattr(owt.optim, k))
    cfg.optim.total_steps = 500_000

    cfg.train.stop_at_step = int(os.environ.get("COBIT_M_STOP_AT_STEP", 250_000))
    cfg.train.data_order_seed = 0
    cfg.train.grad_accum_steps = int(os.environ.get("COBIT_M_ACCUM", 1))

    ck = cfg.train.checkpointing
    ck.save_top_k = 0
    ck.interval.enabled = True
    ck.interval.at_steps = [int(s) for s in os.environ.get(
        "COBIT_M_CKPT_STEPS", "100000,200000,250000").split(",") if s.strip()]
    ck.interval.keep_last = 0
    ck.resume_interval.enabled = True
    ck.resume_interval.every_steps = owt.train.checkpointing.resume_interval.every_steps

    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/step=000250000.pt"
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/gsm8k_eval"
    return cfg

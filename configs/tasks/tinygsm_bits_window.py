# configs/tasks/tinygsm_bits_window.py
#
# WINDOW temporal ordering on TinyGSM, from scratch. One variable per arm:
# (WIN_MODE, WIN_K) in cfg.train.ordering.
#
#   WIN_MODE = none | l2r | r2l | random     (none <=> WIN_K=inf, the control)
#   WIN_K    = inf | 64 | 16 | 4             window width in TOKENS
#   WIN_SEED, WIN_STEPS, WIN_TAG, WIN_CKPT_EVERY
#   WIN_LOSSMATCH=64|16|4  (control only) supervision-matched control: uniform
#                          sigma, each token supervised w.p. k/(k+L) as in a
#                          window-k arm
#
#     t_j(tau) = clip(tau*(1 + L/k) - r_j/k, 0, 1)      sigma_j = F^-1(t_j)
#
# r_j is an absolute token priority (diffusion/continuous/ordering.py), so the
# ~74% of the canvas that is padding cannot dilute the answer's order -- the
# defect that made the w=0.25 study a near no-op for l2r and r2l.
#
# IDENTICAL TRAINING NOISE. Every arm, control included, draws tau ~ U(0,1] and
# maps through ONE frozen noise CDF, assets/noise_tables/tinygsm_window_F_v1.pt
# (the uniform fs500k control's own adapted schedule, frozen). Nothing is
# refitted per arm: the w=0.25 arms' adaptive schedules had drifted apart
# (random's 25% quantile 0.56 vs the control's 0.92), confounding the result.
# Every loss-bearing token then has sigma ~ F in every arm; arms at one k see an
# identical multiset of sigmas, and for a given seed identical data, Gaussian
# noise and tau (dedicated, step-keyed generators). tests/test_window_ordering.py.
#
# Everything else is the ord_fs500k recipe: binary_sm, EDM weighting,
# p_uncond 0.1, batch 512, lr 1e-4 constant, bf16, EMA 0.9999, from scratch.
import importlib.util
import math
import os

_NOISE_TABLE = "assets/noise_tables/tinygsm_window_F_v1.pt"


def _parse_k(s):
    s = str(s).strip().lower()
    return float("inf") if s in ("inf", "infinity", "none", "") else float(s)


def get_config():
    mode = os.environ.get("WIN_MODE", "none")
    k = _parse_k(os.environ.get("WIN_K", "inf"))
    if mode not in ("none", "l2r", "r2l", "random"):
        raise SystemExit(f"WIN_MODE must be none|l2r|r2l|random, got {mode!r}")
    if (mode == "none") != math.isinf(k):
        raise SystemExit(f"WIN_MODE=none iff WIN_K=inf (got mode={mode}, k={k})")
    if not math.isinf(k) and k <= 0:
        raise SystemExit(f"WIN_K must be > 0, got {k}")
    lm = os.environ.get("WIN_LOSSMATCH", "").strip()
    lm = int(lm) if lm else None
    if lm is not None and mode != "none":
        raise SystemExit("WIN_LOSSMATCH is for the control (WIN_MODE=none) only")

    # The w-ordering config carries the recipe (guard, checkpointing, probe,
    # deadlock mitigations). Build it as its own control, then replace ordering.
    over = {"ORD_MODE": "none", "ORD_W": "0", "ORD_INIT": "none",
            "ORD_LR": os.environ.get("WIN_LR", "1e-4"),
            "ORD_SEED": os.environ.get("WIN_SEED", "42"),
            "ORD_STEPS": os.environ.get("WIN_STEPS", "500000"),
            "ORD_CKPT_EVERY": os.environ.get("WIN_CKPT_EVERY", "50000"),
            "ORD_TAG": ""}
    saved = {key: os.environ.get(key) for key in over}
    os.environ.update(over)
    try:
        base = os.path.join(os.path.dirname(__file__), "tinygsm_bits_ordering.py")
        spec = importlib.util.spec_from_file_location("tinygsm_bits_ordering_base", base)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = mod.get_config()
    finally:
        for key, v in saved.items():
            if v is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = v

    o = type(cfg.train.checkpointing)()
    o.enabled = True                 # the control runs the window path too (k=inf)
    o.kind = "window"
    o.mode = mode
    o.k = k
    o.noise_table = _NOISE_TABLE
    o.loss_match_k = lm
    cfg.train.ordering = o
    # Layout only (never changes the effective batch of 512): gradient
    # accumulation for fewer-GPU smokes.
    if os.environ.get("WIN_ACCUM"):
        cfg.train.grad_accum_steps = int(os.environ["WIN_ACCUM"])
    # The frozen F replaces the draw; make sure nothing adapts behind it.
    cfg.train.entropy_use_for_sampling = False

    ktag = "inf" if math.isinf(k) else f"{int(k)}"
    if lm is not None:
        ktag += f"_lm{lm}"
    tag = os.environ.get("WIN_TAG", "win").strip() or "win"
    cfg.experiment = f"tasks/tinygsm/{tag}_{mode}_k{ktag}_s{cfg.train.seed}"
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/last.pt"
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/gsm8k_eval"
    return cfg

"""CoBit-S LM1B / OWT on a precomputed sigma grid (scripts/dgc/build_gamma_grid.py): the gamma-dependent Euler study.

Inherits the DGC A/B configs (same checkpoints, external PPL, entropy pass, no test loader) and replaces the sampling
specs. NFE here is the ACTUAL number of model calls T: num_steps = T points, T - 1 intervals, grid key N<T-1>
(target_nfe = T + 1, because compute_nfe adds a final call the DDIM loop does not make).

Env vars:
  GE_DATA     lm1b | owt
  GE_ARM      grid label for the output directory (log_uniform, karras, table, dgc_profile, gamma_S4, gamma_g0.2, ...)
  GE_GRID     grid npz (keys N<T-1>)
  GE_SAMPLER  euler_lambda (literal Euler, lambda_i = gamma sigma_i / Delta_i, null self-conditioning) |
              ddim_grid (the production DDIM sampler with EDM churn, carry self-conditioning)
  GE_S        euler_lambda only: fixed S, gamma = S / (T - 1) per cell (theory sweep)
  GE_GAMMA    euler_lambda: fixed realized gamma; ddim_grid: EDM churn gamma (s_churn = gamma (T - 1), every step)
  GE_NFES     "64/128/256" actual calls ("/" because sbatch --export splits on commas)
  EVAL_SEED   seed (default 42); the same seed across arms gives the same initial noise
  DGC_NUM_SAMPLES  smoke runs only (default 1024)
"""
import os

from ml_collections import config_dict


def get_config():
    data = os.environ["GE_DATA"].strip().lower()
    arm, grid, sampler = os.environ["GE_ARM"], os.environ["GE_GRID"], os.environ["GE_SAMPLER"].strip().lower()
    nfes = [int(t) for t in os.environ.get("GE_NFES", "64/128/256").replace(",", "/").split("/") if t.strip()]
    S, gamma = os.environ.get("GE_S"), os.environ.get("GE_GAMMA")
    seed = int(os.environ.get("EVAL_SEED", 42))
    if sampler not in ("euler_lambda", "ddim_grid"):
        raise ValueError(f"GE_SAMPLER must be euler_lambda or ddim_grid, got {sampler!r}")
    if sampler == "euler_lambda" and (S is None) == (gamma is None):
        raise ValueError("euler_lambda needs exactly one of GE_S or GE_GAMMA")
    if sampler == "ddim_grid" and (gamma is None or S is not None):
        raise ValueError("ddim_grid needs GE_GAMMA (EDM churn) and no GE_S")

    os.environ.pop("CELLS", None)
    if data == "lm1b":
        os.environ.setdefault("SCHEDULE", "cobit")
        from configs.lm1b.continuous.eval.dgc_sweep import get_config as get_base
    elif data == "owt":
        os.environ["MODEL"], os.environ["SCHEDULE"] = "s", "cobit"
        os.environ.pop("CHURN_SWAP", None)
        from configs.owt.dgc_schedule_ab import get_config as get_base
    else:
        raise ValueError(f"GE_DATA must be lm1b or owt, got {data!r}")
    cfg = get_base()

    ev = cfg.evaluation
    ev.grid_file = grid
    if sampler == "euler_lambda":
        if S is not None:
            ev.euler_S = float(S)
        else:
            ev.euler_gamma = float(gamma)
    knob = f"S{float(S):g}" if S is not None else f"g{float(gamma):g}"
    smoke = f"_smoke{os.environ['DGC_NUM_SAMPLES']}" if os.environ.get("DGC_NUM_SAMPLES") else ""
    ev.out_dir = f"runs/{cfg.experiment}/gamma_euler/{sampler}_{knob}/{arm}/seed{seed}{smoke}"
    ev.samples_dir = f"{ev.out_dir}/samples"
    ev.results_csv = f"{ev.out_dir}/results.csv"
    ev.shared_text_cache_dir = f"{ev.out_dir}/shared_text_cache"
    ext = ev.external_ppl
    ext.shared_cache_dir = ev.shared_text_cache_dir
    ext.seed = seed
    cfg.train.seed = seed
    if os.environ.get("DGC_NUM_SAMPLES"):
        n = int(os.environ["DGC_NUM_SAMPLES"])
        ext.num_samples = n
        ext.micro_batch_size = min(n, ext.micro_batch_size)

    ev.sampling_sweep.target_nfes = [T + 1 for T in nfes]
    ev.sampling_sweep.specs = []
    for T in nfes:
        spec = config_dict.ConfigDict()
        spec.sampler_name = sampler
        spec.sc_refresh_modes = ["null" if sampler == "euler_lambda" else "carry"]
        spec.target_nfes = [T + 1]
        spec.ati_etas = [0.0]
        spec.stochastic_enabled = sampler == "ddim_grid" and float(gamma) > 0
        if spec.stochastic_enabled:
            spec.stochastic_mode = "edm_churn"
            spec.s_churn = float(gamma) * (T - 1)
            spec.s_noise = 1.003
            spec.window_mode = "full"
            spec.s_tmin = None
            spec.s_tmax = None
            spec.gamma_target = float(gamma)
        ev.sampling_sweep.specs.append(spec)
    return cfg

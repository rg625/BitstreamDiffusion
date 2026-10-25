"""CoBit-S LM1B: gamma sweeps on the entropy-rate grid and the DGC grid, one pipeline, for Pareto fronts.

Wraps dgc_schedule_ab.py (same checkpoint, tables and Table 1 sampler settings) for clusters without the
LM1B dataset cache: the test loader is skipped and no real-data pass is run (evaluation.skip_test_loader).

Env vars:
  SCHEDULE   cobit | dgc
  EVAL_SEED  seed (default 42)
  CELLS      "nfe:gamma/nfe:gamma/..." (gamma 0 => deterministic); "/" because sbatch --export splits on commas
  DGC_NUM_SAMPLES  smoke runs only; the sweep uses the inherited 1024
"""
import os

from ml_collections import config_dict

from configs.lm1b.continuous.eval.dgc_schedule_ab import get_config as get_ab_config


def get_config():
    cells = []
    for tok in os.environ.get("CELLS", "256:0.2").replace("/", ",").split(","):
        if tok.strip():
            nfe, gamma = tok.split(":")
            cells.append((int(nfe), float(gamma)))
    os.environ.pop("CHURN_GRID", None)
    cfg = get_ab_config()
    seed = int(os.environ.get("EVAL_SEED", 42))
    schedule = os.environ.get("SCHEDULE", "cobit").strip().lower()

    smoke = f"_smoke{os.environ['DGC_NUM_SAMPLES']}" if os.environ.get("DGC_NUM_SAMPLES") else ""
    part = "_c" + "_".join(f"{n}g{g:g}" for n, g in cells)
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/dgc_sweep/{schedule}/seed{seed}{part}{smoke}"
    cfg.evaluation.samples_dir = f"{cfg.evaluation.out_dir}/samples"
    cfg.evaluation.results_csv = f"{cfg.evaluation.out_dir}/results.csv"
    cfg.evaluation.shared_text_cache_dir = f"{cfg.evaluation.out_dir}/shared_text_cache"
    cfg.evaluation.skip_test_loader = True
    ext = cfg.evaluation.external_ppl
    ext.shared_cache_dir = cfg.evaluation.shared_text_cache_dir
    ext.compute_real_reference = False

    cfg.evaluation.sampling_sweep.target_nfes = sorted({n for n, _ in cells})
    cfg.evaluation.sampling_sweep.specs = []
    for nfe, gamma in cells:   # the spec fields of dgc_schedule_ab.py's add_spec
        spec = config_dict.ConfigDict()
        spec.sampler_name = "ddim_entropic"
        spec.sc_refresh_modes = ["carry"]
        spec.target_nfes = [int(nfe)]
        spec.ati_etas = [0.0]
        spec.stochastic_enabled = gamma > 0
        if gamma > 0:
            spec.s_churn = float(gamma * max(1, nfe - 1))
            spec.s_noise = 1.003
            spec.window_mode = "entropy_cdf"
            spec.entropy_quantile_lo = 0.0
            spec.entropy_quantile_hi = 1.0
            spec.entropy_fallback = "deterministic"
            spec.s_tmin = None
            spec.s_tmax = None
            spec.gamma_target = float(gamma)
        cfg.evaluation.sampling_sweep.specs.append(spec)
    return cfg

"""CoBit OWT (S or M): entropy-rate schedule vs DGC schedule at fixed NFE and gamma.

Inherits the released eval config (eval_750K_seed.py for CoBit-S, eval_cobit_m_750K.py for
CoBit-M) and changes ONLY the sigma grid, by pointing entropy_run_dir at a different table,
as configs/lm1b/continuous/eval/dgc_schedule_ab.py does for LM1B.

Env vars:
  MODEL      s (CoBit-S, 750k) | m (CoBit-M, 750k)
  SCHEDULE   cobit (shipped table, default) | dgc (scripts/dgc/build_dgc_tables.py) |
             dp_dgc (DGC table, finite-NFE DP block allocation, diffusion/continuous/dp_dgc.py)
  DP_DGC_BLOCKS  K for dp_dgc (default 8)
  DP_DGC_WEIGHT  increment (default; the notebook's G_k) | length_squared (the earlier G_k; dir suffix _lensq)
  EVAL_SEED  seed (default 42). Same seed across arms => same initial noise and churn draws.
  CELLS      "nfe:gamma,..." or "nfe:gamma/..." (gamma 0 => deterministic), default the paper's operating points:
             S "256:0.13" (Table 1), M "256:0.21,384:0.24,512:0.26" (Table 2)
  DGC_NUM_SAMPLES  smoke runs only; the A/B itself uses the inherited 1024
  CHURN_SWAP "1": steps from SCHEDULE's table, churn density from the other one

No real-data reference: the OWT test cache is not built on every cluster, so the test loader is
skipped (evaluation.skip_test_loader) and the paper's reference (GenPPL 15.07, entropy 5.44) is
used instead. Generation does not read the loader for unconditional OWT samples.
"""
import os

from ml_collections import config_dict

TABLES = {
    ("s", "cobit"): "assets/entropy_tables/owt",
    ("s", "dgc"): "assets/entropy_tables/owt_dgc",
    ("m", "cobit"): "assets/entropy_tables/owt_medium",
    ("m", "dgc"): "assets/entropy_tables/owt_medium_dgc",
    ("s", "dp_dgc"): "assets/entropy_tables/owt_dgc",          # same DGC table, finite-NFE DP allocation
    ("m", "dp_dgc"): "assets/entropy_tables/owt_medium_dgc",
}
DEFAULT_CELLS = {"s": "256:0.13", "m": "256:0.21,384:0.24,512:0.26"}


def get_config():
    model = os.environ.get("MODEL", "s").strip().lower()
    schedule = os.environ.get("SCHEDULE", "cobit").strip().lower()
    if (model, schedule) not in TABLES:
        raise ValueError(f"MODEL in {{s,m}} and SCHEDULE in {{cobit,dgc}}, got {model!r}, {schedule!r}")
    seed = int(os.environ.get("EVAL_SEED", 42))
    cells = []
    # "/" also separates cells: sbatch --export splits its argument on commas
    for tok in os.environ.get("CELLS", DEFAULT_CELLS[model]).replace("/", ",").split(","):
        if tok.strip():
            nfe, gamma = tok.split(":")
            cells.append((int(nfe), float(gamma)))

    if model == "s":
        from configs.owt.eval_750K_seed import get_config as get_base_config
    else:
        from configs.owt.eval_cobit_m_750K import get_config as get_base_config
    cfg = get_base_config()

    cfg.evaluation.entropy_run_dir = TABLES[(model, schedule)]
    dp_k = int(os.environ.get("DP_DGC_BLOCKS", 8))
    dp_w = os.environ.get("DP_DGC_WEIGHT", "increment")
    cfg.evaluation.dp_dgc_blocks = dp_k
    cfg.evaluation.dp_dgc_weight = dp_w
    # CHURN_SWAP=1: keep this grid's steps but take the churn density from the other table
    # (DDIMSampler reads COBIT_CHURN_TABLE), to separate step placement from churn placement.
    swap = os.environ.get("CHURN_SWAP") == "1"
    if swap:
        os.environ["COBIT_CHURN_TABLE"] = TABLES[(model, "dgc" if schedule == "cobit" else "cobit")]

    # Separate output + sample cache per model, arm and seed so nothing is shared or overwritten.
    smoke = f"_smoke{os.environ['DGC_NUM_SAMPLES']}" if os.environ.get("DGC_NUM_SAMPLES") else ""
    # An explicit CELLS list gets its own directory, so one job per cell never shares a results file.
    part = "_c" + os.environ["CELLS"].replace(":", "g").replace(",", "_").replace("/", "_") if os.environ.get("CELLS") else ""
    arm = (f"dp_dgc_K{dp_k}{'' if dp_w == 'increment' else '_lensq'}" if schedule == "dp_dgc" else schedule)
    cfg.evaluation.out_dir = (f"runs/{cfg.experiment}/dgc_ab/{arm}{'_swapchurn' if swap else ''}"
                              f"/seed{seed}{part}{smoke}")
    cfg.evaluation.samples_dir = f"{cfg.evaluation.out_dir}/samples"
    cfg.evaluation.results_csv = f"{cfg.evaluation.out_dir}/results.csv"
    cfg.evaluation.shared_text_cache_dir = f"{cfg.evaluation.out_dir}/shared_text_cache"
    cfg.evaluation.skip_test_loader = True
    ext = cfg.evaluation.external_ppl
    ext.shared_cache_dir = cfg.evaluation.shared_text_cache_dir
    ext.compute_real_reference = False
    ext.seed = seed
    cfg.train.seed = seed
    if os.environ.get("DGC_NUM_SAMPLES"):
        n = int(os.environ["DGC_NUM_SAMPLES"])
        ext.num_samples = n
        ext.micro_batch_size = min(n, ext.micro_batch_size)

    cfg.evaluation.sampling_sweep.target_nfes = sorted({n for n, _ in cells})
    cfg.evaluation.sampling_sweep.specs = []
    for nfe, gamma in cells:
        spec = config_dict.ConfigDict()
        # "entropic" = inverse CDF of the table; "dp_dgc" = DP block allocation on the same table
        spec.sampler_name = "ddim_dp_dgc" if schedule == "dp_dgc" else "ddim_entropic"
        spec.sc_refresh_modes = ["carry"]
        spec.target_nfes = [int(nfe)]
        spec.ati_etas = [0.0]
        spec.stochastic_enabled = gamma > 0
        if gamma > 0:
            spec.stochastic_mode = "edm_churn"
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

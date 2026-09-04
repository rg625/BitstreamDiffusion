# configs/tasks/tinygsm_bits_probe.py
#
# Throughput probe for the representation (Branch 1) pilot.
#
# Identical to tinygsm_bits.py EXCEPT:
#   - data.max_train_examples caps the corpus so the cache builds from the
#     STREAMING path (no full TinyGSM download). The cache tag encodes the cap
#     (`cap20000`), so this can never be mistaken for the full-corpus cache.
#   - optim.total_steps is tiny: we are measuring steps/sec, not training.
#   - a distinct experiment dir, so it cannot touch any production run.
#
# COST is the whole point of this config: production ran 500k steps on 4xA100
# but throughput was never measured, so every GPU-hour figure quoted for the
# representation and ordering branches is currently a guess. This measures it.
#
# LOSS_TYPE is read from the environment so the SAME config measures both arms:
#   PROBE_LOSS=binary_sm  (production recipe)
#   PROBE_LOSS=binary_ce  (the Branch 1 candidate)
# They should be within noise of each other -- but "should" is what a probe is for.
import importlib.util
import os


def get_config():
    base_path = os.path.join(os.path.dirname(__file__), "tinygsm_bits.py")
    spec = importlib.util.spec_from_file_location("tinygsm_bits_base", base_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.get_config()

    cfg.data.max_train_examples = int(os.environ.get("PROBE_CAP", 20_000))
    cfg.optim.total_steps = int(os.environ.get("PROBE_STEPS", 200))
    cfg.train.loss_type = os.environ.get("PROBE_LOSS", "binary_sm")

    loss_tag = cfg.train.loss_type
    cfg.experiment = f"tasks/tinygsm/probe_{loss_tag}"
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/last.pt"
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/gsm8k_eval"
    return cfg

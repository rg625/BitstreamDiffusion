# configs/tasks/tinygsm_tokens.py
#
# TOKEN-SPACE CoBit on TinyGSM: a V-way softmax over the 49,153-token
# vocabulary at each of the 512 token positions, replacing the per-bit binary
# head.
#
# WHY THIS IS A GENUINELY DIFFERENT OBJECTIVE (unlike a 2-class softmax)
# ---------------------------------------------------------------------
# The binary head predicts 16 INDEPENDENT Bernoullis per token, so it can only
# represent a PRODUCT distribution over a token's bits -- 16 degrees of freedom.
# A V-way categorical represents ANY distribution over tokens -- 49,152 degrees
# of freedom -- including bit correlations the binary head structurally cannot
# express. That extra expressivity is the "added flexibility" being tested.
#
# By contrast a 2-class softmax PER BIT is exactly BCE (proved to 1e-12 in
# tests/test_softmax_equivalence.py) and is deliberately not run.
#
#   TOK_LOSS  = token_ce | token_sm
#   TOK_LR, TOK_SEED, TOK_STEPS, TOK_BATCH, TOK_TAG
import importlib.util
import os


def get_config():
    base = os.path.join(os.path.dirname(__file__), "tinygsm_bits_cfg.py")
    spec = importlib.util.spec_from_file_location("tinygsm_bits_cfg_base", base)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.get_config()          # production recipe: p_uncond=0.1, edm, clamp off

    V = int(cfg.data.token_vocab_size)          # 49153
    n_tok = int(cfg.data.sequence_len_tokens)   # 512

    # ---- representation: one continuous one-hot vector per TOKEN ----
    cfg.data.representation = "tokens"
    cfg.data.vocab_size = V
    cfg.data.sequence_len = n_tok               # positions are tokens, not bits
    cfg.model.patch_size = 1                    # 1 transformer position per token
    cfg.model.out_dim = V                       # V-way head
    # Uniform one-hot mean: the data centre for a V-simplex, not 0.5 as for bits.
    cfg.diffusion.continuous.data_center = 1.0 / V

    loss = os.environ.get("TOK_LOSS", "token_ce")
    if loss not in ("token_ce", "token_sm"):
        raise SystemExit(f"TOK_LOSS must be token_ce|token_sm, got {loss!r}")
    cfg.train.loss_type = loss
    # Chunked loss kernel: a [B,S,V] logits tensor is 26 GB in bf16 at batch 512,
    # so the loss is evaluated in row chunks rather than materialised at once.
    cfg.train.token_sm_chunk_size = int(os.environ.get("TOK_CHUNK", 2048))

    cfg.train.batch_size = int(os.environ.get("TOK_BATCH", cfg.train.batch_size))
    cfg.optim.lr = float(os.environ.get("TOK_LR", cfg.optim.lr))
    cfg.train.seed = int(os.environ.get("TOK_SEED", 42))
    cfg.optim.total_steps = int(os.environ.get("TOK_STEPS", 50000))
    cfg.train.checkpointing.interval.every_steps = int(os.environ.get("TOK_CKPT_EVERY", 25000))
    cfg.train.checkpointing.interval.keep_last = None

    g = type(cfg.train.checkpointing)()
    g.enabled = True; g.factor = 10.0; g.patience = 200
    g.min_steps = 2500; g.ema_decay = 0.99; g.ring_steps = 4000
    cfg.train.divergence_guard = g

    tag = os.environ.get("TOK_TAG", "").strip()
    parts = [p for p in (tag, loss, f"s{cfg.train.seed}") if p]
    cfg.experiment = "tasks/tinygsm/tok_" + "_".join(parts)
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/last.pt"
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/gsm8k_eval"
    return cfg

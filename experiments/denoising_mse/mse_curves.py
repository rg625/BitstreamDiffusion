#!/usr/bin/env python
"""
Expected denoising MSE as a function of the noise level, for CoBit-S (130M) and CoBit-M (462M)
on held-out OpenWebText (cfg.data.valid_split = train[-100000:]).

For a fixed log-spaced grid sigma_1..sigma_G and validation sequences x0_n (16384 bits each),
we estimate, per sequence n and grid point g, the per-bit squared error of the denoiser
D(x, sigma) = sigmoid(logits) against x0, i.e. the unweighted `binary_sm` training loss:

    mse(n, g) = mean_bits (D(x0_n + sigma_g * eps_n, sigma_g) - x0_n)^2

for the three self-conditioning inputs that occur in training (trainers/trainer.py):
    nosc : D(x, s, x0_hat=zeros)                 -- loss on rows without self-conditioning
    sc   : D(x, s, x0_hat=sigmoid(D(x, s, None)))  -- loss on rows with self-conditioning
    none : D(x, s, None)                         -- the SC bootstrap pass (never in the loss)
The trained per-sigma objective is 0.5 * (nosc + sc) (self_condition_prob = 0.5).

Variance reduction: eps_n is one fixed Gaussian draw per sequence (seeded by the sequence index),
reused across the whole sigma grid and across both models (common random numbers). The curves are
therefore smooth in sigma and the S-vs-M difference is a paired estimate; standard errors are
computed across sequences (the i.i.d. unit).

RoPE: by default the rotary angles are computed in float32 (rope_fix.py), matching the compiled
production model; the first runs (tags main, tok) were made without it and are superseded.

Work is split into chunks of `--chunk` sequences, one output file per chunk; finished chunks are
skipped, so the run is resumable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

MODELS = {
    "S": dict(config="configs/owt/eval_750K_seed.py", params="130M"),
    "M": dict(config="configs/owt/eval_cobit_m_750K.py", params="462M"),
}
OUT = REPO / "runs/denoising_mse"
NOISE_SEED = 20261003
SIG_MIN, SIG_MAX = 0.002, 80.0


def sigma_grid(G: int) -> np.ndarray:
    return np.exp(np.linspace(math.log(SIG_MIN), math.log(SIG_MAX), G))


def seq_indices(n_val: int, N: int) -> np.ndarray:
    """N sequence indices spread evenly over the validation set (same for both models)."""
    return np.unique(np.linspace(0, n_val - 1, N).round().astype(np.int64))


def noise_for(idx: torch.Tensor, S: int, device) -> torch.Tensor:
    """Per-sequence fixed Gaussian noise, a function of the sequence index only."""
    out = torch.empty(len(idx), S, device=device)
    g = torch.Generator(device=device)
    for r, i in enumerate(idx.tolist()):
        g.manual_seed(NOISE_SEED * 1_000_003 + int(i))
        out[r].normal_(generator=g)
    return out


class ValBits:
    """The gpt2id_bpe16 val code cache as bit sequences, exactly as OpenWebTextDataset.__getitem__
    returns them, but without loading the code tokenizer (its JSON needs tokenizers>=0.20, newer
    than this env has; the cache itself was built by prep_val.py)."""

    def __init__(self, cfg):
        from data.openwebtext import _get_token_to_bits_table

        root = REPO / cfg.data.root
        base_len = int(getattr(cfg.data, "base_sequence_len_tokens", cfg.data.sequence_len_tokens))
        name = (f"cache_val_{cfg.data.tokenizer_name}_gpt2id_bpe16_v65536_"
                f"base{base_len}_len{cfg.data.sequence_len_tokens}")
        self.meta = json.loads((root / f"{name}.meta.json").read_text())
        assert self.meta["source_flat_hf_split"] == getattr(cfg.data, "valid_split", "train[-100000:]")
        self.mm = np.memmap(root / f"{name}.uint16", dtype=np.uint16, mode="r",
                            shape=(self.meta["num_sequences"], self.meta["code_sequence_len_tokens"]))
        assert cfg.data.binarization == "raw_binary"
        self.table = _get_token_to_bits_table(
            root=root, vocab_size=int(self.meta["code_vocab_size"]),
            data_bits_per_token=int(cfg.data.bits_per_token), binarization="raw_binary",
            old_to_new=None, ecc_enabled=False, ecc_total_bits_per_token=int(cfg.data.bits_per_token),
            ecc_include_overall_parity=True, tokenizer_name=cfg.data.tokenizer_name)

    def __len__(self) -> int:
        return int(self.mm.shape[0])

    def __getitem__(self, i: int) -> torch.Tensor:
        toks = torch.from_numpy(np.asarray(self.mm[i]).astype(np.int64))
        return self.table[toks].view(-1)


def load(model_key: str, device):
    from diffusion.continuous.logit_postprocess import _model_logits_continuous
    from evaluation.utils import load_checkpoint, load_config, unwrap_all
    from models import create_model
    from utils.ema import EMA

    cfg = load_config(str(REPO / MODELS[model_key]["config"]))
    cfg.device = str(device)
    model = create_model(cfg).to(device)
    ema = EMA(unwrap_all(model), decay=0.0)
    load_checkpoint(model, ema, REPO / cfg.evaluation.checkpoint_path, device, apply_ema=True)
    model.eval()
    n_params = sum(p.numel() for p in unwrap_all(model).parameters())
    ds = ValBits(cfg)

    def logits_fn(x, sigma, x0_hat):
        return _model_logits_continuous(model, cfg, x, sigma, x0_hat)

    return cfg, model, logits_fn, ds, n_params


SEQ_METRICS = ("mse", "mseC", "berC", "terC", "tnllC")
VARS = ("nosc", "sc", "none")


@torch.no_grad()
def run_chunk(logits_fn, x0, eps, sigmas, amp_dtype, bits_per_token):
    """Per-sequence metrics, dict f"{metric}_{variant}" -> [G, B]:
      mse   per-bit squared error of D = sigmoid(logits), all bits (the binary_sm training loss)
      mseC  same, content (non-pad) tokens only
      berC  hard bit error rate (round D) on content bits
      terC  token error rate on content tokens: token decoded by rounding D has any bit wrong
      tnllC per-content-token NLL [nats] of the true token under the factorised bit posterior D
    plus position sums: f"pos_{v}" [G, bpt] per-bit-in-token MSE summed over sequences, and
    f"tokpos_{v}" [G, T] token errors summed over sequences per token position (content only),
    with "tokpos_count" [T] the number of content tokens per position."""
    B, S = x0.shape
    G = len(sigmas)
    T = S // bits_per_token
    out = {f"{m}_{v}": np.zeros((G, B), np.float64) for m in SEQ_METRICS for v in VARS}
    pos = {f"pos_{v}": np.zeros((G, bits_per_token), np.float64) for v in VARS}
    tokpos = {f"tokpos_{v}": np.zeros((G, T), np.float64) for v in VARS}
    zeros = torch.zeros_like(x0)
    # pad_id = 0 in the gpt2id_bpe16 codec, i.e. an all-zero 16-bit token
    xt = x0.view(B, T, bits_per_token)
    content = (xt.sum(-1) > 0).float()                                       # [B, T]
    n_tok = content.sum(1).clamp_min(1.0)                                    # [B]
    n_bits = n_tok * bits_per_token
    np_ = lambda t: t.double().cpu().numpy()  # noqa: E731

    def record(v, g, logits):
        lg = logits.float().view(B, T, bits_per_token)
        p = torch.sigmoid(lg)
        se = (p - xt) ** 2
        wrong_bit = ((p > 0.5).float() != xt).float()
        tok_wrong = (wrong_bit.sum(-1) > 0).float() * content                # [B, T]
        bce = torch.nn.functional.binary_cross_entropy_with_logits(lg, xt, reduction="none").sum(-1)
        out[f"mse_{v}"][g] = np_(se.mean((1, 2)))
        out[f"mseC_{v}"][g] = np_((se.sum(-1) * content).sum(1) / n_bits)
        out[f"berC_{v}"][g] = np_((wrong_bit.sum(-1) * content).sum(1) / n_bits)
        out[f"terC_{v}"][g] = np_(tok_wrong.sum(1) / n_tok)
        out[f"tnllC_{v}"][g] = np_((bce * content).sum(1) / n_tok)
        pos[f"pos_{v}"][g] = np_(se.mean(1).sum(0))
        tokpos[f"tokpos_{v}"][g] = np_(tok_wrong.sum(0))

    for g, s in enumerate(sigmas):
        sigma = torch.full((B,), float(s), device=x0.device)
        x = x0 + float(s) * eps
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            l_none = logits_fn(x, sigma, None)
            p_none = torch.sigmoid(l_none.float().view(B, S)).to(x.dtype)
            l_sc = logits_fn(x, sigma, p_none)
            l_nosc = logits_fn(x, sigma, zeros)
        record("none", g, l_none)
        record("sc", g, l_sc)
        record("nosc", g, l_nosc)
    return out, pos, tokpos, np_(content.sum(0))


def cmd_run(args) -> None:
    if not args.no_rope_fix:  # float32 RoPE angles = the compiled production numerics (see rope_fix.py)
        import rope_fix

        rope_fix.apply()
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg, model, logits_fn, ds, n_params = load(args.model, device)
    amp_dtype = torch.bfloat16 if str(cfg.evaluation.amp_dtype) == "bf16" else torch.float16
    bpt = int(cfg.data.bits_per_token)
    sigmas = sigma_grid(args.grid)
    idx_all = seq_indices(len(ds), args.n_seq)
    od = OUT / f"{args.tag}" / args.model
    od.mkdir(parents=True, exist_ok=True)
    meta = dict(model=args.model, params=MODELS[args.model]["params"], n_params=int(n_params),
                config=MODELS[args.model]["config"], checkpoint=str(cfg.evaluation.checkpoint_path),
                entropy_run_dir=str(cfg.evaluation.entropy_run_dir), amp_dtype=str(amp_dtype),
                n_val_sequences=len(ds), n_seq=len(idx_all), grid=args.grid, sigma_min=SIG_MIN,
                sigma_max=SIG_MAX, noise_seed=NOISE_SEED, chunk=args.chunk, batch=args.batch,
                seq_len_bits=int(cfg.data.sequence_len), bits_per_token=bpt,
                sigma_data=float(cfg.diffusion.continuous.sigma_data), rope_fix=not args.no_rope_fix)
    (od / "meta.json").write_text(json.dumps(meta, indent=2))
    chunks = [idx_all[a:a + args.chunk] for a in range(0, len(idx_all), args.chunk)]
    if args.max_chunks is not None:
        chunks = chunks[: args.max_chunks]
    print(f"[run] model {args.model} ({n_params / 1e6:.1f}M params), {len(idx_all)} sequences of "
          f"{len(ds)} val, grid {args.grid}, {len(chunks)} chunks", flush=True)
    for k, idx in enumerate(chunks):
        f = od / f"chunk{k:04d}.npz"
        if f.exists() and not args.force:
            continue
        t0 = time.perf_counter()
        seq, sums, count = {}, {}, 0.0
        for a in range(0, len(idx), args.batch):
            bi = torch.as_tensor(idx[a:a + args.batch])
            x0 = torch.stack([ds[int(i)] for i in bi]).to(device, torch.float32)
            eps = noise_for(bi, x0.size(1), device)
            o, p, tp, cnt = run_chunk(logits_fn, x0, eps, sigmas, amp_dtype, bpt)
            for key, val in o.items():
                seq.setdefault(key, []).append(val)
            for key, val in {**p, **tp}.items():
                sums[key] = sums.get(key, 0.0) + val
            count = count + cnt
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tmp = od / f"chunk{k:04d}.tmp.npz"
        np.savez(tmp, idx=idx, sigmas=sigmas, wall_sec=dt, tokpos_count=count,
                 **{key: np.concatenate(val, 1) for key, val in seq.items()},
                 **{key: val / len(idx) for key, val in sums.items() if key.startswith("pos_")},
                 **{key: val for key, val in sums.items() if key.startswith("tokpos_")})
        os.replace(tmp, f)
        print(f"[run] {args.model} chunk {k + 1}/{len(chunks)}: {len(idx)} seqs in {dt:.1f}s "
              f"({dt / len(idx) / args.grid / 3 * 1e3:.2f} ms/seq/forward)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--n-seq", type=int, default=4096)
    ap.add_argument("--grid", type=int, default=129)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-rope-fix", action="store_true",
                    help="eager bf16 RoPE (positions > 256 collapse); used by the first, superseded runs")
    cmd_run(ap.parse_args())


if __name__ == "__main__":
    main()

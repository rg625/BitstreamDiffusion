#!/usr/bin/env python
"""
GenPPL (GPT-2-large) and GPT-2-token unigram entropy of the final samples saved by trajectories.py,
using the repo's evaluation path unchanged:
  texts   = utils.text_decode.decode_bitstreams_for_eval(cfg, bits, mode="full")
  GenPPL  = evaluation.external_perplexity.HFExternalPerplexityEvaluator.score_texts(texts)
  entropy = mean per-sample token_unigram_entropy of the cleaned GPT-2 token rows
            (as evaluation/compute_entropy_from_caches.py does for the gpt2id_bpe16 codec)
Per-sample NLLs are kept, so CoBit-S and CoBit-M can be compared pairwise (same seeds per example).

Needs tokenizers>=0.20 for the code tokenizer, so it is run outside the `sedd` env, e.g. a venv on
top of the `samudra` env plus ml_collections:
  HF_HOME=~/hf_cache HF_HUB_OFFLINE=1 <venv>/bin/python experiments/denoising_mse/score_traj.py --tag main
Output: runs/denoising_mse/traj/<tag>/scores.json and scores_per_sample.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    from evaluation.external_perplexity import HFExternalPerplexityEvaluator
    from evaluation.text_metrics import token_unigram_entropy_from_token_ids
    from evaluation.utils import load_config
    from utils.text_decode import (bitstreams_to_token_ids_raw_binary, code_ids_to_gpt2_token_id_lists_for_eval,
                                   decode_bitstreams_for_eval, load_openwebtext_gpt2id_bpe16_assets)

    od = REPO / f"runs/denoising_mse/traj/{args.tag}"
    zs = [np.load(f) for f in sorted(glob.glob(str(od / "chunk*.npz")))]
    cfg = load_config(str(REPO / "configs/owt/eval_750K_seed.py"))  # data/codec settings identical for S and M
    ext = cfg.evaluation.external_ppl
    dev = torch.device("cuda:0")
    try:
        scorer = HFExternalPerplexityEvaluator(model_name=str(ext.hf_model_name), device=dev,
                                               torch_dtype=str(ext.hf_dtype),
                                               attn_implementation=str(ext.attn_implementation), use_amp=True)
    except ValueError:
        scorer = HFExternalPerplexityEvaluator(model_name=str(ext.hf_model_name), device=dev,
                                               torch_dtype=str(ext.hf_dtype), attn_implementation="eager",
                                               use_amp=True)
    gpt2_tok, code_tok, code_meta = load_openwebtext_gpt2id_bpe16_assets(
        root=REPO / cfg.data.root, code_tokenizer_path=str(cfg.data.code_tokenizer_path),
        code_tokenizer_meta_path=cfg.data.code_tokenizer_meta_path, tokenizer_name=str(cfg.data.tokenizer_name))

    out, per = {}, {}
    keys = [k for k in zs[0].files if k.startswith("bits_")]
    for k in keys:
        bits = torch.from_numpy(np.concatenate([z[k] for z in zs], 0).astype(np.int64))
        texts = decode_bitstreams_for_eval(cfg, bits, mode="full")
        ps = scorer.score_texts_per_sample(texts)
        nll = np.array([p[0] for p in ps], float)
        ntok = np.array([p[1] for p in ps], float)
        code_ids = bitstreams_to_token_ids_raw_binary(bits, bits_per_token=int(cfg.data.bits_per_token), cfg=cfg)
        rows = code_ids_to_gpt2_token_id_lists_for_eval(code_ids, gpt2_tokenizer=gpt2_tok, code_tokenizer=code_tok,
                                                       code_meta=code_meta, return_debug_stats=False)
        ent = np.array([token_unigram_entropy_from_token_ids(torch.tensor(r, dtype=torch.long)) for r in rows])
        name = k[len("bits_"):]
        out[name] = dict(n=len(texts), genppl=float(math.exp(nll.sum() / ntok.sum())),
                         mean_entropy=float(ent.mean()), entropy_se=float(ent.std(ddof=1) / math.sqrt(len(ent))))
        per[f"nll_{name}"], per[f"ntok_{name}"], per[f"entropy_{name}"] = nll, ntok, ent
        print(f"[score] {name}: n={len(texts)} GenPPL {out[name]['genppl']:.2f}  entropy "
              f"{out[name]['mean_entropy']:.3f} ± {out[name]['entropy_se']:.3f}", flush=True)
    if "S" in out and "M" in out:
        d = per["nll_S"] / np.maximum(per["ntok_S"], 1) - per["nll_M"] / np.maximum(per["ntok_M"], 1)
        out["paired"] = dict(mean_nll_per_tok_S_minus_M=float(d.mean()),
                             se=float(d.std(ddof=1) / math.sqrt(len(d))), frac_M_lower_nll=float((d > 0).mean()))
        print(f"[score] paired per-sample NLL/token S - M: {d.mean():.4f} ± {d.std(ddof=1) / math.sqrt(len(d)):.4f}; "
              f"M lower on {100 * (d > 0).mean():.1f}% of examples", flush=True)
    (od / "scores.json").write_text(json.dumps(out, indent=2))
    np.savez(od / "scores_per_sample.npz", **per)


if __name__ == "__main__":
    main()

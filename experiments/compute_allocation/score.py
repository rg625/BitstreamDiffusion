"""Per-sample quality for LM1B generations: GPT-2-large NLL and unigram entropy.

Matches `HFExternalPerplexityEvaluator.score_texts` (score_mode="full"): each
text is tokenized without special tokens, tokens 1..L-1 are scored, and GenPPL
is exp(sum NLL / sum scored tokens). The only difference is batching (right
padding, causal model, so real positions never see the pads); `anchor` in
run.py checks the two agree on the reference samples.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from utils.text_decode import decode_bitstreams_for_eval, decode_bitstreams_to_token_ids_for_eval
from evaluation.text_metrics import token_unigram_entropy_from_token_ids

GPT2 = "openai-community/gpt2-large"


def decode(cfg, bits_u8: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
    bits_u8 = bits_u8.to(torch.uint8).cpu()
    texts = decode_bitstreams_for_eval(
        cfg, bits_u8, prompt_len_bits=torch.zeros(bits_u8.size(0), dtype=torch.long),
        mode="full", dataset_obj=None,
    )
    ids = decode_bitstreams_to_token_ids_for_eval(cfg, bits_u8, dataset_obj=None)
    return texts, ids


def unigram_entropy(ids: torch.Tensor) -> np.ndarray:
    return np.array([token_unigram_entropy_from_token_ids(r) for r in ids], dtype=np.float64)


class BatchedScorer:
    def __init__(self, device, model_name: str = GPT2):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to(device).eval()
        self.max_len = int(self.model.config.max_position_embeddings)
        self.eos = int(self.tok.eos_token_id)

    @torch.no_grad()
    def score(self, texts: List[str], batch_size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """Return (nll_sum [N] nats, n_scored [N]); NaN/0 for texts under 2 tokens."""
        enc = [self.tok(t, add_special_tokens=False)["input_ids"] if t else [] for t in texts]
        nll = np.full(len(texts), np.nan)
        cnt = np.zeros(len(texts), dtype=np.int64)
        order = sorted(range(len(texts)), key=lambda i: len(enc[i]))
        for lo in range(0, len(order), batch_size):
            idx = [i for i in order[lo:lo + batch_size] if len(enc[i]) >= 2]
            if not idx:
                continue
            L = max(len(enc[i]) for i in idx)
            if L > self.max_len:
                raise ValueError(f"text of {L} tokens exceeds GPT-2 context {self.max_len}")
            ids = torch.full((len(idx), L), self.eos, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                ids[r, : len(enc[i])] = torch.tensor(enc[i])
                att[r, : len(enc[i])] = 1
            ids, att = ids.to(self.device), att.to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                enabled=self.device.type == "cuda"):
                logits = self.model(input_ids=ids, attention_mask=att).logits
            ce = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.size(-1)),
                ids[:, 1:].reshape(-1), reduction="none",
            ).view(len(idx), L - 1)
            m = att[:, 1:].float()
            s = (ce * m).sum(1).double().cpu().numpy()
            c = m.sum(1).long().cpu().numpy()
            for r, i in enumerate(idx):
                nll[i], cnt[i] = s[r], c[r]
        return nll, cnt

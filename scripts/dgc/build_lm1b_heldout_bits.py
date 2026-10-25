"""Held-out LM1B clean sequences in the CoBit-S bit coordinates, for the forward profile (CPU).

    python -I scripts/dgc/build_lm1b_heldout_bits.py --parquet TEST.parquet --out lm1b_test_bits.pt [--n_blocks 8192]

Reproduces the training pipeline on the LM1B *test* split: lm1b_detokenizer (scripts/lm1b/download_lm1b.py),
bert-base-uncased without special tokens, [SEP] between examples, 128-token blocks with the remainder dropped
(scripts/lm1b/build_lm1b_bert_caches.py), then the raw_binary 15-bit table of data/lm1b.py. Writes a dict with
x0 (uint8 [M, 1920]) and ids (int64 [M, 128]).
"""
import argparse
import hashlib
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.lm1b import _get_token_to_bits_table  # noqa: E402
from scripts.lm1b.download_lm1b import lm1b_detokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_blocks", type=int, default=8192)
    ap.add_argument("--seq_tokens", type=int, default=128)
    ap.add_argument("--bits", type=int, default=15)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    texts = pq.read_table(a.parquet, columns=["text"]).column("text").to_pylist()
    blocks, buf, seen = [], [], False
    for i in range(0, len(texts), 1024):
        lines = [t for t in (lm1b_detokenizer(x).strip() for x in texts[i:i + 1024]) if t]
        for ids in tok(lines, add_special_tokens=False)["input_ids"]:
            if not ids:
                continue
            if seen:
                buf.append(tok.sep_token_id)
            buf.extend(ids)
            seen = True
            while len(buf) >= a.seq_tokens:
                blocks.append(buf[:a.seq_tokens])
                del buf[:a.seq_tokens]
        if len(blocks) >= a.n_blocks:
            break
    ids = torch.tensor(blocks[:a.n_blocks], dtype=torch.long)
    if len(ids) < a.n_blocks:
        raise SystemExit(f"only {len(ids)} blocks")
    table = _get_token_to_bits_table(root=Path("datasets/lm1b"), vocab_size=tok.vocab_size, data_bits_per_token=a.bits,
                                     binarization="raw_binary", old_to_new=None, ecc_enabled=False,
                                     ecc_total_bits_per_token=a.bits, ecc_include_overall_parity=True)
    x0 = table[ids].reshape(len(ids), -1).to(torch.uint8)
    src = hashlib.sha256(Path(a.parquet).read_bytes()).hexdigest()
    torch.save(dict(x0=x0, ids=ids, source=a.parquet, source_sha256=src, split="test"), a.out)
    print(f"wrote {a.out}: x0 {tuple(x0.shape)}, ids {tuple(ids.shape)}, mean bit {x0.float().mean():.4f}")
    print("first block:", tok.decode(ids[0][:40]))


if __name__ == "__main__":
    main()

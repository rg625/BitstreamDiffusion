"""Held-out OpenWebText clean sequences in the CoBit-S bit coordinates, for the forward profile (CPU).

    python -I scripts/dgc/build_owt_heldout_bits.py --parquet LAST_SHARD.parquet --skip_rows 172 --out owt_bits.pt
        [--n_seq 4096]

Reproduces data/openwebtext.py on the held-out split train[-100000:] (the last 100k documents; with the HF parquet
conversion's 100172-row last shard, rows 172 onwards): GPT-2 ids of the stripped text plus EOS, one flat stream,
1022-token chunks, BOS + chunk + EOS through the gpt2id_bpe16 code tokenizer, [EOSEQ] appended, truncated or [PAD]ded
to 1024 codes, then the raw_binary 16-bit table. Writes a dict with x0 (uint8 [M, 16384]) and ids (int64 [M, 1024]).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.openwebtext import _get_token_to_bits_table, load_gpt2id_bpe16_tokenizer_and_meta  # noqa: E402

ROOT = Path("datasets/openwebtext_gpt2_trainm100k")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--skip_rows", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_seq", type=int, default=4096)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    gtok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    eos = int(gtok.eos_token_id)
    ctok, meta = load_gpt2id_bpe16_tokenizer_and_meta(ROOT, "tokenizer_gpt2id_bpe16_65536_base1024.json",
                                                      "tokenizer_gpt2id_bpe16_65536_base1024.meta.json")
    off, content, L = int(meta["base_char_offset"]), int(meta["content_len_tokens"]), 1024
    pad, eoseq = int(meta["pad_id"]), int(meta["eoseq_id"])
    bos_ch, eos_ch = chr(off + int(meta["bos_token_id_gpt2"])), chr(off + int(meta["eos_token_id_gpt2"]))

    flat, rows, seen = [], [], 0
    for rb in pq.ParquetFile(a.parquet).iter_batches(batch_size=256, columns=["text"]):
        texts = rb.column(0).to_pylist()
        skip = max(0, min(len(texts), a.skip_rows - seen))
        seen += len(texts)
        batch = [(t or "").replace("\r\n", "\n").replace("\r", "\n").strip() for t in texts[skip:]]
        for ids in gtok(batch, add_special_tokens=False)["input_ids"]:
            if ids:
                flat.extend(ids + [eos])
        while len(flat) >= content and len(rows) < a.n_seq:
            body = "".join(chr(off + t) for t in flat[:content])
            del flat[:content]
            ids = list(ctok.encode(bos_ch + body + eos_ch).ids) + [eoseq]
            if len(ids) > L:
                ids = ids[:L]
                ids[-1] = eoseq
            rows.append(ids + [pad] * (L - len(ids)))
        if len(rows) >= a.n_seq:
            break
    if len(rows) < a.n_seq:
        raise SystemExit(f"only {len(rows)} sequences")
    ids = torch.tensor(rows, dtype=torch.long)
    table = _get_token_to_bits_table(root=ROOT, vocab_size=int(meta["actual_code_vocab_size"]), data_bits_per_token=16,
                                     binarization="raw_binary", old_to_new=None, ecc_enabled=False,
                                     ecc_total_bits_per_token=16, ecc_include_overall_parity=True,
                                     tokenizer_name="gpt2")
    x0 = table.to(torch.uint8)[ids].reshape(len(ids), -1)
    h = hashlib.sha256()
    with open(a.parquet, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    src = h.hexdigest()
    torch.save(dict(x0=x0, ids=ids, source=a.parquet, source_sha256=src, skip_rows=a.skip_rows,
                    split="train[-100000:]"), a.out)
    n_pad = (ids == pad).float().mean()
    print(f"wrote {a.out}: x0 {tuple(x0.shape)}, mean bit {x0.float().mean():.4f}, pad fraction {n_pad:.3f}")
    first = [int(t) for t in ids[0] if int(t) not in (pad, eoseq)]
    print("first sequence:", json.dumps(gtok.decode([ord(c) - off for c in ctok.decode(first[:40])
                                                      if ord(c) >= off])[:300]))


if __name__ == "__main__":
    main()

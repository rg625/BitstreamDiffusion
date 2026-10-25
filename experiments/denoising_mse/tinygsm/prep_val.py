#!/usr/bin/env python
"""
Rebuild the TinyGSM validation split exactly as the training code does (origin/tasks/sudoku-tinygsm,
data/tinygsm.py::_build_tinygsm_cache, full-corpus path), without building the 11.7M-row training cache:

  rows       = TinyGSM/TinyGSM train in file order (17 parquet shards, 11,846,109 rows)
  validation = rng(42).permutation(N)[: int(0.01 N)]  (val_ratio 0.01, val_seed 42)
  example    = [BOS] question.strip() \\n code.strip() [EOS] in SmolLM-135M ids (+[PAD]), padded to 512;
               dropped if longer than 512 tokens (filter_too_long); prompt length = 1 + len(q) + len(\\n)
Output: datasets/tinygsm/val_only/{val_ids.uint16, val_plen.int32, val_alen.int32, meta.json}
        (alen = answer tokens incl. EOS)
"""
from __future__ import annotations

import glob
import json

import numpy as np
import pyarrow.parquet as pq

import common as C


def main() -> None:
    files = sorted(glob.glob(str(C.REPO / "datasets/tinygsm/hf_parquet/data/train-*.parquet")))
    N = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    assert N == 11_846_109, N
    n_val = max(1, int(N * 0.01))
    rng = np.random.default_rng(42)
    is_val = np.zeros(N, dtype=bool)
    is_val[rng.permutation(N)[:n_val]] = True

    tok = C.get_tokenizer()
    bos, eos, pad = C.special_ids(tok)
    sep = tok(C.SEP, add_special_tokens=False).input_ids
    ids = np.full((n_val, C.BLOCK), pad, dtype=np.uint16)
    plen = np.zeros(n_val, dtype=np.int32)
    alen = np.zeros(n_val, dtype=np.int32)
    src = np.zeros(n_val, dtype=np.int64)
    w, g0, dropped = 0, 0, 0
    for f in files:
        t = pq.read_table(f, columns=["question", "code"])
        n = t.num_rows
        sel = np.nonzero(is_val[g0:g0 + n])[0]
        if len(sel):
            q = t.column("question").take(sel).to_pylist()
            a = t.column("code").take(sel).to_pylist()
            qe = tok([x.strip() for x in q], add_special_tokens=False)["input_ids"]
            ae = tok([x.strip() for x in a], add_special_tokens=False)["input_ids"]
            for j, (qi, ai) in enumerate(zip(qe, ae)):
                total = 2 + len(qi) + len(sep) + len(ai)
                if total > C.BLOCK:
                    dropped += 1
                    continue
                row = [bos] + qi + sep + ai + [eos]
                ids[w, :len(row)] = row
                plen[w] = 1 + len(qi) + len(sep)
                alen[w] = len(ai) + 1
                src[w] = g0 + sel[j]
                w += 1
        g0 += n
        print(f"[prep] {f.split('/')[-1]}: {len(sel)} val rows, kept so far {w}", flush=True)
    C.VAL_DIR.mkdir(parents=True, exist_ok=True)
    ids[:w].tofile(C.VAL_DIR / "val_ids.uint16")
    plen[:w].tofile(C.VAL_DIR / "val_plen.int32")
    alen[:w].tofile(C.VAL_DIR / "val_alen.int32")
    np.save(C.VAL_DIR / "val_source_row.npy", src[:w])
    meta = dict(n=int(w), n_val_drawn=int(n_val), dropped_too_long=int(dropped), N=int(N), val_ratio=0.01,
                val_seed=42, tokenizer=C.TOKENIZER, tokenizer_len=int(len(tok)), bos=bos, eos=eos, pad=pad,
                block=C.BLOCK, mean_prompt_len=float(plen[:w].mean()), mean_answer_len=float(alen[:w].mean()))
    (C.VAL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Build the OWT validation caches (cfg.data.valid_split = "train[-100000:]") without downloading
the full 40 GB OpenWebText dataset.

The HF parquet conversion of Skylion007/openwebtext (the dataset `load_dataset("openwebtext")`
resolves to) has 80 shards of ~100k rows; the last 100,000 rows live in shards 78-79, which are
downloaded to datasets/openwebtext_gpt2_trainm100k/raw_parquet/. We patch `datasets.load_dataset`
to return those rows and then run the repo's own cache builders unchanged
(data.openwebtext.ensure_openwebtext_caches_ready), so tokenization, EOS insertion, flattening and
the gpt2id_bpe16 second-stage encoding are byte-identical to the training-time pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import datasets  # noqa: E402

from evaluation.utils import load_config  # noqa: E402

SHARDS = [78, 79]


def main() -> None:
    cfg = load_config(str(REPO / "configs/owt/rate_bits_edm_weight.py"))
    root = REPO / cfg.data.root
    assert cfg.data.valid_split == "train[-100000:]", cfg.data.valid_split
    files = [str(root / "raw_parquet" / f"{s}.parquet") for s in SHARDS]
    tail = datasets.load_dataset("parquet", data_files=files, split="train",
                                 cache_dir=str(root / "hf_cache_tail"))
    print(f"[prep] shards {SHARDS}: {len(tail):,} rows")
    assert len(tail) >= 100_000
    val = tail.select(range(len(tail) - 100_000, len(tail)))

    real_load = datasets.load_dataset

    def fake_load(name, *a, split=None, **kw):
        if name == "openwebtext" and split == "train[-100000:]":
            return val
        raise RuntimeError(f"unexpected load_dataset({name!r}, split={split!r}) in prep_val")

    datasets.load_dataset = fake_load
    try:
        from data.openwebtext import ensure_openwebtext_caches_ready

        cfg.data.cache_num_proc = 16
        ensure_openwebtext_caches_ready(cfg, split="val")
    finally:
        datasets.load_dataset = real_load


if __name__ == "__main__":
    main()

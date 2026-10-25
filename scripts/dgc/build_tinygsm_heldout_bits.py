"""Held-out TinyGSM sequences in the CoBit bit coordinates, with the generated-coordinate mask, for the forward
profile of prompted checkpoints (CPU).

    python scripts/dgc/build_tinygsm_heldout_bits.py --config configs/tasks/tinygsm_bits.py --out tinygsm_val_bits.pt
        [--n_seq 2048]

The validation split of data/tinygsm.py (held out from training: val_ratio / val_seed of the config): x0 = the
training bits, gen_mask = everything after the prompt (answer and the trained-on padding), as in training and as the
GSM8K sampler clamps the prompt. Writes a dict with x0 (uint8 [M, L*bpt]), gen_mask (bool) and ids.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.utils import load_config  # noqa: E402
from data.tinygsm import TinyGSMDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_seq", type=int, default=2048)
    a = ap.parse_args()
    cfg = load_config(a.config)
    ds = TinyGSMDataset(cfg, split="val")
    n = min(a.n_seq, len(ds))
    items = [ds[i] for i in range(n)]
    x0 = torch.stack([it["x0"] for it in items]).to(torch.uint8)
    gen = ~torch.stack([it["prefix_mask"] for it in items]).bool()
    ids = torch.stack([it["input_ids"] for it in items])
    torch.save(dict(x0=x0, gen_mask=gen, ids=ids, split="validation", config=a.config), a.out)
    print(f"wrote {a.out}: x0 {tuple(x0.shape)} of {len(ds)} val rows, generated fraction {gen.float().mean():.3f}, "
          f"mean bit {x0.float().mean():.4f}")


if __name__ == "__main__":
    main()

"""
Shared pieces for the TinyGSM -> GSM8K version of the denoising-error / trajectory study.

TinyGSM is a CONDITIONAL task (branch origin/tasks/sudoku-tinygsm, configs/tasks/tinygsm_bits.py):
  example = [BOS] question \\n code [EOS], padded to 512 SmolLM-135M tokens (+[PAD]), 16 bits/token
  (8192 bits, MSB-first, same convention as OWT). The prompt [BOS] question \\n is kept clean (clamped)
  and excluded from the loss; noise and loss cover answer + EOS + pad.
Model: identical architecture to OWT CoBit-S; sigma_data = 0.39984 from the run's sigma_data.json
(training overwrote the config's 0.5); training used torch.compile, so RoPE runs in float32
(../rope_fix.py) unless --no-rope-fix.

Run directories (config.json, original_config.py, entropy_*.pt, sigma_data.json) live under
runs/tasks/tinygsm/<run>/ (downloaded from the shared Drive folder); checkpoints under assets/tinyGSM/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent))

MODELS = {
    "S": dict(run_dir="runs/tasks/tinygsm/cobit_raw_binary_bits", ckpt="assets/tinyGSM/step=000250000.pt",
              params="130M"),
    "M": dict(run_dir="assets/cobit_m_bits_s42", ckpt="assets/cobit_m_bits_s42/checkpoints/step=000250000.pt",
              params="462M", config="config.json"),  # uploaded 6 Oct 2026; SHA256 verified against SHA256SUMS_250k
    # (its original_config.py composes sibling config files that were not uploaded, so the run's flattened
    #  config.json is used; it holds the same resolved values)
}
TOKENIZER = "HuggingFaceTB/SmolLM-135M"
SEP = "\n"
BLOCK = 512
BPT = 16
VAL_DIR = REPO / "datasets/tinygsm/val_only"
GSM8K_TEST = REPO / "datasets/gsm8k/gsm8k_test.json"


def numpy2_pickle_shim():
    """Checkpoints were pickled with numpy 2.x (numpy._core); alias it for numpy 1.x envs."""
    import numpy.core
    import numpy.core.multiarray

    sys.modules.setdefault("numpy._core", numpy.core)
    sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)


def get_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    if tok.pad_token_id is None:  # S-FLM behaviour, as in the training code
        tok.add_special_tokens({"pad_token": "[PAD]"})
    return tok


def special_ids(tok):
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    eos = tok.eos_token_id if tok.eos_token_id is not None else bos
    return int(bos), int(eos), int(tok.pad_token_id)


def ids_to_bits(ids: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(BPT - 1, -1, -1, device=ids.device, dtype=torch.long)
    return ((ids.long().unsqueeze(-1) >> shifts) & 1).reshape(*ids.shape[:-1], ids.shape[-1] * BPT)


def bits_to_ids(bits: torch.Tensor) -> torch.Tensor:
    b = bits.long().reshape(*bits.shape[:-1], -1, BPT)
    powers = 2 ** torch.arange(BPT - 1, -1, -1, device=bits.device, dtype=torch.long)
    return (b * powers).sum(-1)


def tok_mask_to_bits(m: torch.Tensor) -> torch.Tensor:
    return m.unsqueeze(-1).expand(*m.shape, BPT).reshape(*m.shape[:-1], m.shape[-1] * BPT)


def load_cfg(model_key: str, *, rope_fix: bool = True):
    from evaluation.utils import load_config

    spec = MODELS[model_key]
    run = REPO / spec["run_dir"]
    cfg_file = run / spec.get("config", "original_config.py")
    if cfg_file.suffix == ".json":
        from ml_collections import config_dict

        cfg = config_dict.ConfigDict(json.loads(cfg_file.read_text()))
    else:
        cfg = load_config(str(cfg_file))
    sd = json.loads((run / "sigma_data.json").read_text())["sigma_data"]
    cfg.diffusion.continuous.sigma_data = float(sd)  # the value the model was trained with
    cfg.evaluation.checkpoint_path = str(REPO / spec["ckpt"])
    cfg.evaluation.entropy_run_dir = str(run)
    cfg.device = "cuda"
    if rope_fix:
        import rope_fix as RF

        RF.apply()
    return cfg


def load_model(model_key: str, device, *, rope_fix: bool = True):
    """(cfg, eval-mode model with EMA weights applied)."""
    from evaluation.utils import load_checkpoint, unwrap_all
    from models import create_model
    from utils.ema import EMA

    numpy2_pickle_shim()
    cfg = load_cfg(model_key, rope_fix=rope_fix)
    model = create_model(cfg).to(device)
    ema = EMA(unwrap_all(model), decay=0.0)
    load_checkpoint(model, ema, Path(cfg.evaluation.checkpoint_path), device, apply_ema=True)
    model.eval()
    return cfg, model


class TinyGSMVal:
    """The training code's 1% validation split (prep_val.py), as conditional bit sequences."""

    def __init__(self):
        meta = json.loads((VAL_DIR / "meta.json").read_text())
        self.n = int(meta["n"])
        self.ids = np.memmap(VAL_DIR / "val_ids.uint16", dtype=np.uint16, mode="r", shape=(self.n, BLOCK))
        self.plen = np.memmap(VAL_DIR / "val_plen.int32", dtype=np.int32, mode="r", shape=(self.n,))
        self.alen = np.memmap(VAL_DIR / "val_alen.int32", dtype=np.int32, mode="r", shape=(self.n,))
        self.meta = meta

    def __len__(self):
        return self.n

    def batch(self, idx):
        """x0 bits [B,S] float, prefix bit mask [B,S] bool, answer-token mask [B,T] bool (answer + EOS)."""
        idx = np.asarray(idx)
        ids = torch.from_numpy(np.asarray(self.ids[idx]).astype(np.int64))
        plen = torch.from_numpy(np.asarray(self.plen[idx]).astype(np.int64))
        alen = torch.from_numpy(np.asarray(self.alen[idx]).astype(np.int64))
        pos = torch.arange(BLOCK)[None]
        prefix_tok = pos < plen[:, None]
        answer_tok = (pos >= plen[:, None]) & (pos < (plen + alen)[:, None])
        return ids_to_bits(ids).float(), tok_mask_to_bits(prefix_tok), answer_tok


class GSM8KTest:
    """GSM8K test prompts ([BOS] question \\n, padded), exactly as the branch's GSM8KTestDataset."""

    def __init__(self):
        self.records = json.loads(GSM8K_TEST.read_text())
        self.tok = get_tokenizer()
        self.bos, self.eos, self.pad = special_ids(self.tok)
        self.sep_ids = self.tok(SEP, add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.records)

    def batch(self, idx):
        rows, plens = [], []
        for i in idx:
            q = self.tok(self.records[i]["prompt"].strip(), add_special_tokens=False).input_ids
            p = ([self.bos] + q + self.sep_ids)[:BLOCK]
            plens.append(len(p))
            rows.append(p + [self.pad] * (BLOCK - len(p)))
        ids = torch.tensor(rows, dtype=torch.long)
        prefix_tok = torch.arange(BLOCK)[None] < torch.tensor(plens)[:, None]
        return ids_to_bits(ids).float(), tok_mask_to_bits(prefix_tok), plens

    def score(self, bits: torch.Tensor, plens, idx, timeout_s: float = 5.0):
        """Decode generated answers and score with the vendored S-FLM sandbox (as gsm8k_eval.py)."""
        from sandbox_gsm8k import evaluate_samples

        ids = bits_to_ids(bits.cpu())
        n_tok = len(self.tok)
        out, invalid, total = [], 0, 0
        for b, i in enumerate(idx):
            suf = ids[b, plens[b]:].tolist()
            total += len(suf)
            invalid += sum(1 for t in suf if t >= n_tok)
            text = self.tok.decode([t if 0 <= t < n_tok else self.tok.eos_token_id for t in suf],
                                   skip_special_tokens=True)
            try:  # a degenerate generation (e.g. null bytes) is a wrong answer, not a crash (as upstream 55f8b00)
                ok = bool(evaluate_samples(text, self.records[i]["response_ground_truth"], timeout_s))
            except Exception:
                ok = False
                self.n_scoring_errors = getattr(self, "n_scoring_errors", 0) + 1
            out.append(ok)
        return out, invalid, total

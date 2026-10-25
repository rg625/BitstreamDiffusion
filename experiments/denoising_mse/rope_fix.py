"""
Make eager inference reproduce the RoPE numerics of the compiled production model.

models/sdt.py RotaryEmbedding builds the angles with torch.einsum, which bf16 autocast runs in bf16 in
EAGER mode: positions above 256 are rounded (511 of 1,024 positions get their neighbour's cos/sin).
Training and the paper's evaluation ran under torch.compile (torch 2.4), where the angles are computed
in float32: with torch 2.5.1, compiled RotaryEmbedding output is bitwise equal to the float32 version
below (checked on this machine; torch 2.0.1's compile crashes here). Eager sampling without this fix is
therefore a train/test mismatch for 1024-token OWT sequences (harmless for <= 256 tokens, e.g. LM1B).

Usage: import rope_fix; rope_fix.apply()   (before running the model; idempotent)
"""
import torch

_APPLIED = False


def _forward_fp32(self, seq_len: int, device: torch.device, dtype: torch.dtype):
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    from models import sdt

    sdt.RotaryEmbedding.forward = _forward_fp32
    _APPLIED = True

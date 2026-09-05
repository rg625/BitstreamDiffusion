"""Per-position sigma schedules — the mechanism for temporal ordering.

CoBit conditions on ONE sigma per example broadcast to all positions, so it has
no notion of which token resolves first. An ordering is introduced by giving
each TOKEN its own point on the noise schedule.

Sigma is per token, never per bit: all `bits_per_token` bits of a token must
resolve together or the codeword is meaningless.

Definition. With `u_j` the normalised rank of token j in the chosen order
(0 = resolves first), `t` global time and `w >= 0` the ordering strength:

    t_j(t) = clip( t*(1+w) - w*u_j , 0, 1 )

`w = 0` gives `t_j = t` for every j, i.e. EXACTLY the current model. That is not
a special case bolted on -- it falls out of the definition, which is what makes
the equivalence test meaningful.

Convention, stated once: LOWER u_j  =>  reaches low sigma earlier  =>  RESOLVES
FIRST.
"""
from __future__ import annotations

from typing import Optional

import math

import torch

__all__ = ["ordering_ranks", "positional_time", "expand_token_sigma_to_bits"]


def ordering_ranks(
    mode: str,
    n_tokens: int,
    batch: int,
    *,
    suffix_mask: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    device=None,
) -> torch.Tensor:
    """Normalised rank u in [0,1] per (example, token). Shape [B, n_tokens].

    `suffix_mask` [B, n_tokens] marks the positions that are actually generated.
    Ranking is over the SUFFIX ONLY: prompt tokens are clamped clean and never
    denoise, so including them would let the fixed prompt consume part of the
    ordering range and quietly turn "left-to-right" into "prompt-first".
    Prompt positions receive u = 0; their sigma is irrelevant because they are
    overwritten by the clamp.

    `random` draws a fresh permutation per example per call. A permutation fixed
    per example would be a memorisable property of that example and would test
    something else entirely.
    """
    dev = device or (suffix_mask.device if suffix_mask is not None else "cpu")
    u = torch.zeros(batch, n_tokens, device=dev, dtype=torch.float32)
    if suffix_mask is None:
        suffix_mask = torch.ones(batch, n_tokens, dtype=torch.bool, device=dev)

    for b in range(batch):
        idx = torch.nonzero(suffix_mask[b], as_tuple=True)[0]
        k = idx.numel()
        if k == 0:
            continue
        if mode == "l2r":
            order = torch.arange(k, device=dev, dtype=torch.float32)
        elif mode == "r2l":
            order = torch.arange(k - 1, -1, -1, device=dev, dtype=torch.float32)
        elif mode == "random":
            order = torch.randperm(k, generator=generator, device=dev).float()
        elif mode in ("none", "simultaneous"):
            order = torch.zeros(k, device=dev, dtype=torch.float32)
        else:
            raise ValueError(f"unknown ordering mode {mode!r}")
        u[b, idx] = order / max(k - 1, 1)
    return u


def positional_time(t: torch.Tensor, u: torch.Tensor, w: float) -> torch.Tensor:
    """Per-token time. t [B] -> [B, n_tokens]. At w=0 every column equals t."""
    t_col = t.reshape(-1, 1).to(u.dtype)
    if w == 0.0:
        return t_col.expand_as(u).contiguous()
    return torch.clamp(t_col * (1.0 + w) - w * u, 0.0, 1.0)


def expand_token_sigma_to_bits(sigma_tok: torch.Tensor, bits_per_token: int) -> torch.Tensor:
    """[B, n_tokens] -> [B, n_tokens*bits_per_token], each token's sigma repeated."""
    return sigma_tok.repeat_interleave(int(bits_per_token), dim=-1)


# -----------------------------------------------------------------------------
# Deterministic ordering sampler
# -----------------------------------------------------------------------------

@torch.no_grad()
def sample_ordered(
    denoise_fn,
    *,
    sigmas: torch.Tensor,
    x_init: torch.Tensor,
    u: Optional[torch.Tensor] = None,
    w: float = 0.0,
    bits_per_token: int = 16,
    prefix_full: Optional[torch.Tensor] = None,
    prefix_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Deterministic Euler (probability-flow) sampling with per-position sigma.

    Scope, stated plainly: this is the ORDERING path only. It deliberately does
    NOT implement EDM churn, ATI sigma shifts, FKC or posterior tempering --
    reimplementing those would duplicate DDIMSampler and invite the two to
    diverge. At w=0 it is validated to reproduce a deterministic
    (gamma=0, no-churn) DDIM trajectory exactly; anything stochastic must go
    through DDIMSampler.

    `denoise_fn(x, sigma_bits) -> D` takes sigma already expanded to bits, so
    the caller controls how the model is conditioned.

    `sigmas` is the 1-D global schedule [T]. Per-token times come from
    `positional_time`, which at w=0 gives every token the global sigma, so this
    reduces to the ordinary Euler loop by construction rather than by a branch.
    """
    x = x_init.clone()
    B = x.shape[0]
    n_tok = x.shape[1] // int(bits_per_token)
    if u is None:
        u = torch.zeros(B, n_tok, device=x.device, dtype=torch.float32)

    # Map the global schedule to a normalised time in [0,1] (1 = noisiest), so
    # the ordering offset is applied in time, not in sigma -- sigma is highly
    # non-linear in t and offsetting it directly would not define an ordering.
    s_hi, s_lo = float(sigmas[0]), float(sigmas[-1])
    def t_of(sig):
        if s_hi <= s_lo:
            return torch.zeros(B, device=x.device)
        frac = (math.log(max(float(sig), 1e-20)) - math.log(max(s_lo, 1e-20))) / \
               (math.log(s_hi) - math.log(max(s_lo, 1e-20)))
        return torch.full((B,), float(min(max(frac, 0.0), 1.0)), device=x.device)

    def sigma_at(t_tok):
        lg = math.log(max(s_lo, 1e-20)) + t_tok * (math.log(s_hi) - math.log(max(s_lo, 1e-20)))
        return lg.exp()

    for i in range(len(sigmas) - 1):
        t_cur = positional_time(t_of(sigmas[i]), u, w)          # [B, n_tok]
        t_nxt = positional_time(t_of(sigmas[i + 1]), u, w)
        sig_cur = expand_token_sigma_to_bits(sigma_at(t_cur), bits_per_token)
        sig_nxt = expand_token_sigma_to_bits(sigma_at(t_nxt), bits_per_token)
        if prefix_mask is not None and prefix_full is not None:
            x = torch.where(prefix_mask, prefix_full, x)
        D = denoise_fn(x, sig_cur)
        score = (D - x) / (sig_cur ** 2)
        # probability-flow ODE: dx/dsigma = -sigma * score = (x - D)/sigma
        d = -sig_cur * score
        x = x + (sig_nxt - sig_cur) * d
    if prefix_mask is not None and prefix_full is not None:
        x = torch.where(prefix_mask, prefix_full, x)
    return x

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

__all__ = ["ordering_ranks", "positional_time", "expand_token_sigma_to_bits",
           "NoiseCDF", "window_ranks", "window_time", "training_window_sigma",
           "sample_window"]


def ordering_ranks(
    mode: str,
    n_tokens: int,
    batch: int,
    *,
    suffix_mask: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    device=None,
) -> torch.Tensor:
    """Normalised DENOISING PRIORITY u in [0,1] per (example, token). [B, n_tokens].

    CONVENTION, and it is easy to get backwards: in
    `t_j(t) = clip(t(1+w) - w*u_j, 0, 1)` a LARGER u_j subtracts more, giving a
    SMALLER per-token time and therefore a LOWER sigma. So **u = 1 means
    denoised FIRST**, u = 0 means denoised last. u is a priority, not a
    position. "l2r" therefore assigns u = 1 to the first suffix token and u = 0
    to the last.

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
            # First suffix token gets the HIGHEST priority so it denoises first.
            order = torch.arange(k - 1, -1, -1, device=dev, dtype=torch.float32)
        elif mode == "r2l":
            order = torch.arange(k, device=dev, dtype=torch.float32)
        elif mode == "random":
            # Always draw on CPU, then move. A CUDA generator cannot be used with
            # a CPU one and vice versa, and drawing on CPU additionally makes the
            # permutation reproducible across devices for a given seed -- which
            # the paired design depends on.
            order = torch.randperm(k, generator=generator, device="cpu").float().to(dev)
        elif mode in ("none", "simultaneous"):
            order = torch.zeros(k, device=dev, dtype=torch.float32)
        else:
            raise ValueError(f"unknown ordering mode {mode!r}")
        u[b, idx] = order / max(k - 1, 1)
    return u


def _apply_prompt_sigma(sigma_tok, sigma_global, prefix_mask, bits_per_token):
    """Expand per-token sigma to bits, pinning prompt bits to the global sigma."""
    sig = expand_token_sigma_to_bits(sigma_tok, bits_per_token)
    if prefix_mask is None:
        return sig
    g = sigma_global.reshape(-1, 1).expand_as(sig)
    return torch.where(prefix_mask, g, sig)


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
        # Prompt positions keep the GLOBAL sigma, i.e. exactly what they get at
        # w=0 and exactly how the model was trained (clean prefix clamped into a
        # globally-noised state). Letting the ordering move them would change the
        # conditioning the model sees at prompt positions, which is a second
        # intervention on top of the one under test.
        sig_cur = _apply_prompt_sigma(sigma_at(t_cur), sigma_at(t_of(sigmas[i])),
                                      prefix_mask, bits_per_token)
        sig_nxt = _apply_prompt_sigma(sigma_at(t_nxt), sigma_at(t_of(sigmas[i + 1])),
                                      prefix_mask, bits_per_token)
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


# -----------------------------------------------------------------------------
# Training-time ordering
# -----------------------------------------------------------------------------

def sigma_to_time(sigma: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """Map sigma -> normalised time t in [0,1], 1 = noisiest.

    Log-linear, and deliberately the SAME map `sample_ordered` uses, so an
    ordering applied during training means the same thing as one applied during
    sampling. Using two different maps would make the train/eval convention
    silently inconsistent, which is the easiest way to get a meaningless result.
    """
    lo, hi = math.log(max(sigma_min, 1e-20)), math.log(max(sigma_max, 1e-20))
    t = (sigma.clamp_min(1e-20).log() - lo) / max(hi - lo, 1e-12)
    return t.clamp(0.0, 1.0)


def time_to_sigma(t: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    lo, hi = math.log(max(sigma_min, 1e-20)), math.log(max(sigma_max, 1e-20))
    return (lo + t * (hi - lo)).exp()


def training_position_sigma(
    sigma_global: torch.Tensor,      # [B]
    *,
    n_bits: int,
    bits_per_token: int,
    mode: str,
    w: float,
    sigma_min: float,
    sigma_max: float,
    prefix_mask=None,                # [B, n_bits] bool, True = prompt
    generator=None,
) -> torch.Tensor:
    """[B] global sigma -> [B, n_bits] per-position sigma for TRAINING.

    Ordering acts on the generated SUFFIX only; prompt positions keep the global
    sigma, exactly as at sampling time. At w == 0 this returns the global sigma
    broadcast to every position, which is bit-identical to the scalar path.

    `random` draws a fresh permutation per example on every call, so a
    permutation can never become a memorisable property of an example.
    """
    B = sigma_global.shape[0]
    n_tok = n_bits // int(bits_per_token)
    dev = sigma_global.device
    suffix_tok = None
    if prefix_mask is not None:
        suffix_tok = ~prefix_mask.view(B, n_tok, int(bits_per_token)).all(dim=-1)
    u = ordering_ranks(mode, n_tok, B, suffix_mask=suffix_tok,
                       generator=generator, device=dev)
    t = sigma_to_time(sigma_global, sigma_min, sigma_max)
    t_pos = positional_time(t, u, float(w))
    sig_tok = time_to_sigma(t_pos, sigma_min, sigma_max)
    sig_bits = expand_token_sigma_to_bits(sig_tok, int(bits_per_token))
    if prefix_mask is not None:
        sig_bits = torch.where(prefix_mask, sigma_global.view(-1, 1).expand_as(sig_bits),
                               sig_bits)
    return sig_bits


# -----------------------------------------------------------------------------
# Window ordering: absolute token ranks, a frozen noise CDF
# -----------------------------------------------------------------------------
#
# The w-parameterised ordering above normalises ranks over the whole generated
# suffix. On TinyGSM that suffix is ~74% padding, so at w=0.25 the real answer
# spanned a 2x sigma range and adjacent tokens differed by 0.6%: l2r and r2l
# were simultaneous denoising with a global noise shift, not an ordering.
#
# The window form fixes the scale in TOKENS:
#
#     t_j(tau) = clip( tau*(1 + L/k) - r_j/k , 0, 1 )
#
# r_j in {0..L-1} is token j's denoising PRIORITY (larger = resolves first, the
# same convention as u above), L the number of ranked positions and k the width
# of the active window: about k tokens sit strictly inside (0,1) at any tau.
# k = inf gives t_j = tau for every token, i.e. the uniform model.
#
# Time is mapped to sigma through a FROZEN noise CDF F, sigma_j = F^-1(t_j),
# shared by every arm. With tau ~ U(0,1) the k = inf arm draws sigma ~ F exactly,
# and for any finite k every token strictly inside the window also has sigma ~ F
# (tau enters linearly, so the interior of t_j is uniform). Arms therefore
# differ only in WHICH tokens are at which noise level, never in the noise
# distribution a loss-bearing token sees. The atoms t=0 (resolved context) and
# t=1 (not yet entered) are what finite k adds.

class NoiseCDF:
    """Piecewise log-uniform noise distribution: bin masses over log-sigma edges.

    This is exactly the distribution `EntropyScheduleController.sample_entropy_sigma`
    draws from (pick a bin by its CDF, then log-uniform inside the bin), written
    as an invertible map so a time t in [0,1] has one sigma. t=1 is noisiest.
    """

    def __init__(self, edges: torch.Tensor, cdf: torch.Tensor):
        edges = torch.as_tensor(edges, dtype=torch.float64).flatten()
        cdf = torch.as_tensor(cdf, dtype=torch.float64).flatten().clamp(0.0, 1.0)
        if edges.numel() != cdf.numel() + 1:
            raise ValueError(f"need len(edges) == len(cdf)+1, got {edges.numel()} / {cdf.numel()}")
        if not bool((edges[1:] > edges[:-1]).all()):
            raise ValueError("edges must be strictly increasing")
        cdf = torch.cummax(cdf, dim=0).values
        cdf[-1] = 1.0
        self.log_edges = edges.log()
        self.cum = torch.cat([torch.zeros(1, dtype=torch.float64), cdf])   # [K+1]
        self.sigma_lo = float(edges[0])
        self.sigma_hi = float(self.icdf(torch.ones(1))[0])   # top of the SUPPORT

    @classmethod
    def load(cls, path) -> "NoiseCDF":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(d["edges"], d["cdf"])

    @classmethod
    def lognormal(cls, p_mean=-1.2, p_std=1.2, sigma_min=0.002, sigma_max=80.0, n_bins=4096):
        """Truncated log-normal, binned finely. For tests and as a fallback."""
        e = torch.linspace(math.log(sigma_min), math.log(sigma_max), n_bins + 1, dtype=torch.float64)
        z = (e - p_mean) / p_std
        Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        c = (Phi[1:] - Phi[0]) / (Phi[-1] - Phi[0])
        return cls(e.exp(), c)

    def icdf(self, t: torch.Tensor) -> torch.Tensor:
        """t in [0,1] (any shape) -> sigma. Monotone; t=0 -> sigma_min."""
        dev, dt = t.device, (t.dtype if t.is_floating_point() else torch.float32)
        cum, le = self.cum.to(dev), self.log_edges.to(dev)
        u = t.to(torch.float64).clamp(0.0, 1.0)
        # first bin whose upper cumulative mass reaches u
        idx = torch.searchsorted(cum[1:].contiguous(), u.contiguous(), right=False)
        idx = idx.clamp(0, cum.numel() - 2)
        c0, c1 = cum[idx], cum[idx + 1]
        frac = torch.where(c1 > c0, (u - c0) / (c1 - c0).clamp_min(1e-300),
                           torch.zeros_like(u)).clamp(0.0, 1.0)
        return (le[idx] + frac * (le[idx + 1] - le[idx])).exp().to(dt)

    def cdf_of(self, sigma: torch.Tensor) -> torch.Tensor:
        """sigma -> t = F(sigma). Inverse of icdf on the support."""
        dev = sigma.device
        cum, le = self.cum.to(dev), self.log_edges.to(dev)
        ls = sigma.to(torch.float64).clamp_min(1e-300).log()
        idx = (torch.searchsorted(le.contiguous(), ls.contiguous(), right=True) - 1)
        idx = idx.clamp(0, le.numel() - 2)
        frac = ((ls - le[idx]) / (le[idx + 1] - le[idx])).clamp(0.0, 1.0)
        return (cum[idx] + frac * (cum[idx + 1] - cum[idx])).to(sigma.dtype if sigma.is_floating_point() else torch.float32)


def window_ranks(mode: str, suffix_mask: torch.Tensor, *, generator=None):
    """ABSOLUTE denoising priority per token, and the ranked count L. Vectorised.

    suffix_mask [B, n] bool marks generated positions. Returns (r [B,n] float,
    L [B] float). r is an integer priority in {0..L-1} over the suffix, larger
    = resolves first; prompt positions get r = 0 and are overridden by callers.

      l2r    first suffix token has r = L-1          (resolves first)
      r2l    last suffix position has r = L-1
      random a fresh uniform permutation per example per call, from `generator`
             when given so that it never consumes the global RNG stream.

    Ranks are NOT normalised: the window width k in `window_time` is in tokens,
    so how much padding follows the answer cannot dilute the answer's ordering.
    """
    sm = suffix_mask.bool()
    B, n = sm.shape
    dev = sm.device
    L = sm.sum(dim=1)
    pos = torch.cumsum(sm.long(), dim=1) - 1                  # index within the suffix
    if mode == "l2r":
        r = (L.view(-1, 1) - 1 - pos)
    elif mode == "r2l":
        r = pos
    elif mode == "random":
        keys = torch.rand(B, n, device=dev, generator=generator)
        keys = keys.masked_fill(~sm, 2.0)                     # prompt sorts last
        order = torch.argsort(keys, dim=1)
        r = torch.empty_like(order)
        r.scatter_(1, order, torch.arange(n, device=dev).expand(B, n).contiguous())
    elif mode in ("none", "simultaneous"):
        r = torch.zeros(B, n, dtype=torch.long, device=dev)
    else:
        raise ValueError(f"unknown ordering mode {mode!r}")
    r = torch.where(sm, r, torch.zeros_like(r))
    return r.to(torch.float32), L.to(torch.float32)


def window_time(tau: torch.Tensor, r: torch.Tensor, L: torch.Tensor, k: float) -> torch.Tensor:
    """t_j = clip(tau*(1+L/k) - r_j/k, 0, 1). tau [B] or scalar; -> [B, n].

    k = inf returns tau for every token, by construction rather than by a limit
    evaluated in floating point.
    """
    tau = torch.as_tensor(tau, dtype=torch.float32, device=r.device)
    tau_col = (tau.reshape(-1, 1) if tau.dim() else tau.reshape(1, 1)).expand(r.shape[0], 1)
    if math.isinf(float(k)):
        return tau_col.expand_as(r).contiguous()
    k = float(k)
    if k <= 0:
        raise ValueError(f"window width k must be > 0, got {k}")
    return torch.clamp(tau_col * (1.0 + L.view(-1, 1) / k) - r / k, 0.0, 1.0)


def training_window_sigma(
    tau: torch.Tensor,               # [B] in (0, 1]
    *,
    n_bits: int,
    bits_per_token: int,
    mode: str,
    k: float,
    noise: NoiseCDF,
    prefix_mask=None,                # [B, n_bits] bool, True = prompt
    generator=None,                  # for `random` keys / loss subsampling only
    loss_match_k=None,               # k=inf only: supervise like a window-k arm
):
    """Per-bit sigma, per-bit loss mask and the global sigma for one batch.

    Returns (sigma_bits [B,n_bits], keep_bits [B,n_bits] bool, sigma_global [B]).

    * Prompt bits keep the global sigma F^-1(tau), exactly as the uniform model
      trains; they are clamped clean anyway and excluded from the loss by the
      caller's suffix mask.
    * keep = suffix & (0 < t_j < 1): only tokens strictly INSIDE the window
      carry loss, so every loss-bearing token has sigma ~ F in every arm and
      mode, with the same probability k/(k+L) per token. The two atoms are
      excluded for different reasons:
        t=0  resolved context at sigma_min. Its EDM target is -eps recovered
             from x at 1e-3 resolution, which bf16 cannot represent: an
             irreducible O(1) loss per token and pure-noise gradient.
        t=1  not yet entered, at the top of the support. Supervising it would
             make the SHARE of sigma_max supervision on the answer depend on
             direction -- measured on real TinyGSM at k=64: 27% of answer
             tokens loss-bearing under l2r against 85% under r2l -- so the
             modes would no longer see the same training noise.
      At k = inf keep == suffix and the mask is a no-op.
    * loss_match_k (k = inf only): the SUPERVISION-MATCHED control. A window-k
      arm supervises each generated token with probability k/(k+L); this keeps
      the uniform sigma field but supervises each token independently with that
      same probability, so "ordering" can be separated from "fewer supervised
      token-noise pairs per step".
    """
    B = tau.shape[0]
    bpt = int(bits_per_token)
    n_tok = int(n_bits) // bpt
    dev = tau.device
    if prefix_mask is not None:
        suffix_tok = ~prefix_mask.view(B, n_tok, bpt).all(dim=-1)
    else:
        suffix_tok = torch.ones(B, n_tok, dtype=torch.bool, device=dev)
    r, L = window_ranks(mode, suffix_tok, generator=generator)
    t = window_time(tau, r, L, k)
    sig_tok = noise.icdf(t).to(torch.float32)
    sigma_global = noise.icdf(tau.to(torch.float32)).to(torch.float32)
    sig_tok = torch.where(suffix_tok, sig_tok, sigma_global.view(-1, 1).expand_as(sig_tok))
    if math.isinf(float(k)):
        keep_tok = suffix_tok
        if loss_match_k is not None:
            km = float(loss_match_k)
            p = (km / (km + L)).view(-1, 1)
            keep_tok = keep_tok & (torch.rand(keep_tok.shape, device=dev, generator=generator) < p)
    else:
        if loss_match_k is not None:
            raise ValueError("loss_match_k is for the k=inf control only")
        keep_tok = suffix_tok & (t > 0) & (t < 1)
    return (expand_token_sigma_to_bits(sig_tok, bpt),
            expand_token_sigma_to_bits(keep_tok, bpt).bool(),
            sigma_global)


@torch.no_grad()
def sample_window(
    denoise_fn,
    *,
    num_steps: int,
    x_init: torch.Tensor,
    noise: NoiseCDF,
    r: torch.Tensor,                 # [B, n_tok] priorities from window_ranks
    L: torch.Tensor,                 # [B]
    k: float,
    bits_per_token: int = 16,
    prefix_full=None,
    prefix_mask=None,
):
    """Deterministic Euler over a UNIFORM global-time grid, per-token sigma.

    `num_steps` = N grid points, N-1 Euler steps, and the caller adds one final
    denoise: N network evaluations, the same count the standard sampler spends
    for the same N. The grid is uniform in tau so every token gets the same
    number of steps inside its window, N*k/(k+L) of them; at k=inf that is all
    N and the per-token path is the F-quantile schedule.

    The time->sigma map is the SAME F the model was trained with. Using any
    other map (Karras, log-linear) would put the tokens into joint noise
    configurations the model never saw.
    """
    x = x_init.clone()
    bpt = int(bits_per_token)
    taus = torch.linspace(1.0, 0.0, int(num_steps), device=x.device)
    B = x.shape[0]

    def sig_bits(tau):
        t = window_time(tau.expand(B), r, L, k)
        s = expand_token_sigma_to_bits(noise.icdf(t).to(torch.float32), bpt)
        if prefix_mask is not None:
            g = noise.icdf(tau.reshape(1)).to(torch.float32).expand_as(s)
            s = torch.where(prefix_mask, g, s)
        return s

    for i in range(len(taus) - 1):
        sc, sn = sig_bits(taus[i]), sig_bits(taus[i + 1])
        if prefix_mask is not None and prefix_full is not None:
            x = torch.where(prefix_mask, prefix_full, x)
        D = denoise_fn(x, sc)
        x = x + (sn - sc) * (x - D) / sc
    if prefix_mask is not None and prefix_full is not None:
        x = torch.where(prefix_mask, prefix_full, x)
    return x

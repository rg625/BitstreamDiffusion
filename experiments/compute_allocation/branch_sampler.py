"""Branching replay of the production LM1B DDIM sampler (unconditional, no guidance).

The loop is a line-for-line copy of `DDIMSampler.sample` for the configuration the
paper's LM1B Table 1 row uses (entropic grid, EDM churn, self-conditioning
`carry`, ati_eta=0, posterior temperature 1, no score temperature). It adds
three things the adaptive-compute study needs and nothing else:

  1. Saved states. `run` returns the full sampler state (x, self-conditioning
     tensors) at requested step indices, and can start from such a state.
  2. Keyed noise. With `noise=KeyedNoise(...)` the starting noise and the churn
     noise of ORIGINAL step i come from a generator seeded by (seed, chunk,
     stream, i), so branches that split steps into sub-steps still see exactly
     the same churn noise as the reference. With `noise=GlobalNoise()` the
     draws come from the global RNG in production order (equivalence tests).
  3. Per-step sub-steps. `factors[i] = m` integrates original step i in m
     sub-steps placed log-uniformly between the churned sigma and the next grid
     sigma. Churn is applied once per original step, before the first
     sub-step, with that step's own gamma and noise. One sub-step = one NFE.

`factors` all 1 + GlobalNoise reproduces `DDIMSampler.sample` bit for bit
(tests/test_compute_allocation_branching.py).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from diffusion.continuous.guidance import GuidedDenoiser
from diffusion.continuous.samplers import (
    DDIMSampler,
    _ati_shift_sigma_label,
    _build_mask_conditioning,
    _compute_edm_gamma,
    _resolve_guidance_config,
    _score_from_probs,
)


def _key_seed(*parts: int) -> int:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(int(p).to_bytes(8, "little", signed=True))
    return int.from_bytes(h.digest(), "little") & ((1 << 63) - 1)


class GlobalNoise:
    """Draw from the global RNG in the production call order."""

    def initial(self, B: int, S: int, device) -> torch.Tensor:
        return torch.randn(B, S, device=device, dtype=torch.float32)

    def churn(self, step: int, like: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(like)


@dataclass
class KeyedNoise:
    """Noise for a batch made of consecutive blocks, block j = (chunk, stream, n).

    The starting noise is keyed by (seed, chunk) only, so every stream of a chunk
    starts from the same x_T. Churn noise for original step i is keyed by
    (seed, chunk, stream, i): the same in every branch that shares the stream.
    """

    seed: int
    blocks: Sequence[Tuple[int, int, int]]  # (chunk, stream, n_samples)

    def _draw(self, key: Tuple[int, ...], n: int, S: int, device) -> torch.Tensor:
        g = torch.Generator(device=device)
        g.manual_seed(_key_seed(self.seed, *key))
        return torch.randn(n, S, generator=g, device=device, dtype=torch.float32)

    def initial(self, B: int, S: int, device) -> torch.Tensor:
        assert B == sum(n for _, _, n in self.blocks)
        return torch.cat([self._draw((c, -1), n, S, device) for c, _, n in self.blocks], 0)

    def churn(self, step: int, like: torch.Tensor) -> torch.Tensor:
        B, S = like.shape
        out = torch.cat(
            [self._draw((c, r, int(step)), n, S, like.device) for c, r, n in self.blocks], 0
        )
        assert out.shape[0] == B
        return out.to(dtype=like.dtype)


@dataclass
class SamplerState:
    step: int                      # original step index the state is AT (before its churn)
    x: torch.Tensor                # [B,S] fp32
    sc: Dict[tuple, Optional[torch.Tensor]] = field(default_factory=dict)

    def to(self, device) -> "SamplerState":
        return SamplerState(
            self.step,
            self.x.to(device),
            {k: (None if v is None else v.to(device)) for k, v in self.sc.items()},
        )

    def select(self, idx: torch.Tensor) -> "SamplerState":
        return SamplerState(
            self.step,
            self.x.index_select(0, idx.to(self.x.device)),
            {k: (None if v is None else v.index_select(0, idx.to(v.device))) for k, v in self.sc.items()},
        )

    @staticmethod
    def cat(states: Sequence["SamplerState"]) -> "SamplerState":
        assert len({s.step for s in states}) == 1
        keys = states[0].sc.keys()
        return SamplerState(
            states[0].step,
            torch.cat([s.x for s in states], 0),
            {k: (None if states[0].sc[k] is None else torch.cat([s.sc[k] for s in states], 0)) for k in keys},
        )


StepHook = Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], None]
# hook(step, sigma_state, x_state, D) -- called after the FIRST denoiser call of
# every original step, i.e. on information that is identical in every branch
# that has not yet diverged at that step.


class BranchingSampler:
    def __init__(
        self,
        model,
        proc,
        cfg,
        *,
        num_steps: int,
        sigma_min: float,
        entropy_run_dir: str,
        sigma_max: Optional[float] = None,
    ):
        self.base = DDIMSampler(model, proc, cfg)
        self.cfg = cfg
        self.device = self.base.device
        self.data_center = self.base.data_center
        if self.base.is_cont_tokens:
            raise ValueError("BranchingSampler supports the continuous-bits representation only")
        self.sigmas = self.base.sigmas.prepare(
            schedule="entropic",
            num_steps=num_steps,
            entropic_blend_alpha=0.0,
            entropy_run_dir=entropy_run_dir,
            sigma_min_override=sigma_min,
            sigma_max_override=sigma_max,
        )
        self.stoch = self.base.sigmas.resolve_stochastic_cfg(
            entropy_run_dir=entropy_run_dir,
            sigma_min_override=sigma_min,
            sigma_max_override=sigma_max,
        )
        self.gdn = GuidedDenoiser(
            model, cfg, _resolve_guidance_config(cfg, 0.0, None),
            is_cont_tokens=False, sigma_hi_cap=float(self.sigmas[0]),
        )

    @property
    def n_steps(self) -> int:
        return int(self.sigmas.numel()) - 1

    def segments(self, n_seg: int) -> List[Tuple[int, int]]:
        """[start, end) original-step ranges of n_seg near-equal segments.

        The grid is uniform in entropy-CDF quantile, so equal step counts are
        equal shares of entropy-rate mass.
        """
        idx = torch.arange(self.n_steps).tensor_split(n_seg)
        return [(int(c[0]), int(c[-1]) + 1) for c in idx]

    def initial_state(self, B: int, S: int, noise) -> SamplerState:
        self.gdn.reset()
        cond_enabled, *_ = _build_mask_conditioning(
            cfg=self.cfg, B=B, S=S, device=self.device,
            conditioning_prefix_full=None, cond_prefix_mask=None,
            conditioning_prefix=None, cond_len_bits=None,
        )
        if cond_enabled:
            raise ValueError("BranchingSampler is for the unconditional model")
        x = noise.initial(B, S, self.device) * self.sigmas[0]
        x = x + self.data_center
        sc = {}
        if self.base.sc_enabled:
            sc = {b: torch.zeros_like(x) for b in self.gdn.branches(False)}
        return SamplerState(0, x, sc)

    @torch.no_grad()
    def run(
        self,
        state: SamplerState,
        noise,
        *,
        factors: Optional[Dict[int, int]] = None,
        save_at: Sequence[int] = (),
        hook: Optional[StepHook] = None,
        stop_at: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[int, SamplerState], int]:
        """Integrate from `state` to the end (or to `stop_at`).

        Returns (probs or None, {step: saved state}, nfe_per_sample).
        probs is the final denoiser output (as with return_probs=True); it is
        None when stopping early.
        """
        factors = factors or {}
        sigmas = self.sigmas
        gdn = self.gdn
        gdn.reset()
        branches = gdn.branches(False)
        sc_state = gdn.make_self_cond_state(False)
        for b in branches:
            sc_state.set(b, state.sc.get(b))
        x = state.x.clone()
        saved: Dict[int, SamplerState] = {}
        save_at = set(int(s) for s in save_at)
        nfe = 0
        end = self.n_steps if stop_at is None else int(stop_at)

        for i in range(int(state.step), end):
            if i in save_at:
                saved[i] = SamplerState(i, x.clone(), {b: sc_state.get(b) for b in branches})
            sigma_cur, sigma_next = sigmas[i], sigmas[i + 1]
            sigma_prev = sigmas[i - 1] if i > 0 else None

            gamma_i = _compute_edm_gamma(
                sigma_cur,
                num_intervals=max(1, len(sigmas) - 1),
                s_churn=self.stoch.s_churn,
                s_tmin=self.stoch.s_tmin,
                s_tmax=self.stoch.s_tmax,
            ) if self.stoch.enabled else 0.0
            if gamma_i > 0.0:
                sigma_state = sigma_cur * (1.0 + float(gamma_i))
                sigma_delta = (sigma_state.square() - sigma_cur.square()).clamp_min(0.0).sqrt()
                eps = noise.churn(i, x) * float(self.stoch.s_noise)
                x_state = x + sigma_delta.to(device=x.device, dtype=x.dtype) * eps
            else:
                x_state, sigma_state = x, sigma_cur

            m = int(factors.get(i, 1))
            if m < 1:
                raise ValueError(f"factor for step {i} must be >= 1, got {m}")
            if m == 1:
                sub = [sigma_state, sigma_next]
            else:
                ratio = (sigma_next / sigma_state).clamp_min(1e-30)
                sub = [sigma_state] + [sigma_state * ratio ** (j / m) for j in range(1, m)] + [sigma_next]

            x_cur = x_state
            for j in range(m):
                s_a, s_b = sub[j], sub[j + 1]
                sigma_eval = _ati_shift_sigma_label(s_a, sigma_prev if j == 0 else None, 0.0)
                pred = gdn.denoise(
                    x_cur, sigma_eval, sc=sc_state, cond_enabled=False,
                    posterior_temp=1.0, posterior_temp_target="learned", pt_ctx=None,
                )
                nfe += 1
                if j == 0 and hook is not None:
                    hook(i, sigma_state, x_cur, pred.D)
                score = _score_from_probs(pred.D, x_cur, s_a, is_cont_tokens=False)
                d = -s_a * score
                x_cur = x_cur + (s_b - s_a) * d
                for b in branches:
                    sc_state.set(b, pred.D_branches[b])
            x = x_cur

        if end in save_at:
            saved[end] = SamplerState(end, x.clone(), {b: sc_state.get(b) for b in branches})
        if stop_at is not None and end < self.n_steps:
            return None, saved, nfe

        sigma_final = _ati_shift_sigma_label(sigmas[-1], sigmas[-2] if len(sigmas) > 1 else None, 0.0)
        pred_final = gdn.denoise(
            x, sigma_final, sc=sc_state, cond_enabled=False,
            posterior_temp=1.0, posterior_temp_target="learned", pt_ctx=None,
            update_sg_cache=False,
        )
        nfe += 1
        return pred_final.D, saved, nfe

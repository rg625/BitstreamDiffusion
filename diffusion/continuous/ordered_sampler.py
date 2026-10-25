"""Eval-facing wrapper that runs `sample_ordered` through the standard sampler API.

WHY A WRAPPER RATHER THAN A NEW SAMPLER
---------------------------------------
The control arm of the ordering experiment must differ from the intervention in
exactly one thing: `order_w`. So the schedule, the x initialisation, the prompt
clamping and the model call all come from the SAME code as the rest of the
codebase -- `SigmaSchedule.prepare` and `_model_logits_continuous` -- rather than
being reimplemented here. At `order_w=0` this is a plain uniform-sigma
deterministic Euler run, which is the control.

SCOPE, stated plainly: deterministic probability-flow only. No EDM churn, no
ATI, no FKC, no posterior tempering, no CFG. Any of those would need
DDIMSampler; asking for one here raises rather than being silently ignored.
"""
from __future__ import annotations

from typing import Optional

import torch

from diffusion.continuous.logit_postprocess import _model_logits_continuous
from diffusion.continuous.ordering import (
    expand_token_sigma_to_bits,
    ordering_ranks,
    sample_ordered,
)
from diffusion.continuous.samplers import DDIMSampler


def token_prefix_mask_from_bits(prefix_mask: torch.Tensor, bits_per_token: int) -> torch.Tensor:
    """[B, S_bits] bool -> [B, n_tok] bool. A token is prompt iff ALL its bits are.

    Partially-prompt tokens cannot occur with the task datasets (the prompt ends
    on a token boundary), but `all` is the safe reading: a token with any free
    bit must be allowed to participate in the ordering.
    """
    B, S = prefix_mask.shape
    return prefix_mask.view(B, S // int(bits_per_token), int(bits_per_token)).all(dim=-1)


class OrderedSampler(DDIMSampler):
    """DDIMSampler subclass whose `sample` runs the per-position-sigma path.

    Ordering parameters live on the instance, not in `sample()`'s signature, so
    the shared `sample_bits` eval path needs no change and the control and
    intervention arms go through byte-identical calling code.
    """

    def __init__(self, model, forward_process, cfg, *,
                 order_w: float = 0.0,
                 order_mode: str = "l2r",
                 order_seed: Optional[int] = None):
        super().__init__(model, forward_process, cfg)
        self.order_w = float(order_w)
        self.order_mode = str(order_mode)
        self.order_seed = order_seed
        # sample() calls made so far: the eval loop calls once per batch, so
        # this is the batch index (T5).
        self._order_batch_index = 0
        if self.is_cont_tokens:
            raise NotImplementedError(
                "OrderedSampler is binary-bitstream only; token-space ordering "
                "would need its own per-position sigma expansion.")

    def _next_order_generator(self, device):
        """Generator for this batch's ordering draws, or None without order_seed.

        Seeded from (order_seed, batch index) so each batch gets its own random
        orderings. Re-seeding from order_seed alone on every call gave row i of
        every batch the same permutation (T5). Batch 0 keeps the old seed, and
        the 1_000_003 stride keeps seeds 1234..1236 from aliasing each other.
        """
        idx = getattr(self, "_order_batch_index", 0)   # stubs may skip __init__
        self._order_batch_index = idx + 1
        if self.order_seed is None:
            return None
        return torch.Generator(device=device).manual_seed(int(self.order_seed) + 1_000_003 * idx)

    @staticmethod
    def _reject_active_guidance(guidance_scale, guidance, bad_model, collect_diagnostics):
        # Reject only when guidance is ACTUALLY ACTIVE. The task evals always
        # construct a guidance config and pass it, with every scale at 0.0; a
        # truthiness check on the object itself rejected every ordering cell.
        # (Same distinction as sample_bits' tempering guard: a companion object
        # being present is not a switch being on.)
        def _scale(g, k):
            if g is None:
                return 0.0
            v = g.get(k, 0.0) if isinstance(g, dict) else getattr(g, k, 0.0)
            try:
                return abs(float(v or 0.0))
            except (TypeError, ValueError):
                return 0.0
        active = abs(float(guidance_scale or 0.0)) > 0.0
        active = active or any(_scale(guidance, k) > 0.0
                               for k in ("cfg_scale", "ag_scale", "sg_scale",
                                         "guidance_scale", "scale"))
        active = active or (bad_model is not None)
        if active:
            raise NotImplementedError(
                "OrderedSampler does not implement guidance; the ordering study "
                "is deliberately guidance-free (the guidance study is closed).")
        if collect_diagnostics:
            raise NotImplementedError(
                "OrderedSampler has no guidance trace to collect.")

    def _make_denoise_fn(self, prefix_mask, prefix_full, last):
        """The model call both ordered samplers share: sigma already per bit.

        Self-conditioning carry, matching DDIMSampler's sc_refresh_mode="carry",
        which is what the task evals request. These checkpoints are trained with
        model.self_condition=True (p=0.5), so a sampler that never supplies the
        previous estimate runs the model in a weaker mode than its baseline.
        """
        sc = {"x0_hat": None}

        def denoise_fn(x_in, sigma_bits):
            x0_hat = None
            if self.sc_enabled:
                x0_hat = sc["x0_hat"]
                if x0_hat is None:
                    x0_hat = torch.zeros_like(x_in)
                if prefix_mask is not None and prefix_full is not None:
                    x0_hat = torch.where(prefix_mask, prefix_full, x0_hat)
            logits = _model_logits_continuous(self.model, self.cfg, x_in, sigma_bits, x0_hat)
            D = torch.sigmoid(logits.float())
            if self.sc_enabled:
                sc["x0_hat"] = D.detach()
            last["D"] = D
            last["sigma"] = sigma_bits
            return D
        return denoise_fn

    @torch.no_grad()
    def sample(self, num_samples, seq_len, *,
               conditioning_prefix_full=None,
               cond_prefix_mask=None,
               num_steps=None,
               schedule=None,
               entropy_run_dir=None,
               sigma_min_override=None,
               sigma_max_override=None,
               return_probs=False,
               guidance_scale=None,
               guidance=None,
               bad_model=None,
               collect_diagnostics=False,
               return_sigma_trace=False,
               **ignored):
        self._reject_active_guidance(guidance_scale, guidance, bad_model,
                                     collect_diagnostics)

        B, S = int(num_samples), int(seq_len)
        bpt = int(self.bits_per_token)
        sigmas = self.sigmas.prepare(
            schedule=schedule, num_steps=num_steps,
            entropy_run_dir=entropy_run_dir,
            sigma_min_override=sigma_min_override,
            sigma_max_override=sigma_max_override,
        )

        # Identical initialisation to DDIMSampler: N(0, sigma0^2) about the data
        # centre. Reproduced rather than shared because DDIM's copy is buried in
        # a 300-line method; the values are pinned by a test.
        x = torch.randn(B, S, device=self.device, dtype=torch.float32) * float(sigmas[0])
        x = x + self.data_center

        prefix_full = prefix_mask = None
        if conditioning_prefix_full is not None and cond_prefix_mask is not None:
            prefix_full = conditioning_prefix_full.to(self.device).float()
            prefix_mask = cond_prefix_mask.to(self.device).bool()

        n_tok = S // bpt
        gen = self._next_order_generator("cpu")
        suffix_tok = (~token_prefix_mask_from_bits(prefix_mask, bpt)
                      if prefix_mask is not None else None)
        u = ordering_ranks(self.order_mode, n_tok, B,
                           suffix_mask=suffix_tok, generator=gen).to(self.device)

        last = {}
        denoise_fn = self._make_denoise_fn(prefix_mask, prefix_full, last)

        x = sample_ordered(
            denoise_fn, sigmas=sigmas, x_init=x, u=u, w=self.order_w,
            bits_per_token=bpt, prefix_full=prefix_full, prefix_mask=prefix_mask,
        )

        # FINAL denoise at the smallest sigma, exactly as DDIMSampler does before
        # returning probs. Without it the decoded bits come from the denoise at
        # sigmas[-2] -- a far noisier level -- which is not what the baseline
        # returns and is not a fair control. Costs one forward, the same one
        # DDIM spends, so NFE stays matched at num_steps + 1.
        sig_final = torch.full_like(x, float(sigmas[-1]))
        probs = denoise_fn(x, sig_final)
        if prefix_mask is not None and prefix_full is not None:
            probs = torch.where(prefix_mask, prefix_full, probs)

        if return_sigma_trace:
            return x, probs, last.get("sigma")
        return (x, probs) if return_probs else x


class WindowSampler(OrderedSampler):
    """Window-ordered deterministic Euler: t_j = clip(tau(1+L/k) - r_j/k, 0, 1).

    Uses the SAME frozen noise CDF the arm was trained with (`noise_table`), a
    uniform tau grid of `num_steps` points and one final denoise, so it spends
    exactly `num_steps` network evaluations -- the same as the standard sampler
    (`OrderedSampler`, order_w=0, Karras) at the same `num_steps`. That is the
    matched-compute comparison.

    Ranks are over the whole generated canvas in absolute token units. At
    sampling time the answer's length is unknown, so this is the only order
    that can actually be run: l2r resolves the answer first and the padding
    after it; r2l resolves the padding (i.e. the length) first; random draws a
    fresh permutation per sample, seeded by `order_seed`.
    """

    def __init__(self, model, forward_process, cfg, *,
                 order_k: float = float("inf"),
                 order_mode: str = "none",
                 order_seed: Optional[int] = None,
                 noise_table: Optional[str] = None):
        super().__init__(model, forward_process, cfg, order_w=0.0,
                         order_mode=order_mode, order_seed=order_seed)
        from diffusion.continuous.ordering import NoiseCDF
        if not noise_table:
            raise ValueError("WindowSampler needs the arm's frozen noise table")
        self.order_k = float(order_k)
        self.noise = NoiseCDF.load(noise_table)

    @torch.no_grad()
    def sample(self, num_samples, seq_len, *,
               conditioning_prefix_full=None,
               cond_prefix_mask=None,
               num_steps=None,
               return_probs=False,
               guidance_scale=None,
               guidance=None,
               bad_model=None,
               collect_diagnostics=False,
               return_sigma_trace=False,
               **ignored):
        from diffusion.continuous.ordering import sample_window, window_ranks
        self._reject_active_guidance(guidance_scale, guidance, bad_model,
                                     collect_diagnostics)
        B, S = int(num_samples), int(seq_len)
        bpt = int(self.bits_per_token)
        N = int(num_steps if num_steps is not None
                else getattr(self.cfg.evaluation, "num_sampling_steps", 256))

        x = torch.randn(B, S, device=self.device, dtype=torch.float32) * float(self.noise.sigma_hi)
        x = x + self.data_center

        prefix_full = prefix_mask = None
        if conditioning_prefix_full is not None and cond_prefix_mask is not None:
            prefix_full = conditioning_prefix_full.to(self.device).float()
            prefix_mask = cond_prefix_mask.to(self.device).bool()
        n_tok = S // bpt
        suffix_tok = (~token_prefix_mask_from_bits(prefix_mask, bpt) if prefix_mask is not None
                      else torch.ones(B, n_tok, dtype=torch.bool, device=self.device))
        gen = self._next_order_generator(self.device)
        r, L = window_ranks(self.order_mode, suffix_tok.to(self.device), generator=gen)

        last = {}
        denoise_fn = self._make_denoise_fn(prefix_mask, prefix_full, last)
        x = sample_window(denoise_fn, num_steps=N, x_init=x, noise=self.noise,
                          r=r, L=L, k=self.order_k, bits_per_token=bpt,
                          prefix_full=prefix_full, prefix_mask=prefix_mask)

        # Final denoise at the bottom of the support, as every sampler here does.
        sig_final = torch.full_like(x, float(self.noise.sigma_lo))
        probs = denoise_fn(x, sig_final)
        if prefix_mask is not None and prefix_full is not None:
            probs = torch.where(prefix_mask, prefix_full, probs)
        if return_sigma_trace:
            return x, probs, last.get("sigma")
        return (x, probs) if return_probs else x

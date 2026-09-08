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
        if self.is_cont_tokens:
            raise NotImplementedError(
                "OrderedSampler is binary-bitstream only; token-space ordering "
                "would need its own per-position sigma expansion.")

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
        for name, val in (("guidance_scale", guidance_scale), ("guidance", guidance),
                          ("bad_model", bad_model)):
            if val:
                raise NotImplementedError(
                    f"OrderedSampler does not implement {name}; the ordering study "
                    "is deliberately guidance-free (the guidance study is closed).")
        if collect_diagnostics:
            raise NotImplementedError(
                "OrderedSampler has no guidance trace to collect.")

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
        gen = None
        if self.order_seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(int(self.order_seed))
        suffix_tok = (~token_prefix_mask_from_bits(prefix_mask, bpt)
                      if prefix_mask is not None else None)
        u = ordering_ranks(self.order_mode, n_tok, B,
                           suffix_mask=suffix_tok, generator=gen).to(self.device)

        last = {}

        def denoise_fn(x_in, sigma_bits):
            logits = _model_logits_continuous(self.model, self.cfg, x_in, sigma_bits, None)
            D = torch.sigmoid(logits.float())
            last["D"] = D
            last["sigma"] = sigma_bits
            return D

        x = sample_ordered(
            denoise_fn, sigmas=sigmas, x_init=x, u=u, w=self.order_w,
            bits_per_token=bpt, prefix_full=prefix_full, prefix_mask=prefix_mask,
        )
        probs = last.get("D", torch.full_like(x, 0.5))
        if return_sigma_trace:
            return x, probs, last.get("sigma")
        return (x, probs) if return_probs else x

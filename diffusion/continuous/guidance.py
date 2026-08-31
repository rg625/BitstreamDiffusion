#diffusion/continuous/guidance.py
"""Reusable diffusion-guidance policies for CoBit's continuous samplers.

This module owns *all* guidance algebra so the samplers do not each carry their
own copy. Three mechanisms are supported and may be combined:

  CFG   Classifier-free guidance (Ho & Salimans).
  AG    AutoGuidance (Karras et al., "Guiding a Diffusion Model with a Bad
        Version of Itself"): the guiding "bad" model is a weaker version of the
        *same* model (an earlier training checkpoint, or a smaller net).
  SG    Self-guidance: the "bad" model is the *same* good model evaluated at a
        higher noise level, so no second network is required.

Why every combination happens in D-space
----------------------------------------
CoBit's continuous binary parameterisation is

    D(x, sigma) = sigmoid(ell_raw(x, sigma) + mf(x, sigma))          (posterior mean)
    score(x, sigma) = (D(x, sigma) - x) / sigma^2                    (Tweedie)

`score` is an *affine* function of `D` whose coefficients (x, sigma) are shared
by every branch we interpolate, so for any weights {a_k} with sum(a_k) == 1

    sum_k a_k * score_k = ( sum_k a_k * D_k - x ) / sigma^2 .

Guiding the posterior mean and guiding the score are therefore the *same
operation*, and mixing in D-space is what the repo already did for CFG
(`probs_g = probs_u + w*(probs_c - probs_u)`). We keep that convention: it is
numerically identical, it keeps probabilities interpretable for diagnostics, and
it means one combinator serves CFG, AG and SG at once.

The affine-equivalence argument requires the branches to share `x` and `sigma`.
That holds for CFG (both branches read the same `x_state`; only the *clamped
prompt* coordinates differ, and those are masked out of the drift) and for AG
(both models see the identical input). It does NOT hold across noise levels,
which is exactly why SG needs the extra care documented below.

The matched filter must not be amplified
----------------------------------------
`mf(x, sigma) = matched_filter_scale * (x - center) / sigma^2` is an *analytic*
data-consistency term, not a learned belief. It carries no model error, so
amplifying it is not "correcting the model" -- it is just distorting the
likelihood. Note it enters inside the sigmoid, so CoBit's score does NOT
decompose additively as `learned_score + matched_filter_score`; the usual
"guide only the learned part" recipe cannot be implemented by splitting the
score. What we can do is control the *difference* each mechanism takes:

  CFG: the two branches share `x` and `sigma` at every free coordinate (the
       branches differ only on clamped prompt positions), hence mf_c == mf_u
       there and `D_c - D_u` is already purely learned. Nothing to do.
  AG:  good and bad see identical (x, sigma), hence identical mf. `D_good -
       D_bad` is purely learned. Nothing to do.
  SG:  the two evaluations sit at different sigma, so mf(x, sigma') !=
       mf(x, sigma) and a naive difference WOULD amplify the analytic term.
       `sg_mf_mode="hold"` (the default) therefore evaluates the network at the
       shifted noise level but re-attaches the matched filter at the *true*
       sigma:

           D_bad_sg = sigmoid( ell_raw(x, sigma') + mf(x, sigma) )

       so the SG direction isolates the learned logit's noise-level
       sensitivity. `sg_mf_mode="vary"` keeps the naive
       `D_bad_sg = D(x, sigma')` and is retained for ablation.

The self-guidance coordinate is log-sigma
-----------------------------------------
The natural finite-difference coordinate is `u = log(sigma)`, for two concrete
repo-grounded reasons:

  1. The network conditions on log-sigma directly -- `SigmaEmbedding.forward`
     computes `phases = sigma.log()[:, None] * freq` (models/sdt.py). A fixed
     step in log-sigma is a fixed step in the model's own time coordinate.
  2. CoBit's headline schedule is `entropic`: sigmas are the inverse-CDF of the
     entropy-rate distribution, so the grid is uniform in *entropy*, not in
     sigma and not in log-sigma. Anything keyed to raw grid spacing would make
     the guidance strength silently NFE-dependent.

So with `g(u) = D(x, e^u)` and a bad model at `u + delta` (delta > 0, i.e. a
noisier level), we use the spacing-normalised direction

    Delta_sg = delta_ref * [ g(u) - g(u + delta) ] / delta

which is `delta_ref` times a finite-difference estimate of `-dD/du`. Dividing by
the actual spacing `delta` and multiplying by the fixed reference `delta_ref` is
what makes SG-prev's strength comparable across NFE: the raw difference
`g(u) - g(u+delta)` shrinks as the grid refines, the normalised one does not.

  SG-exact: choose `delta = delta_ref = sg_delta` and evaluate the network a
            second time at `sigma * exp(sg_delta)`. Costs one extra model
            evaluation per branch per step; grid-independent by construction.
  SG-prev:  reuse the previous sampler step, whose spacing is
            `delta_i = log(sigma_prev) - log(sigma_cur) > 0`. Costs *zero*
            extra evaluations. Two approximations, both documented and tested:
            the spacing varies with the grid (removed by the normalisation
            above) and the cached prediction was taken at `x_prev`, not `x_cur`
            (irreducible; this is the "cheap" in cheap self-guidance).

Self-guidance is not self-conditioning
--------------------------------------
Self-conditioning feeds a previous x0 estimate into the model *input*.
Self-guidance uses differences between predictions to correct the *score*. They
are independent, and each guided branch keeps its own self-conditioning state
(see `SelfCondState`) so conditional and unconditional trajectories never share
one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import torch

from diffusion.continuous.logit_postprocess import (
    _matched_filter_binary,
    _model_logits_continuous,
)

# Branch keys: (model, condition) with model in {"good","bad"}, condition in {"c","u"}.
Branch = Tuple[str, str]

GOOD_C: Branch = ("good", "c")
GOOD_U: Branch = ("good", "u")
BAD_C: Branch = ("bad", "c")
BAD_U: Branch = ("bad", "u")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class GuidanceConfig:
    """Declarative guidance policy.

    Scale conventions follow the repository's existing `guidance_scale`:
    `w == 0` means the mechanism is OFF (and the sampler takes its legacy
    single-branch path), while `w == 1` is the algebraic no-op
    `a + 1*(b - a) == b`. Both therefore reproduce the unguided conditional
    prediction; `w == 0` does it in one model call, `w == 1` in two.
    """

    cfg_scale: float = 0.0
    ag_scale: float = 0.0
    sg_scale: float = 0.0

    # Self-guidance shape.
    sg_variant: str = "prev"        # "prev" (cached, 0 extra NFE) | "exact"
    sg_delta: float = 0.5           # delta_ref, in log-sigma units
    sg_mf_mode: str = "hold"        # "hold" (default) | "vary"

    def __post_init__(self) -> None:
        for name in ("cfg_scale", "ag_scale", "sg_scale", "sg_delta"):
            v = float(getattr(self, name))
            if not (v == v) or v in (float("inf"), float("-inf")):
                raise ValueError(f"{name} must be finite, got {v}")
        if self.sg_variant not in ("prev", "exact"):
            raise ValueError(f"sg_variant must be prev|exact, got {self.sg_variant!r}")
        if self.sg_mf_mode not in ("hold", "vary"):
            raise ValueError(f"sg_mf_mode must be hold|vary, got {self.sg_mf_mode!r}")
        if self.sg_enabled and float(self.sg_delta) <= 0.0:
            raise ValueError(f"sg_delta must be > 0 when self-guidance is on, got {self.sg_delta}")

    # -- predicates -----------------------------------------------------------
    @property
    def cfg_enabled(self) -> bool:
        return float(self.cfg_scale) > 0.0

    @property
    def ag_enabled(self) -> bool:
        return float(self.ag_scale) > 0.0

    @property
    def sg_enabled(self) -> bool:
        return float(self.sg_scale) != 0.0

    @property
    def any_enabled(self) -> bool:
        return self.cfg_enabled or self.ag_enabled or self.sg_enabled

    def branches(self, *, cond_enabled: bool) -> Tuple[Branch, ...]:
        """Which (model, condition) branches this policy must evaluate."""
        use_cfg = self.cfg_enabled and cond_enabled
        out = [GOOD_C]
        if use_cfg:
            out.append(GOOD_U)
        if self.ag_enabled:
            out.append(BAD_C)
            if use_cfg:
                out.append(BAD_U)
        return tuple(out)

    def describe(self) -> Dict[str, object]:
        return {
            "cfg_scale": float(self.cfg_scale),
            "ag_scale": float(self.ag_scale),
            "sg_scale": float(self.sg_scale),
            "sg_variant": str(self.sg_variant) if self.sg_enabled else None,
            "sg_delta": float(self.sg_delta) if self.sg_enabled else None,
            "sg_mf_mode": str(self.sg_mf_mode) if self.sg_enabled else None,
        }


# -----------------------------------------------------------------------------
# Pure combination algebra (no model, no state -- directly unit-testable)
# -----------------------------------------------------------------------------

def lerp_guidance(base: torch.Tensor, target: torch.Tensor, w: float) -> torch.Tensor:
    """The single guidance primitive: `base + w * (target - base)`.

    CFG is `lerp_guidance(D_uncond, D_cond, w_cfg)`.
    AG  is `lerp_guidance(D_bad,    D_good, w_ag)`.
    Written in this exact order so it is bit-identical to the expression the
    repository used before this refactor (`probs_u + w*(probs_c - probs_u)`).
    """
    return base + float(w) * (target - base)


def combine_cfg_ag(
    branches: Dict[Branch, torch.Tensor],
    *,
    cfg_scale: float,
    ag_scale: float,
    cond_enabled: bool,
) -> torch.Tensor:
    """Combine the CFG and AutoGuidance axes into one posterior mean.

    With both active this is the nested form requested by the experiment design:

        good_cfg = good_u + w_cfg * (good_c - good_u)
        bad_cfg  = bad_u  + w_cfg * (bad_c  - bad_u)
        D        = bad_cfg + w_ag * (good_cfg - bad_cfg)

    With only AG active it degenerates to Karras et al.'s conditional form
    `D_bad_c + w_ag * (D_good_c - D_bad_c)`; with only CFG active to the
    classifier-free form; with neither, to the plain conditional prediction.
    """
    use_cfg = bool(cond_enabled) and float(cfg_scale) > 0.0

    good = (
        lerp_guidance(branches[GOOD_U], branches[GOOD_C], cfg_scale)
        if use_cfg else branches[GOOD_C]
    )
    if float(ag_scale) <= 0.0:
        return good

    bad = (
        lerp_guidance(branches[BAD_U], branches[BAD_C], cfg_scale)
        if use_cfg else branches[BAD_C]
    )
    return lerp_guidance(bad, good, ag_scale)


def sg_direction(
    D_cur: torch.Tensor,
    D_bad: torch.Tensor,
    *,
    delta: float,
    delta_ref: float,
) -> torch.Tensor:
    """Spacing-normalised self-guidance direction in log-sigma coordinates.

        Delta_sg = delta_ref * (D_cur - D_bad) / delta

    `D_bad` is the prediction at the *noisier* level `log sigma + delta`
    (delta > 0). Dividing by the realised spacing and rescaling by the fixed
    `delta_ref` makes the direction a `delta_ref`-sized step along the estimated
    derivative `-dD/dlog sigma`, so its magnitude does not drift with the sigma
    grid / NFE. When `delta == delta_ref` this reduces to the plain difference
    `D_cur - D_bad`.
    """
    d = float(delta)
    if not (d > 0.0):
        raise ValueError(f"self-guidance spacing delta must be > 0, got {d}")
    return (float(delta_ref) / d) * (D_cur - D_bad)


def apply_sg(D_base: torch.Tensor, direction: torch.Tensor, w: float) -> torch.Tensor:
    """`D_base + w_sg * Delta_sg`."""
    return D_base + float(w) * direction


def score_from_D(
    D: torch.Tensor,
    x_t: torch.Tensor,
    sigma: torch.Tensor,
    *,
    is_cont_tokens: bool = False,
) -> torch.Tensor:
    """Tweedie score `(D - x)/sigma^2`; mirrors `samplers._score_from_probs`."""
    if isinstance(sigma, float):
        sigma = torch.tensor(sigma, device=x_t.device, dtype=x_t.dtype)
    if isinstance(sigma, torch.Tensor) and sigma.device != x_t.device:
        sigma = sigma.to(device=x_t.device)
    if sigma.dim() == 0:
        sigma = sigma.expand(x_t.size(0))
    view = (-1, 1, 1) if is_cont_tokens else (-1, 1)
    sigma2 = (sigma ** 2).view(*view).to(torch.float32)
    return (D.to(torch.float32) - x_t.to(torch.float32)) / sigma2


# -----------------------------------------------------------------------------
# Per-branch self-conditioning state
# -----------------------------------------------------------------------------

class SelfCondState:
    """Holds one self-conditioning tensor per guided branch.

    Conditional and unconditional trajectories -- and good/bad models -- each
    keep their own previous-x0 estimate. Mixing them would silently feed the
    conditional trajectory's belief into the unconditional branch and destroy
    the meaning of `D_c - D_u`, so branches are keyed explicitly and there is no
    fallback that would quietly return the wrong one.
    """

    def __init__(self, branches: Sequence[Branch]):
        self._branches = tuple(branches)
        self._state: Dict[Branch, Optional[torch.Tensor]] = {b: None for b in self._branches}

    @property
    def branches(self) -> Tuple[Branch, ...]:
        return self._branches

    def get(self, branch: Branch) -> Optional[torch.Tensor]:
        if branch not in self._state:
            raise KeyError(f"no self-conditioning state for branch {branch!r}")
        return self._state[branch]

    def set(self, branch: Branch, value: Optional[torch.Tensor]) -> None:
        if branch not in self._state:
            raise KeyError(f"no self-conditioning state for branch {branch!r}")
        # Detach: the cached estimate is an input, never a path for gradients.
        self._state[branch] = None if value is None else value.detach()

    def reset(self) -> None:
        for b in self._branches:
            self._state[b] = None


# -----------------------------------------------------------------------------
# Cached previous-step material for SG-prev
# -----------------------------------------------------------------------------

@dataclass
class _SGCache:
    """Previous-step material reused by SG-prev (zero extra model evaluations).

    We cache the *raw* logits (network output before the matched filter) rather
    than the posterior mean, so `sg_mf_mode="hold"` can re-attach the matched
    filter at the current sigma. `log_sigma` is the level the cached prediction
    was produced at, giving the realised spacing for the normalisation.
    """
    log_sigma: Optional[float] = None
    raw_logits: Dict[Branch, torch.Tensor] = field(default_factory=dict)
    D: Dict[Branch, torch.Tensor] = field(default_factory=dict)

    def clear(self) -> None:
        self.log_sigma = None
        self.raw_logits = {}
        self.D = {}

    @property
    def ready(self) -> bool:
        return self.log_sigma is not None and bool(self.D)


# -----------------------------------------------------------------------------
# Local masking helpers
# -----------------------------------------------------------------------------
# Deliberate small duplicates of `samplers._clamp_mask_` / `_zero_mask_`:
# `samplers` imports this module, so importing back would be circular. Semantics
# are identical and `tests/test_guidance.py::test_masking_helpers_match_samplers`
# pins them together.

def _clamp_mask_(x: torch.Tensor, full: torch.Tensor, mask: Optional[torch.Tensor]) -> None:
    if mask is None or (not bool(mask.any().item())):
        return
    if x.dim() == 3 and mask.dim() == 2:
        mask = mask.unsqueeze(-1).expand_as(x)
    x[mask] = full[mask]


def _zero_mask_(d: torch.Tensor, mask: Optional[torch.Tensor]) -> None:
    if mask is None or (not bool(mask.any().item())):
        return
    if d.dim() == 3 and mask.dim() == 2:
        mask = mask.unsqueeze(-1).expand_as(d)
    d[mask] = 0.0


def _as_batch_sigma(sigma, B: int, device, dtype=torch.float32) -> torch.Tensor:
    """Normalise a scalar/0-dim/[B] sigma to a [B] tensor."""
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor(float(sigma), device=device, dtype=dtype)
    sigma = sigma.to(device=device)
    if sigma.dim() == 0:
        return sigma.expand(B)
    if sigma.numel() == 1:
        return sigma.reshape(1).expand(B)
    if sigma.numel() != B:
        raise ValueError(f"sigma must be scalar or length-{B}, got {tuple(sigma.shape)}")
    return sigma.reshape(B)


# -----------------------------------------------------------------------------
# Result bundle
# -----------------------------------------------------------------------------

@dataclass
class GuidedPrediction:
    """Everything a sampler step needs from one guided denoiser evaluation.

    D             guided posterior mean actually used to build the drift.
    D_branches    per-branch unguided posterior means (self-conditioning carry,
                  decoding, entropy diagnostics).
    diagnostics   per-step scalars (guidance direction norms etc.); empty unless
                  diagnostics were requested.
    """
    D: torch.Tensor
    D_branches: Dict[Branch, torch.Tensor]
    diagnostics: Dict[str, float] = field(default_factory=dict)

    @property
    def D_cond(self) -> torch.Tensor:
        return self.D_branches[GOOD_C]

    def score(self, x_t, sigma, *, is_cont_tokens: bool = False) -> torch.Tensor:
        return score_from_D(self.D, x_t, sigma, is_cont_tokens=is_cont_tokens)


# -----------------------------------------------------------------------------
# Guided denoiser
# -----------------------------------------------------------------------------

class GuidedDenoiser:
    """Evaluates the model(s) under a `GuidanceConfig` and returns a guided D.

    The model stays completely unaware of guidance: this class only decides
    *which inputs* to build and *how to combine outputs*. Branches belonging to
    the same network are concatenated into a single forward pass.
    """

    def __init__(
        self,
        model,
        cfg,
        gcfg: GuidanceConfig,
        *,
        bad_model=None,
        is_cont_tokens: bool = False,
        collect_diagnostics: bool = False,
    ):
        if gcfg.ag_enabled and bad_model is None:
            raise ValueError(
                "AutoGuidance requires a `bad_model` (an earlier checkpoint or a "
                "smaller net of the same architecture); none was supplied."
            )
        if gcfg.sg_enabled and gcfg.sg_mf_mode == "hold" and is_cont_tokens:
            raise ValueError(
                "sg_mf_mode='hold' is implemented for the binary-bit representation "
                "only (the matched filter is recomputed by "
                "logit_postprocess._matched_filter_binary). Use sg_mf_mode='vary' "
                "for continuous-token runs."
            )
        self.model = model
        self.bad_model = bad_model
        self.cfg = cfg
        self.gcfg = gcfg
        self.is_cont_tokens = bool(is_cont_tokens)
        self.collect_diagnostics = bool(collect_diagnostics)
        self._sg_cache = _SGCache()
        self._nfe = 0

    # -- lifecycle ------------------------------------------------------------
    def reset(self) -> None:
        """Clear per-trajectory state. Call once at the start of `sample()`."""
        self._sg_cache.clear()
        self._nfe = 0

    @property
    def model_evaluations(self) -> int:
        """Denoiser evaluations issued, in units of one batch of B rows.

        Batching several branches into one forward pass saves kernel launches
        but not FLOPs, so this counts *rows / B*, which is what NFE-matched
        comparisons need.
        """
        return self._nfe

    def make_self_cond_state(self, cond_enabled: bool) -> SelfCondState:
        return SelfCondState(self.gcfg.branches(cond_enabled=cond_enabled))

    def branches(self, cond_enabled: bool) -> Tuple[Branch, ...]:
        return self.gcfg.branches(cond_enabled=cond_enabled)

    # -- model plumbing -------------------------------------------------------
    def _net(self, which: str):
        return self.model if which == "good" else self.bad_model

    def _branch_inputs(self, branch: Branch, x_state, prefix_full, prefix_mask, null_full,
                       cond_enabled: bool):
        """Build the (possibly prompt-clamped) network input for one branch.

        Conditioning in CoBit is inpainting-style: the prompt enters by clamping
        the prompt coordinates of `x_t` to the clean prefix. The unconditional
        branch clamps them to the null prefix instead. Free (suffix) coordinates
        are untouched and therefore identical across branches -- which is what
        makes `D_c - D_u` a purely learned difference.
        """
        x = x_state.clone()
        if cond_enabled:
            _clamp_mask_(x, null_full if branch[1] == "u" else prefix_full, prefix_mask)
        return x

    def _branch_sc(self, branch: Branch, sc: "SelfCondState", x_state,
                   prefix_full, prefix_mask, null_full, cond_enabled):
        """Self-conditioning input for one branch, prompt-clamped like the state.

        Cloned before clamping so the stored per-branch state is never mutated:
        each branch's trajectory must keep its own belief (see `SelfCondState`).
        """
        s = sc.get(branch)
        s = torch.zeros_like(x_state) if s is None else s.clone()
        if cond_enabled:
            _clamp_mask_(s, null_full if branch[1] == "u" else prefix_full, prefix_mask)
        return s

    def _logits(self, which: str, x, sigma_b, sc, *, posterior_temp, posterior_temp_target,
                pt_ctx, rows_per_eval: int):
        # Count in units of one batch of `rows_per_eval` rows, NOT in forward
        # passes: branches (and, for SG-exact, noise levels) are concatenated, so
        # one pass can carry several evaluations' worth of compute. Phase-14
        # cost comparisons depend on this being the honest number.
        self._nfe += max(1, int(x.shape[0]) // max(1, int(rows_per_eval)))
        return _model_logits_continuous(
            self._net(which), self.cfg, x, sigma_b, sc,
            posterior_temp=posterior_temp,
            posterior_temp_target=posterior_temp_target,
            pt_ctx=pt_ctx,
        )

    def _to_D(self, logits, dtype):
        if self.is_cont_tokens:
            return torch.softmax(logits.float(), dim=-1).to(dtype=dtype)
        return torch.sigmoid(logits.float()).to(dtype=dtype)

    def _mf(self, x, sigma_b) -> Optional[torch.Tensor]:
        """Matched-filter residual actually used by the postprocessing, or None."""
        if self.is_cont_tokens:
            return None
        return _matched_filter_binary(self.cfg, x, sigma_b)

    @staticmethod
    def _temp_divisor(posterior_temp: float, posterior_temp_target: str) -> float:
        """Denominator the matched filter is divided by inside the postprocessing.

        `apply_continuous_logit_postprocessing` builds, for temperature T:
            target "learned":  logit = ell_raw/T + mf        -> mf divisor 1
            target "full":     logit = (ell_raw + mf)/T      -> mf divisor T
        Knowing the divisor lets us strip and re-attach `mf` exactly.
        """
        return float(posterior_temp) if str(posterior_temp_target).lower() == "full" else 1.0

    def _strip_mf(self, logits, mf, divisor: float):
        """Remove the matched filter from postprocessed logits (`None` mf = no-op)."""
        return logits if mf is None else logits - mf.to(logits.dtype) / divisor

    def _attach_mf(self, stripped, mf, divisor: float):
        return stripped if mf is None else stripped + mf.to(stripped.dtype) / divisor

    # -- the step -------------------------------------------------------------
    def denoise(
        self,
        x_state: torch.Tensor,
        sigma_eval,
        *,
        sc: SelfCondState,
        prefix_full: Optional[torch.Tensor] = None,
        prefix_mask: Optional[torch.Tensor] = None,
        null_full: Optional[torch.Tensor] = None,
        cond_enabled: bool = False,
        posterior_temp: float = 1.0,
        posterior_temp_target: str = "learned",
        pt_ctx: Optional[dict] = None,
        update_sg_cache: bool = True,
    ) -> GuidedPrediction:
        """Evaluate every required branch and return the guided posterior mean."""
        gc = self.gcfg
        B = int(x_state.size(0))
        dtype = x_state.dtype
        device = x_state.device
        branches = gc.branches(cond_enabled=cond_enabled)
        sigma_b = _as_batch_sigma(sigma_eval, B, device)
        divisor = self._temp_divisor(posterior_temp, posterior_temp_target)

        sg_exact = gc.sg_enabled and gc.sg_variant == "exact"
        if gc.sg_enabled and gc.sg_mf_mode == "hold" and pt_ctx is not None \
                and str(pt_ctx.get("space", "bit")).lower() == "token":
            raise ValueError(
                "sg_mf_mode='hold' cannot be combined with posterior_temp_space='token': "
                "codeword sharpening folds the matched filter through a softmax over "
                "valid codewords, so it cannot be stripped and re-attached exactly. "
                "Use sg_mf_mode='vary' (and note it amplifies the analytic term)."
            )

        # Shifted (noisier) evaluation level for exact self-guidance.
        sigma_hi_b = sigma_b * float(torch.exp(torch.tensor(float(gc.sg_delta)))) if sg_exact else None

        # Per-branch network inputs (prompt-clamped) are shared by both levels.
        x_in = {
            b: self._branch_inputs(b, x_state, prefix_full, prefix_mask, null_full, cond_enabled)
            for b in branches
        }
        mf_cur = self._mf(x_state, sigma_b)

        total_logits: Dict[Branch, torch.Tensor] = {}
        total_logits_hi: Dict[Branch, torch.Tensor] = {}

        # One forward pass per distinct network; conditions (and, for SG-exact,
        # noise levels) are concatenated along the batch dimension.
        for which in ("good", "bad"):
            group = [b for b in branches if b[0] == which]
            if not group:
                continue
            xs, sigs, scs = [], [], []
            for b in group:
                xs.append(x_in[b])
                sigs.append(sigma_b)
                scs.append(self._branch_sc(b, sc, x_state, prefix_full, prefix_mask,
                                           null_full, cond_enabled))
            if sg_exact:
                # NOTE: this packs two different noise levels into one forward
                # pass, so the network MUST condition on sigma per row. CoBit's
                # SDT does (`SigmaEmbedding` computes sigma.log()[:, None]); a
                # model that collapsed sigma to a scalar would silently return
                # the current level twice and make self-guidance a no-op.
                # `test_d9_sg_exact_requires_per_row_sigma` pins this.
                for b in group:
                    xs.append(x_in[b])
                    sigs.append(sigma_hi_b)
                    scs.append(self._branch_sc(b, sc, x_state, prefix_full, prefix_mask,
                                               null_full, cond_enabled))

            out = self._logits(
                which, torch.cat(xs, 0), torch.cat(sigs, 0), torch.cat(scs, 0),
                posterior_temp=posterior_temp,
                posterior_temp_target=posterior_temp_target,
                pt_ctx=pt_ctx,
                rows_per_eval=B,
            )
            for j, b in enumerate(group):
                total_logits[b] = out[j * B:(j + 1) * B]
            if sg_exact:
                off = len(group)
                for j, b in enumerate(group):
                    total_logits_hi[b] = out[(off + j) * B:(off + j + 1) * B]

        # Posterior means at the true noise level.
        D_branches: Dict[Branch, torch.Tensor] = {}
        for b in branches:
            D = self._to_D(total_logits[b], dtype)
            if cond_enabled:
                _clamp_mask_(D, null_full if b[1] == "u" else prefix_full, prefix_mask)
            D_branches[b] = D

        base = combine_cfg_ag(
            D_branches,
            cfg_scale=gc.cfg_scale, ag_scale=gc.ag_scale, cond_enabled=cond_enabled,
        )

        # ---- self-guidance ---------------------------------------------------
        direction = None
        delta_used = None
        if gc.sg_enabled:
            hold = gc.sg_mf_mode == "hold"
            if sg_exact:
                D_hi: Dict[Branch, torch.Tensor] = {}
                mf_hi = self._mf(x_state, sigma_hi_b)
                for b in branches:
                    lg = total_logits_hi[b]
                    if hold:
                        lg = self._attach_mf(self._strip_mf(lg, mf_hi, divisor), mf_cur, divisor)
                    D = self._to_D(lg, dtype)
                    if cond_enabled:
                        _clamp_mask_(D, null_full if b[1] == "u" else prefix_full, prefix_mask)
                    D_hi[b] = D
                bad_side = combine_cfg_ag(
                    D_hi, cfg_scale=gc.cfg_scale, ag_scale=gc.ag_scale,
                    cond_enabled=cond_enabled,
                )
                delta_used = float(gc.sg_delta)
                direction = sg_direction(
                    base, bad_side, delta=delta_used, delta_ref=float(gc.sg_delta),
                )
            elif self._sg_cache.ready:
                log_sig_cur = float(torch.log(sigma_b.reshape(-1)[0].clamp_min(1e-20)))
                delta_i = float(self._sg_cache.log_sigma) - log_sig_cur
                if delta_i > 1e-8:
                    D_prev: Dict[Branch, torch.Tensor] = {}
                    ok = True
                    for b in branches:
                        if b not in self._sg_cache.D:
                            ok = False
                            break
                        if hold and b in self._sg_cache.raw_logits:
                            lg = self._attach_mf(self._sg_cache.raw_logits[b], mf_cur, divisor)
                            D = self._to_D(lg, dtype)
                        else:
                            D = self._sg_cache.D[b]
                        if cond_enabled:
                            _clamp_mask_(D, null_full if b[1] == "u" else prefix_full, prefix_mask)
                        D_prev[b] = D
                    if ok:
                        bad_side = combine_cfg_ag(
                            D_prev, cfg_scale=gc.cfg_scale, ag_scale=gc.ag_scale,
                            cond_enabled=cond_enabled,
                        )
                        delta_used = delta_i
                        direction = sg_direction(
                            base, bad_side, delta=delta_i, delta_ref=float(gc.sg_delta),
                        )

        D_used = base if direction is None else apply_sg(base, direction, gc.sg_scale)
        if cond_enabled:
            _clamp_mask_(D_used, prefix_full, prefix_mask)

        # ---- cache for the next step's SG-prev --------------------------------
        if gc.sg_enabled and gc.sg_variant == "prev" and update_sg_cache:
            self._sg_cache.log_sigma = float(torch.log(sigma_b.reshape(-1)[0].clamp_min(1e-20)))
            self._sg_cache.D = {b: D_branches[b].detach() for b in branches}
            self._sg_cache.raw_logits = {
                b: self._strip_mf(total_logits[b], mf_cur, divisor).detach() for b in branches
            }

        diagnostics: Dict[str, float] = {}
        if self.collect_diagnostics:
            diagnostics = self._diagnostics(
                D_branches, base, D_used, direction, delta_used,
                x_state, sigma_b, prefix_mask, cond_enabled,
            )
        return GuidedPrediction(D=D_used, D_branches=D_branches, diagnostics=diagnostics)

    # -- diagnostics ----------------------------------------------------------
    @staticmethod
    def _free_rms(t: Optional[torch.Tensor], mask: Optional[torch.Tensor],
                  cond_enabled: bool) -> Optional[float]:
        """RMS over free (non-prompt) coordinates; prompt positions are clamped
        and carry no drift, so including them would dilute every norm."""
        if t is None:
            return None
        v = t.detach().to(torch.float32)
        if cond_enabled and mask is not None:
            m = mask
            if v.dim() == 3 and m.dim() == 2:
                m = m.unsqueeze(-1).expand_as(v)
            free = ~m
            n = int(free.sum().item())
            if n == 0:
                return 0.0
            return float(torch.sqrt((v[free] ** 2).sum() / n))
        return float(torch.sqrt((v ** 2).mean()))

    def _diagnostics(self, D_branches, base, D_used, direction, delta_used,
                     x_state, sigma_b, prefix_mask, cond_enabled) -> Dict[str, float]:
        """Per-step guidance telemetry.

        Direction norms are reported in D-space. The map to score-space is the
        shared factor 1/sigma^2, so every *ratio* below (the scientifically
        interesting quantity) is identical in either space.
        """
        gc = self.gcfg
        out: Dict[str, float] = {
            "sigma": float(sigma_b.reshape(-1)[0]),
            "log_sigma": float(torch.log(sigma_b.reshape(-1)[0].clamp_min(1e-20))),
        }
        use_cfg = gc.cfg_enabled and cond_enabled

        cfg_dir = (D_branches[GOOD_C] - D_branches[GOOD_U]) if use_cfg else None
        ag_dir = None
        if gc.ag_enabled:
            good = (lerp_guidance(D_branches[GOOD_U], D_branches[GOOD_C], gc.cfg_scale)
                    if use_cfg else D_branches[GOOD_C])
            bad = (lerp_guidance(D_branches[BAD_U], D_branches[BAD_C], gc.cfg_scale)
                   if use_cfg else D_branches[BAD_C])
            ag_dir = good - bad

        for name, vec in (("cfg", cfg_dir), ("ag", ag_dir), ("sg", direction)):
            r = self._free_rms(vec, prefix_mask, cond_enabled)
            if r is not None:
                out[f"{name}_dir_rms"] = r

        base_rms = self._free_rms(base - x_state, prefix_mask, cond_enabled)
        used_rms = self._free_rms(D_used - x_state, prefix_mask, cond_enabled)
        sig2 = float(sigma_b.reshape(-1)[0]) ** 2
        out["score_rms"] = (used_rms / sig2) if (used_rms is not None and sig2 > 0) else float("nan")
        # Total displacement guidance added, relative to the unguided score.
        shift = self._free_rms(D_used - D_branches[GOOD_C], prefix_mask, cond_enabled)
        if shift is not None and base_rms:
            out["guidance_over_score"] = shift / base_rms
        if delta_used is not None:
            out["sg_delta_used"] = float(delta_used)

        # Bit-level saturation / entropy of the conditional posterior.
        p = D_branches[GOOD_C].detach().to(torch.float32)
        if cond_enabled and prefix_mask is not None and p.dim() == prefix_mask.dim():
            p = p[~prefix_mask]
        p = p.reshape(-1)
        if p.numel() and not self.is_cont_tokens:
            pc = p.clamp(1e-6, 1 - 1e-6)
            ent = -(pc * pc.log() + (1 - pc) * (1 - pc).log())
            out["bit_entropy_mean"] = float(ent.mean())
            out["p_mean"] = float(p.mean())
            out["p_min"] = float(p.min())
            out["p_max"] = float(p.max())
            for thr in (0.01, 0.001):
                out[f"frac_p_lt_{thr}"] = float((p < thr).float().mean())
                out[f"frac_p_gt_{1 - thr}"] = float((p > 1 - thr).float().mean())
        return out

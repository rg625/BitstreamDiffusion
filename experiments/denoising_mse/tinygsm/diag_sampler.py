"""
Sampler loop for the TinyGSM diagnostics: DDIMSampler's conditional, binary, self-conditioned, non-CFG path
(EDM churn + Euler step, SC carry, final denoise; ati_eta = 0) written out so that the variants can intervene inside
a step. variant "base" reproduces DDIMSampler.sample(sc_refresh_mode="carry") exactly (same helpers, same order of
operations, same churn generators).

Variants (Variant dataclass):
  sc_reset   sigma_r    x0_hat <- 0 (prompt := x0) once, at the first step with sigma_i <= sigma_r; carry afterwards
  sc_decay   alpha      x0_hat <- alpha x0_hat + (1 - alpha) D  instead of x0_hat <- D
  clamp      tau, mode, (sigma_c)  a token whose 16 bit probabilities all satisfy max(p, 1-p) > tau is fixed to its hard bits
                        for all later steps (state, SC input and output, as the prompt): mode "scatter" clamps every
                        confident answer token, mode "prefix" only extends the clamped prefix by the longest run of
                        confident tokens right after it (= a longer prompt, which the model was trained on);
                        optional gate: clamp only at steps with sigma_i <= sigma_c   (names clampS0.99, clampP0.99s1.0)
  stagger    dt         position-dependent noise: answer token at relative position r (of L = 512 - prompt length)
                        follows sigma_base(clip(u_k + dt r / L, 0, 1)), u_k = 1 - (1 + dt) k / 255, so earlier tokens
                        are ahead; the network gets one sigma per sequence (geometric mean over answer bits) and the
                        matched-filter term uses each position's own sigma; the network gets the largest answer-bit
                        sigma (stagger<dt>) or the geometric mean (staggerG<dt>)
  tok        T, p, (sigma_g)  token-space decode: the per-bit posterior is replaced by the mean over a tempered (T)
                        and nucleus-truncated (top-p) categorical over the valid tokens (tok<T>p<p>[s<sigma_g>][L])
  stemp      tau, (u)   global (Sigma / CDCD) score temperature: the drift is divided by tau at every step,
                        D_tau = x + (D - x)/tau, as DDIMSampler(score_temp_tau=tau); suffix u leaves the last step
                        (onto sigma_min) untempered, score_temp_final="unsharpened"   (names stemp0.3, stemp0.3u)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

import common as C


@dataclass
class Variant:
    name: str = "base"
    sc_reset: float | None = None
    sc_decay: float | None = None
    clamp_tau: float | None = None
    clamp_mode: str = "prefix"
    clamp_sigma: float | None = None  # clamp only at steps with sigma_i <= clamp_sigma (None: any step)
    stagger: float | None = None
    stagger_emb: str = "max"          # network sigma under staggering: "max" or "gmean" over answer bits
    tok_T: float | None = None        # token-space (valid-codeword) temperature; None = off (per-bit sigmoid)
    tok_p: float = 1.0                # nucleus (top-p) over valid tokens after tempering; 1.0 = off
    tok_sigma: float | None = None    # apply the token-space decode only at steps with sigma_i <= tok_sigma
    tok_target: str = "full"          # "full": <C_v, l_raw + mf>/T ; "learned": <C_v, l_raw>/T + <C_v, mf>
    st_tau: float | None = None       # global score temperature (drift / tau at every step); None = off
    st_final: str = "sharpened"       # "unsharpened": the last step uses the untempered drift


def _temper(d, var: Variant, i: int, n_int: int):
    """Global score temperature on the drift d of step i (of n_int), as DDIMSampler(score_temp_tau, score_temp_final)."""
    if var.st_tau is None or (i == n_int - 1 and var.st_final == "unsharpened"):
        return d
    return d / var.st_tau


def parse_variant(name: str) -> Variant:
    if name == "base":
        return Variant()
    if name.startswith("screset"):
        return Variant(name, sc_reset=float(name[7:]))
    if name.startswith("scdecay"):
        return Variant(name, sc_decay=float(name[7:]))
    if name.startswith("clampS") or name.startswith("clampP"):
        tau, _, sg = name[6:].partition("s")
        return Variant(name, clamp_tau=float(tau), clamp_mode="scatter" if name[5] == "S" else "prefix",
                       clamp_sigma=float(sg) if sg else None)
    if name.startswith("tok"):
        # tok<T>p<p>[s<sigma>][L]   e.g. tok0.5p1.0, tok1.0p0.9, tok0.5p0.9s2.5, tok0.5p1.0L (learned target)
        body = name[3:]
        target = "learned" if body.endswith("L") else "full"
        body = body.rstrip("L")
        t, _, rest = body.partition("p")
        pp, _, sg = rest.partition("s")
        return Variant(name, tok_T=float(t), tok_p=float(pp) if pp else 1.0, tok_sigma=float(sg) if sg else None,
                       tok_target=target)
    if name.startswith("stemp"):
        body = name[5:]
        final = "unsharpened" if body.endswith("u") else "sharpened"
        tau = float(body.rstrip("u"))
        if tau <= 0.0:
            raise ValueError(f"{name}: tau must be > 0")
        return Variant(name, st_tau=tau, st_final=final)
    if name.startswith("staggerG"):
        return Variant(name, stagger=float(name[8:]), stagger_emb="gmean")
    if name.startswith("stagger"):
        return Variant(name, stagger=float(name[7:]))
    raise ValueError(name)


def _edm_gamma_vec(sig, n_int, stc):
    g = min(float(stc.s_churn) / float(n_int), math.sqrt(2.0) - 1.0) if stc.enabled else 0.0
    return torch.where((sig >= float(stc.s_tmin)) & (sig <= float(stc.s_tmax)), torch.full_like(sig, g),
                       torch.zeros_like(sig))


_CODEBOOK = {}


def codebook(device):
    """[V, 16] bits of every valid token id (MSB-first, as the data pipeline), cached per device."""
    if device not in _CODEBOOK:
        V = len(C.get_tokenizer())
        _CODEBOOK[device] = C.ids_to_bits(torch.arange(V)[None]).view(V, C.BPT).float().to(device)
    return _CODEBOOK[device]


def nucleus_threshold(q, p, k=1024, iters=30):
    """Per-row probability threshold thr with {v: q_v >= thr} = the top-p nucleus (smallest most-likely set with mass
    >= p; ties kept). Exact from the top-k values when they already hold mass p, otherwise found by bisection on
    log thr (the kept mass is monotone in thr), for the remaining rows only."""
    vals = q.topk(min(k, q.shape[-1]), dim=-1).values
    cs = vals.cumsum(-1)
    covered = cs[:, -1] >= p
    kk = ((cs - vals) < p).sum(-1, keepdim=True).clamp(1, vals.shape[-1])
    thr = vals.gather(-1, kk - 1)
    rows = (~covered).nonzero().squeeze(-1)
    if rows.numel():
        qr = q[rows]
        lo = torch.log(qr.min(-1, keepdim=True).values.clamp_min(1e-30))   # kept mass 1 >= p
        hi = torch.log(vals[rows, -1:].clamp_min(1e-30))                    # kept mass < p (row not covered)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            ok = (qr * (qr >= mid.exp())).sum(-1, keepdim=True) >= p
            lo = torch.where(ok, mid, lo)
            hi = torch.where(ok, hi, mid)
        thr[rows] = lo.exp()
    return thr


def token_decode(raw, mf, T, top_p, target, cb, chunk_rows=2048):
    """Joint valid-codeword decode (port of origin/tasks/sudoku-tinygsm fc401f4 _codeword_sharpen, + top-p).
    Per token with per-bit logits l: L(v) = <C_v, l_raw + mf>/T ("full") or <C_v, l_raw>/T + <C_v, mf> ("learned");
    q = softmax over the V valid tokens; top-p keeps the smallest set of most likely tokens with mass >= p (ties kept)
    and renormalises; returns the bit-posterior mean D = q C, shape [B, S]."""
    B, S = raw.shape
    m = cb.shape[1]
    Ct = cb.t().contiguous()
    r_all = raw.float().reshape(-1, m)
    mf_all = mf.float().reshape(-1, m)
    out = torch.empty_like(r_all)
    for i in range(0, r_all.shape[0], chunk_rows):
        L = r_all[i:i + chunk_rows] @ Ct
        Lm = mf_all[i:i + chunk_rows] @ Ct
        L = (L + Lm) / T if target == "full" else L / T + Lm
        q = torch.softmax(L, dim=-1)
        if top_p < 1.0:
            q = q / q.sum(-1, keepdim=True)
            q = q * (q >= nucleus_threshold(q, top_p))
            q = q / q.sum(-1, keepdim=True)
        out[i:i + chunk_rows] = q @ cb
    return out.reshape(B, S)


def _denoise(model, cfg, x, sig, cond, var, use_tok):
    """D(x, sigma, cond): per-bit sigmoid of the post-processed logits, or the token-space decode."""
    from diffusion.continuous.logit_postprocess import _model_logits_continuous, canonicalize_continuous_logits

    B = x.shape[0]
    if not use_tok:
        return torch.sigmoid(_model_logits_continuous(model, cfg, x, sig.expand(B), cond).float()).to(x.dtype)
    raw = canonicalize_continuous_logits(model(x, sig.expand(B), cond), is_cont_tokens=False)
    mf = (float(cfg.model.matched_filter_scale) * (x.float() - float(cfg.model.matched_filter_center))
          / sig.float().square().clamp_min(1e-12))
    clip = getattr(cfg.model, "matched_filter_clip", None)
    if clip is not None:
        mf = mf.clamp(-float(clip), float(clip))
    D = token_decode(raw, mf, var.tok_T, var.tok_p, var.tok_target, codebook(x.device))
    return D.clamp(1e-6, 1 - 1e-6).to(x.dtype)


@torch.no_grad()
def sample(model, cfg, stc, sigmas, x_init, prefix_full, prefix_mask, plens, churn_seed, var: Variant):
    """Returns (final probs [B,S], info dict). x_init has the prompt clamped already."""
    from diffusion.continuous.logit_postprocess import _model_logits_continuous, canonicalize_continuous_logits
    from diffusion.continuous.samplers import (_apply_edm_churn, _churn_generator, _clamp_mask_, _compute_edm_gamma,
                                               _score_from_probs, _zero_mask_)

    B, S = x_init.shape
    T = S // C.BPT
    dev = x_init.device
    n_int = len(sigmas) - 1
    full = prefix_full.clone()
    mask = prefix_mask.clone()
    x = x_init.clone().float()
    _clamp_mask_(x, full, mask)
    x0_hat = torch.zeros_like(x)
    _clamp_mask_(x0_hat, full, mask)
    info = {}
    plen_t = torch.as_tensor(plens, device=dev)
    tok = torch.arange(T, device=dev)[None]
    ans_tok = tok >= plen_t[:, None]                                                    # [B, T]
    clamp_step = torch.full((B, T), -1, dtype=torch.long, device=dev)
    reset_done = False
    if var.stagger is not None:
        # sigma_base(u): log-linear interpolation of the schedule, u_k = 1 - k/n_int at sigmas[k]
        lsig = sigmas.double().log()
        ugrid = 1 - torch.arange(len(sigmas), device=dev, dtype=torch.float64) / n_int
        L = (S // C.BPT - plen_t).clamp_min(1).double()
        rel = ((tok - plen_t[:, None]).clamp_min(0).double() / L[:, None])               # [B, T]
        off = (var.stagger * rel).repeat_interleave(C.BPT, 1)                              # [B, S]

        def sig_at(k):
            u = (1 - (1 + var.stagger) * k / n_int + off).clamp(0, 1)
            # np.interp on a decreasing grid: flip
            return torch.from_numpy(np.interp(u.cpu().numpy(), ugrid.flip(0).cpu().numpy(),
                                              lsig.flip(0).cpu().numpy())).to(dev).exp().float()

        sig_cur = sig_at(0)
    for i in range(n_int):
        _clamp_mask_(x, full, mask)
        if var.sc_reset is not None and not reset_done and float(sigmas[i]) <= var.sc_reset:
            x0_hat = torch.zeros_like(x)
            _clamp_mask_(x0_hat, full, mask)
            reset_done = True
            info["sc_reset_step"] = i
        if var.stagger is None:
            gamma_i = _compute_edm_gamma(sigmas[i], num_intervals=n_int, s_churn=stc.s_churn, s_tmin=stc.s_tmin,
                                         s_tmax=stc.s_tmax) if stc.enabled else 0.0
            gen = _churn_generator(x.device, churn_seed, i) if gamma_i > 0.0 else None
            x_state, sigma_state = _apply_edm_churn(x, sigmas[i], gamma=gamma_i, s_noise=stc.s_noise, generator=gen)
            _clamp_mask_(x_state, full, mask)
            cond_in = x0_hat
            _clamp_mask_(cond_in, full, mask)
            use_tok = var.tok_T is not None and (var.tok_sigma is None or float(sigmas[i]) <= var.tok_sigma)
            probs = _denoise(model, cfg, x_state, sigma_state, cond_in, var, use_tok)
            _clamp_mask_(probs, full, mask)
            d = -sigma_state * _score_from_probs(probs, x_state, sigma_state)
            _zero_mask_(d, mask)
            d = _temper(d, var, i, n_int)
            x = x_state + (sigmas[i + 1] - sigma_state) * d
        else:
            sig_next = sig_at(i + 1)
            moving = sig_next < sig_cur
            g = torch.where(moving, _edm_gamma_vec(sig_cur, n_int, stc), torch.zeros_like(sig_cur))
            sig_hat = sig_cur * (1 + g)
            gen = _churn_generator(x.device, churn_seed, i)
            eps = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype) * float(stc.s_noise)
            x_state = x + (sig_hat.square() - sig_cur.square()).clamp_min(0).sqrt() * eps
            _clamp_mask_(x_state, full, mask)
            cond_in = x0_hat
            _clamp_mask_(cond_in, full, mask)
            ansb = (~mask).float()
            if var.stagger_emb == "gmean":
                s_emb = ((sig_hat.log() * ansb).sum(1) / ansb.sum(1).clamp_min(1)).exp()
            else:
                s_emb = (sig_hat * ansb).amax(1)
            raw = canonicalize_continuous_logits(model(x_state, s_emb, cond_in), is_cont_tokens=False)
            mf = (float(cfg.model.matched_filter_scale) * (x_state.float() - float(cfg.model.matched_filter_center))
                  / sig_hat.square().clamp_min(1e-12))
            clip = getattr(cfg.model, "matched_filter_clip", None)
            if clip is not None:
                mf = mf.clamp(-float(clip), float(clip))
            probs = torch.sigmoid((raw.float() + mf)).to(x.dtype)
            _clamp_mask_(probs, full, mask)
            d = (x_state - probs) / sig_hat
            _zero_mask_(d, mask)
            d = _temper(d, var, i, n_int)
            x = x_state + (sig_next - sig_hat) * d
            sig_cur = sig_next
        _clamp_mask_(x, full, mask)
        if var.sc_decay is not None:
            x0_hat = var.sc_decay * x0_hat + (1 - var.sc_decay) * probs
            _clamp_mask_(x0_hat, full, mask)
        else:
            x0_hat = probs
        if var.clamp_tau is not None and (var.clamp_sigma is None or float(sigmas[i]) <= var.clamp_sigma):
            pt = probs.float().view(B, T, C.BPT)
            conf = (torch.maximum(pt, 1 - pt) > var.clamp_tau).all(-1) & ans_tok & (clamp_step < 0)
            if var.clamp_mode == "prefix":
                done = ~ans_tok | (clamp_step >= 0)
                # tokens after the clamped prefix that are confident, cut at the first non-confident one
                run = torch.cumprod((done | conf).long(), 1).bool()
                conf = conf & run
            if bool(conf.any()):
                clamp_step[conf] = i
                nb = conf.repeat_interleave(C.BPT, 1)
                hard = (probs > 0.5).to(x.dtype)
                full = torch.where(nb, hard, full)
                mask = mask | nb
                _clamp_mask_(x, full, mask)
                x0_hat = x0_hat.clone()
                _clamp_mask_(x0_hat, full, mask)
    _clamp_mask_(x, full, mask)
    probs = _denoise(model, cfg, x, sigmas[-1], x0_hat, var, var.tok_T is not None)
    _clamp_mask_(probs, full, mask)
    if var.clamp_tau is not None:
        info["clamp_step"] = clamp_step.cpu().numpy()
    return probs, info

#!/usr/bin/env python
"""
GSM8K diagnostics for CoBit-S / CoBit-M on TinyGSM (250k, EMA, float32 RoPE): Maj@N, SC reset / decay, confidence
clamping, staggered noise. Sampler: diag_sampler.sample (= DDIMSampler, ddim_entropic, 256 steps, gamma 0.31, SC carry,
CoBit-S's schedule). Seeds: seed k uses initial noise x_T(example + 7,777,777 k) and churn seed
(CHURN_SEED * 1,000,003 + batch index + 99,991 k) % MOD with batch 32, as compute_demand.py, so seeds 0 and 1 of
variant base reproduce compute_demand's base / M runs problem by problem.

Per run (model, variant, seed): results_tinygsm/diagnostics/raw/<model>_<variant>_seed<k>.json with per-problem
correctness, executed answer, failure status (ok / wrong / no_function / exec_error / timeout / no_answer), number of
invalid token ids (>= vocab) and the decoded answer text; clamp variants also store per-token clamp steps.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import time

import numpy as np
import torch

import common as C
import diag_sampler as DS
from gsm8k_eval import configure_stochastic
from traj_tinygsm import CHURN_SEED, MOD, x_T

def out_dir():
    d = C.REPO / "experiments/denoising_mse/results_tinygsm/diagnostics/raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_answer(text: str, timeout_s: float = 5.0):
    """Execute a generated solution as sandbox_gsm8k.evaluate_samples does; return (answer, status)."""
    import sandbox_gsm8k as SB

    try:
        code = SB._extract_code(text)
        with SB._time_limit(timeout_s):
            ns = SB._safe_exec_no_timer(code)
            fn = ns.get("simple_math_problem", None)
            if fn is None:
                return None, "no_function"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                out = fn()
    except SB._Timeout:
        return None, "timeout"
    except Exception:
        return None, "exec_error"
    v = SB._to_number(out)
    if v is None:
        v = SB._extract_text_answer(buf.getvalue())
    if v is None:
        return None, "no_answer"
    if isinstance(v, float) and not np.isfinite(v):
        return None, "no_answer"
    return v, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="S")
    ap.add_argument("--variants", default="base")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--gamma", type=float, default=0.31)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    import sandbox_gsm8k as SB
    from diffusion.continuous.processes import ContinuousForwardProcess
    from diffusion.continuous.samplers import DDIMSampler

    dev = torch.device("cuda:0")
    cfg, model = C.load_model(args.model, dev)
    configure_stochastic(cfg, args.gamma, args.steps)
    run_dir = str(C.REPO / C.MODELS["S"]["run_dir"])
    probe = DDIMSampler(model, ContinuousForwardProcess(cfg), cfg)
    sigmas = probe.sigmas.prepare(schedule="entropic", num_steps=args.steps, entropy_run_dir=run_dir)
    stc = probe.sigmas.resolve_stochastic_cfg(entropy_run_dir=run_dir)
    test = C.GSM8KTest()
    n = len(test) if args.limit is None else min(args.limit, len(test))
    lo, _, hi = args.seeds.partition("-")
    seeds = list(range(int(lo), int(hi) + 1)) if hi else [int(s) for s in args.seeds.split(",")]
    od = out_dir()
    n_tok = len(test.tok)
    for vname in args.variants.split(","):
        var = DS.parse_variant(vname)
        for seed in seeds:
            tag = f"{args.model}_{vname}_seed{seed}{('_' + args.tag) if args.tag else ''}" \
                  f"{'' if args.limit is None else f'_n{n}'}"
            f = od / f"{tag}.json"
            if f.exists():
                print(f"[diag] {tag}: exists, skipped", flush=True)
                continue
            t0 = time.perf_counter()
            rec = dict(correct=[], answer=[], status=[], invalid=[], text=[], clamp_step=[], plen=[])
            for s in range(0, n, args.batch):
                idx = list(range(s, min(s + args.batch, n)))
                x0p, pm, plens = test.batch(idx)
                x0p, pm = x0p.to(dev), pm.to(dev)
                xT = torch.where(pm, x0p, x_T([e + 7_777_777 * seed for e in idx], x0p.shape[1], sigmas[0], dev))
                churn = (CHURN_SEED * 1_000_003 + s // args.batch + 99_991 * seed) % MOD
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    probs, info = DS.sample(model, cfg, stc, sigmas, xT, x0p, pm, plens, churn, var)
                ids = C.bits_to_ids((probs.float() >= 0.5).long().cpu())
                for b, i in enumerate(idx):
                    suf = ids[b, plens[b]:].tolist()
                    text = test.tok.decode([t if 0 <= t < n_tok else test.tok.eos_token_id for t in suf],
                                           skip_special_tokens=True)
                    ans, status = run_answer(text)
                    gold = SB._extract_gold_answer(test.records[i]["response_ground_truth"])
                    ok = bool(SB._numbers_equal(ans, gold))
                    rec["correct"].append(int(ok))
                    rec["answer"].append(ans if ans is None or isinstance(ans, int) else float(ans))
                    rec["status"].append("correct" if ok else ("wrong" if status == "ok" else status))
                    rec["invalid"].append(int(sum(1 for t in suf if t >= n_tok)))
                    rec["text"].append(text)
                    rec["plen"].append(int(plens[b]))
                    if "clamp_step" in info:
                        rec["clamp_step"].append(info["clamp_step"][b].tolist())
            res = dict(model=args.model, variant=vname, seed=seed, n=n, gamma=args.gamma, steps=args.steps,
                       accuracy=float(np.mean(rec["correct"])), wall_sec=time.perf_counter() - t0, **rec)
            f.write_text(json.dumps(res))
            st = {k: rec["status"].count(k) for k in sorted(set(rec["status"]))}
            print(f"[diag] {tag}: acc {100 * res['accuracy']:.2f}%  {st}  {res['wall_sec'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

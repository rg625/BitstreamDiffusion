#!/usr/bin/env python
"""
Compute-allocation diagnostic for CoBit-S on LM1B (experiments/compute_allocation/PLAN.md).

Subcommands
  reference        Run 1: logged reference trajectory (boundary states, per-step logs,
                   final samples, per-sample scores).
  probe            Run 2: branches refine[s] / nochurn[s] / null from Run 1 saved states.
                   One output file per branch; finished branches are skipped.
  identity-check   GPU check: the sampler with no new options vs. the sampler on `main`.
  analyze          CPU only: paired metrics, bootstrap CIs, settling-time, concentration.

Large artefacts go to --out-root (default runs/compute_allocation, gitignored);
small per-branch score files and analysis outputs go to --results-root
(default experiments/compute_allocation/results).

Production settings are taken from configs/lm1b/continuous/eval/rate_eval_seeds.py
(the γ=0.200 spec): entropic grid, 255 σ points (254 Euler steps + final denoise),
terminal σ 0.08, churn s_churn=51 / S_noise=1.003 over the full entropy band,
self-conditioning `carry`, bf16 autocast + TF32, torch.compile, 1024 samples in
two chunks of 512 seeded exactly as evaluation/generation_driver.py does.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("TORCHINDUCTOR_DISABLE_CUDAGRAPHS", "1")  # as evaluation/run_eval.py

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

CONFIG = REPO / "configs/lm1b/continuous/eval/rate_eval_seeds.py"
DEFAULT_OUT = REPO / "runs/compute_allocation"
DEFAULT_RESULTS = REPO / "experiments/compute_allocation/results"

N_SAMPLES = 1024
MICRO_BATCH = 512  # production generation micro-batch (cfg.evaluation.external_ppl.micro_batch_size)
SEG_LEN = 32
N_SEGMENTS = 8
BITS = 15
SEQ_TOK = 128
MOD = 2**63 - 1
GPU_HOUR_CAP = 6.0  # user-set cap for this session (RTX 4090 GPU-hours)
BASE_SEED = 42  # cfg.evaluation.external_ppl.seed (EVAL_SEED default)


# -----------------------------------------------------------------------------
# small utilities
# -----------------------------------------------------------------------------
def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=REPO) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=1)
    tmp.replace(path)


def _atomic_torch_save(path: Path, obj: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _ledger_path(results_root: Path) -> Path:
    return results_root / "gpu_ledger.jsonl"


def _ledger_total_hours(results_root: Path) -> float:
    p = _ledger_path(results_root)
    if not p.exists():
        return 0.0
    tot = 0.0
    for line in p.read_text().splitlines():
        if line.strip():
            tot += float(json.loads(line)["gpu_seconds"])
    return tot / 3600.0


def _ledger_add(results_root: Path, what: str, seconds: float, **extra) -> None:
    p = _ledger_path(results_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = dict(time=_now(), what=what, gpu_seconds=float(seconds), **extra)
    with p.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _check_cap(results_root: Path, upcoming_hours: float = 0.0) -> None:
    used = _ledger_total_hours(results_root)
    if used + upcoming_hours > GPU_HOUR_CAP:
        raise SystemExit(
            f"STOP: GPU-hour cap would be exceeded (used {used:.3f} h + upcoming "
            f"{upcoming_hours:.3f} h > cap {GPU_HOUR_CAP} h)."
        )


def batch_seed(batch: int) -> int:
    return BASE_SEED + int(batch)


# -----------------------------------------------------------------------------
# model / sampler context
# -----------------------------------------------------------------------------
class CountingModel:
    """Thin call counter around the (compiled) denoiser; counts NFE."""

    def __init__(self, inner):
        import torch

        self.inner = inner
        self.calls = 0
        self._torch = torch

    def __call__(self, *a, **k):
        self.calls += 1
        return self.inner(*a, **k)

    def parameters(self):
        return self.inner.parameters()

    def buffers(self):
        return self.inner.buffers()


class Ctx:
    """Loaded production config, model, sampler and scorer."""

    def __init__(self, *, gamma: float = 0.2, compile_model: bool = True, need_scorer: bool = True,
                 sampler_module=None):
        import torch
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = 64  # as evaluation/run_eval.py

        from evaluation.utils import load_config, unwrap_all, load_checkpoint
        from evaluation.sampling_specs import build_sampling_specs, _tag_for_spec
        from evaluation.generation_driver import _resolve_autocast_dtype, _stable_int_hash
        from diffusion.continuous.processes import ContinuousForwardProcess
        from models import create_model
        from utils.ema import EMA

        self.torch = torch
        cfg = load_config(str(CONFIG))
        self.cfg = cfg
        specs = build_sampling_specs(cfg=cfg, metric_cfg=cfg.evaluation.external_ppl)
        prod = [s for s in specs if abs(float(s["s_churn"]) - 0.2 * (int(s["target_nfe"]) - 1)) < 1e-9]
        assert len(prod) == 1, specs
        spec = dict(prod[0])
        if abs(gamma - 0.2) > 1e-12:
            spec["s_churn"] = float(gamma * (int(spec["target_nfe"]) - 1))
            spec["tag"] = _tag_for_spec(spec)
        self.spec = spec
        self.gamma = float(gamma)
        self.tag = str(spec["tag"])
        self.tag_hash = _stable_int_hash(self.tag)

        # apply the spec's stochastic settings exactly as GenerationDriver does
        st = cfg.evaluation.stochastic
        for spec_k, cfg_k in {
            "stochastic_enabled": "enabled", "s_churn": "s_churn", "s_noise": "s_noise",
            "window_mode": "window_mode", "entropy_quantile_lo": "entropy_quantile_lo",
            "entropy_quantile_hi": "entropy_quantile_hi", "s_tmin": "s_tmin", "s_tmax": "s_tmax",
            "entropy_fallback": "entropy_fallback",
        }.items():
            setattr(st, cfg_k, spec[spec_k])

        self.device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(BASE_SEED)  # set_global_seed equivalent (irrelevant: chunks fork the RNG)
        torch.cuda.manual_seed_all(BASE_SEED)

        model = create_model(cfg).to(self.device)
        ema = EMA(unwrap_all(model), decay=0.0)
        load_checkpoint(model, ema, REPO / cfg.evaluation.checkpoint_path, self.device, apply_ema=True)
        model.eval()
        self.compiled = bool(compile_model and cfg.evaluation.use_compile)
        if self.compiled:
            model = torch.compile(model, mode=str(cfg.evaluation.compile_mode), fullgraph=False)
            model.eval()
        self.model = CountingModel(model)
        self.proc = ContinuousForwardProcess(cfg)
        if sampler_module is None:
            from diffusion.continuous import samplers as sampler_module
        self.sampler = sampler_module.DDIMSampler(self.model, self.proc, cfg)
        self.amp_dtype = _resolve_autocast_dtype(cfg, "auto", self.device)
        self.S_bits = int(cfg.data.sequence_len)
        self.entropy_dir = str(REPO / cfg.evaluation.entropy_run_dir)

        self.scorer = None
        if need_scorer:
            from evaluation.external_perplexity import HFExternalPerplexityEvaluator

            ext = cfg.evaluation.external_ppl
            self.scorer_attn = str(getattr(ext, "attn_implementation", "sdpa"))
            try:
                self.scorer = HFExternalPerplexityEvaluator(
                    model_name=str(ext.hf_model_name),
                    device=self.device,
                    torch_dtype=str(ext.hf_dtype),
                    attn_implementation=self.scorer_attn,
                    use_amp=True,
                )
            except ValueError as e:
                # transformers<4.40 (env `sedd`: 4.38.1) has no SDPA path for GPT-2.
                if "scaled_dot_product_attention" not in str(e):
                    raise
                print(f"[ctx] GPT-2 scorer: {self.scorer_attn} unsupported here; using eager attention")
                self.scorer_attn = "eager"
                self.scorer = HFExternalPerplexityEvaluator(
                    model_name=str(ext.hf_model_name),
                    device=self.device,
                    torch_dtype=str(ext.hf_dtype),
                    attn_implementation="eager",
                    use_amp=True,
                )

    # ------------------------------------------------------------------
    def chunk_seed(self, seed: int, start: int) -> int:
        return (int(seed) * 1_000_003 + self.tag_hash * 10_007 + int(start)) % MOD

    def sample_kwargs(self) -> Dict[str, Any]:
        sp = self.spec
        return dict(
            schedule="entropic",
            num_steps=int(sp["num_steps"]),
            sigma_min_override=float(sp["terminal_sigma"]),
            sigma_max_override=None,
            entropic_blend_alpha=0.0,
            entropy_run_dir=self.entropy_dir,
            guidance_scale=float(sp["guidance_scale"]),
            sc_refresh_mode=str(sp["sc_refresh_mode"]),
            ati_eta=float(sp["ati_eta"]),
            return_probs=True,
            progress=False,
        )

    def run(
        self,
        *,
        seed: int,
        seeded_churn: bool = True,
        init_states: Optional[List[Dict[str, Any]]] = None,
        segment_overrides: Optional[Dict[int, Dict[str, Any]]] = None,
        save_states_at: Optional[List[int]] = None,
        logger=None,
        n_samples: int = N_SAMPLES,
    ) -> Dict[str, Any]:
        """Sample n_samples in production chunks; returns final probs/bits and saved states."""
        torch = self.torch
        dev_idx = [int(self.device.index or 0)]
        probs_all, states_all = [], []
        self.model.calls = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for c, start in enumerate(range(0, n_samples, MICRO_BATCH)):
            end = min(start + MICRO_BATCH, n_samples)
            cs = self.chunk_seed(seed, start)
            chunk_states: Dict[int, Dict[str, Any]] = {}
            ref_zero = torch.zeros(end - start, self.S_bits, device=self.device)
            mask_zero = torch.zeros(end - start, self.S_bits, device=self.device, dtype=torch.bool)
            if logger is not None:
                logger.begin_chunk(start, end)
            with torch.random.fork_rng(devices=dev_idx, enabled=True):
                torch.manual_seed(cs)
                torch.cuda.manual_seed_all(cs)
                with torch.autocast(device_type="cuda", enabled=True, dtype=self.amp_dtype):
                    kw = self.sample_kwargs()
                    extra: Dict[str, Any] = {}
                    if seeded_churn:
                        extra["churn_noise_seed"] = cs
                    if init_states is not None:
                        extra["init_state"] = init_states[c]
                    if segment_overrides:
                        extra["segment_overrides"] = segment_overrides
                        extra["segment_len"] = SEG_LEN
                    if save_states_at:
                        extra["save_states_at"] = save_states_at
                        extra["state_hook"] = lambda s, d=chunk_states: d.__setitem__(s["i"], s)
                    if logger is not None:
                        extra["step_callback"] = logger.callback
                    _, probs = self.sampler.sample(
                        end - start,
                        self.S_bits,
                        conditioning_prefix_full=ref_zero,
                        cond_prefix_mask=mask_zero,
                        **kw,
                        **extra,
                    )
            probs_all.append(probs.float())
            states_all.append(chunk_states)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        probs = torch.cat(probs_all, 0)
        bits = (probs > 0.5).to(torch.uint8)
        return dict(probs=probs, bits=bits, states=states_all, wall=wall, nfe_calls=self.model.calls,
                    chunk_seeds=[self.chunk_seed(seed, s) for s in range(0, n_samples, MICRO_BATCH)])

    # ------------------------------------------------------------------
    def score(self, bits) -> Dict[str, Any]:
        """Decode and score with GPT-2-large, per sample (production 'full' view)."""
        torch = self.torch
        from utils.text_decode import decode_bitstreams_for_eval, decode_bitstreams_to_token_ids_for_eval
        from evaluation.text_metrics import token_unigram_entropy_from_token_ids

        bits_cpu = bits.cpu()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        texts = decode_bitstreams_for_eval(self.cfg, bits_cpu, mode="full", dataset_obj=None)
        tok = decode_bitstreams_to_token_ids_for_eval(self.cfg, bits_cpu, dataset_obj=None)
        ent = [token_unigram_entropy_from_token_ids(row) for row in tok]
        per = self.scorer.score_texts_per_sample(texts)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        nll = [p[0] for p in per]
        ntok = [p[1] for p in per]
        tot_nll = 0.0
        for v in nll:  # same summation order as HFExternalPerplexityEvaluator.score_texts
            tot_nll += v
        tot_tok = int(sum(ntok))
        return dict(
            texts=texts,
            tokens=tok.to(torch.int16),
            nll=nll,
            ntok=ntok,
            entropy=ent,
            genppl=float(math.exp(tot_nll / tot_tok)),
            nll_per_token=float(tot_nll / tot_tok),
            mean_entropy=float(sum(ent) / len(ent)),
            score_wall=wall,
        )


# -----------------------------------------------------------------------------
# per-step logger for Run 1
# -----------------------------------------------------------------------------
class StepLogger:
    """Collects per-step token guesses, per-token mean bit entropy, churn sizes."""

    def __init__(self, n_steps_total: int, n_samples: int):
        import torch

        self.torch = torch
        T = n_steps_total  # 254 steps + final denoise
        self.tok = torch.zeros(T, n_samples, SEQ_TOK, dtype=torch.int16)
        self.bit_ent = torch.zeros(T, n_samples, SEQ_TOK, dtype=torch.float16)
        self.eps_rms = torch.zeros(T, n_samples, dtype=torch.float32)
        self.sigma = torch.zeros(T, dtype=torch.float64)
        self.gamma = torch.zeros(T, dtype=torch.float64)
        self.powers = None
        self.start = 0
        self.end = 0

    def begin_chunk(self, start: int, end: int) -> None:
        self.start, self.end = start, end

    def callback(self, *, i, sub, n_sub, final, sigma, sigma_next, gamma, D, x, eps) -> None:
        torch = self.torch
        if sub != n_sub - 1 and not final:
            return  # log once per original step (after its last sub-step)
        B = D.size(0)
        p = D.float().view(B, SEQ_TOK, BITS)
        if self.powers is None:
            self.powers = (2 ** torch.arange(BITS - 1, -1, -1, device=D.device)).to(torch.int32)
        tok = ((p > 0.5).to(torch.int32) * self.powers).sum(-1)
        pc = p.clamp(1e-6, 1 - 1e-6)  # 1-1e-12 rounds to 1.0 in fp32 -> log(0)
        h = -(pc * pc.log() + (1 - pc) * (1 - pc).log()) / math.log(2.0)  # bits
        s, e = self.start, self.end
        self.tok[i, s:e] = tok.to(torch.int16).cpu()
        self.bit_ent[i, s:e] = h.mean(-1).to(torch.float16).cpu()
        if eps is not None and sub == 0:
            self.eps_rms[i, s:e] = eps.float().pow(2).mean(-1).sqrt().cpu()
        self.sigma[i] = float(sigma)
        self.gamma[i] = float(gamma)


# -----------------------------------------------------------------------------
# I/O for branch outputs
# -----------------------------------------------------------------------------
def _branch_paths(out_root: Path, results_root: Path, batch: int, name: str):
    big = out_root / f"batch{batch}" / f"{name}.pt"
    small = results_root / f"batch{batch}" / f"{name}.json"
    return big, small


def _write_branch(out_root, results_root, batch, name, run, sc, meta) -> None:
    big, small = _branch_paths(out_root, results_root, batch, name)
    _atomic_torch_save(big, dict(bits=run["bits"].cpu(), tokens=sc["tokens"], texts=sc["texts"]))
    _atomic_json(small, dict(
        meta=meta,
        genppl=sc["genppl"],
        nll_per_token=sc["nll_per_token"],
        mean_entropy=sc["mean_entropy"],
        per_sample=dict(nll=sc["nll"], ntok=sc["ntok"], entropy=sc["entropy"]),
    ))


def _meta(ctx: Ctx, **extra) -> Dict[str, Any]:
    import torch

    return dict(
        time=_now(),
        commit=_git_commit(),
        gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        compiled=ctx.compiled,
        scorer_attn=getattr(ctx, "scorer_attn", None),
        transformers=__import__("transformers").__version__,
        amp_dtype=str(ctx.amp_dtype),
        tag=ctx.tag,
        gamma_target=ctx.gamma,
        s_churn=float(ctx.spec["s_churn"]),
        num_sigma_points=int(ctx.spec["num_steps"]),
        terminal_sigma=float(ctx.spec["terminal_sigma"]),
        micro_batch=MICRO_BATCH,
        pid=os.getpid(),
        **extra,
    )


# -----------------------------------------------------------------------------
# subcommands
# -----------------------------------------------------------------------------
def cmd_reference(args) -> None:
    out_root, results_root = Path(args.out_root), Path(args.results_root)
    big, small = _branch_paths(out_root, results_root, args.batch, "reference")
    if args.step_logs_only:
        return _reference_step_logs_only(args, out_root, results_root, big)
    if small.exists() and big.exists() and not args.force:
        print(f"[reference] batch {args.batch} already done: {small}")
        return
    _check_cap(results_root, upcoming_hours=0.05)
    t_proc = time.perf_counter()
    ctx = Ctx(compile_model=not args.no_compile)
    n_points = int(ctx.spec["num_steps"])
    boundaries = [SEG_LEN * s for s in range(N_SEGMENTS)]
    logger = StepLogger(n_points, N_SAMPLES) if args.batch == 0 or args.log_steps else None
    seed = batch_seed(args.batch)
    run = ctx.run(seed=seed, seeded_churn=True, save_states_at=boundaries, logger=logger)
    print(f"[reference] generation {run['wall']:.1f}s, model calls {run['nfe_calls']}", flush=True)

    # boundary states, merged over chunks
    torch = ctx.torch
    sdir = out_root / f"batch{args.batch}" / "states"
    for s, i0 in enumerate(boundaries):
        per_chunk = [run["states"][c][i0] for c in range(len(run["states"]))]
        merged = dict(
            i=i0,
            x=torch.cat([p["x"].cpu() for p in per_chunk], 0),
            x0_hat=torch.cat([p["x0_hat"].cpu() for p in per_chunk], 0),
            sigmas=per_chunk[0]["sigmas"].cpu(),
            chunk_seeds=run["chunk_seeds"],
        )
        _atomic_torch_save(sdir / f"seg{s}.pt", merged)
    if logger is not None:
        _atomic_torch_save(out_root / f"batch{args.batch}" / "step_logs.pt", dict(
            tok=logger.tok, bit_ent=logger.bit_ent, eps_rms=logger.eps_rms,
            sigma=logger.sigma, gamma=logger.gamma,
        ))

    sc = ctx.score(run["bits"])
    print(f"[reference] GenPPL {sc['genppl']:.3f}  entropy {sc['mean_entropy']:.4f}  "
          f"scoring {sc['score_wall']:.1f}s", flush=True)
    total = time.perf_counter() - t_proc
    meta = _meta(ctx, batch=args.batch, seed=seed, branch="reference",
                 generation_wall_sec=run["wall"], scoring_wall_sec=sc["score_wall"],
                 process_wall_sec=total, nfe_calls_total=run["nfe_calls"],
                 nfe_per_sample=run["nfe_calls"] // math.ceil(N_SAMPLES / MICRO_BATCH),
                 chunk_seeds=run["chunk_seeds"], step_logs=logger is not None)
    _write_branch(out_root, results_root, args.batch, "reference", run, sc, meta)
    _ledger_add(results_root, f"reference batch{args.batch}", total)
    print(f"[reference] wrote {small}")


def _reference_step_logs_only(args, out_root: Path, results_root: Path, big: Path) -> None:
    """Re-run the reference trajectory only to (re)write step logs; must reproduce its bits."""
    _check_cap(results_root, upcoming_hours=0.05)
    t_proc = time.perf_counter()
    ctx = Ctx(compile_model=not args.no_compile, need_scorer=False)
    torch = ctx.torch
    logger = StepLogger(int(ctx.spec["num_steps"]), N_SAMPLES)
    run = ctx.run(seed=batch_seed(args.batch), seeded_churn=True, logger=logger)
    saved = torch.load(big, map_location="cpu")
    same = bool(torch.equal(saved["bits"], run["bits"].cpu()))
    _atomic_torch_save(out_root / f"batch{args.batch}" / "step_logs.pt", dict(
        tok=logger.tok, bit_ent=logger.bit_ent, eps_rms=logger.eps_rms,
        sigma=logger.sigma, gamma=logger.gamma,
    ))
    total = time.perf_counter() - t_proc
    res = dict(time=_now(), batch=args.batch, rerun_bits_identical_to_reference=same,
               n_bits_differing=int((saved["bits"] != run["bits"].cpu()).sum()),
               generation_wall_sec=run["wall"], process_wall_sec=total, commit=_git_commit())
    _atomic_json(results_root / f"batch{args.batch}" / "step_logs_rerun.json", res)
    _ledger_add(results_root, f"reference step-logs rerun batch{args.batch}", total)
    print(json.dumps(res))


def _branches(args) -> List[str]:
    if args.all:
        return ["null"] + [f"{k}_{s}" for k in ("refine", "nochurn") for s in range(N_SEGMENTS)]
    out = []
    for k in args.kind:
        if k == "null":
            out.append("null")
        else:
            out += [f"{k}_{s}" for s in args.segment]
    return out


def cmd_probe(args) -> None:
    out_root, results_root = Path(args.out_root), Path(args.results_root)
    todo = []
    for name in _branches(args):
        big, small = _branch_paths(out_root, results_root, args.batch, name)
        if small.exists() and big.exists() and not args.force:
            print(f"[probe] skip {name} (done)")
            continue
        todo.append(name)
    if not todo:
        return
    _check_cap(results_root, upcoming_hours=0.02 * len(todo))
    ctx = Ctx(compile_model=not args.no_compile)
    torch = ctx.torch
    sdir = out_root / f"batch{args.batch}" / "states"
    seed = batch_seed(args.batch)
    for name in todo:
        _check_cap(results_root, upcoming_hours=0.02)
        t_br = time.perf_counter()
        if name == "null":
            kind, s = "null", 0
            overrides = None
        else:
            kind, s = name.split("_")[0], int(name.split("_")[1])
            overrides = {s: ({"substeps": int(args.substeps)} if kind == "refine" else {"gamma": 0.0})}
        st = torch.load(sdir / f"seg{s}.pt", map_location="cpu")
        n_chunks = math.ceil(N_SAMPLES / MICRO_BATCH)
        init_states = []
        for c in range(n_chunks):
            a, b = c * MICRO_BATCH, min((c + 1) * MICRO_BATCH, N_SAMPLES)
            init_states.append(dict(i=int(st["i"]), x=st["x"][a:b].to(ctx.device),
                                    x0_hat=st["x0_hat"][a:b].to(ctx.device), sigmas=st["sigmas"]))
        assert [ctx.chunk_seed(seed, c * MICRO_BATCH) for c in range(n_chunks)] == list(st["chunk_seeds"])
        run = ctx.run(seed=seed, seeded_churn=True, init_states=init_states, segment_overrides=overrides)
        sc = ctx.score(run["bits"])
        total = time.perf_counter() - t_br
        print(f"[probe] {name}: GenPPL {sc['genppl']:.3f} entropy {sc['mean_entropy']:.4f} "
              f"gen {run['wall']:.1f}s score {sc['score_wall']:.1f}s calls {run['nfe_calls']}", flush=True)
        meta = _meta(ctx, batch=args.batch, seed=seed, branch=name, kind=kind, segment=s,
                     start_step=int(st["i"]), overrides={str(k): v for k, v in (overrides or {}).items()},
                     generation_wall_sec=run["wall"], scoring_wall_sec=sc["score_wall"],
                     process_wall_sec=total, nfe_calls_total=run["nfe_calls"],
                     nfe_per_sample=run["nfe_calls"] // n_chunks)
        _write_branch(out_root, results_root, args.batch, name, run, sc, meta)
        _ledger_add(results_root, f"probe batch{args.batch} {name}", total)


def cmd_identity_check(args) -> None:
    """Production-scale check: new sampler without options == sampler on `main` (one 512 chunk)."""
    import importlib.util
    import tempfile

    results_root = Path(args.results_root)
    _check_cap(results_root, upcoming_hours=0.05)
    t_proc = time.perf_counter()
    src = subprocess.check_output(["git", "show", "main:diffusion/continuous/samplers.py"], cwd=REPO)
    tmpd = Path(tempfile.mkdtemp())
    (tmpd / "orig_samplers.py").write_bytes(src)
    spec = importlib.util.spec_from_file_location("orig_samplers", tmpd / "orig_samplers.py")
    orig = importlib.util.module_from_spec(spec)
    sys.modules["orig_samplers"] = orig
    spec.loader.exec_module(orig)

    ctx = Ctx(compile_model=not args.no_compile, need_scorer=False)
    torch = ctx.torch
    n = int(args.n)
    res = {}
    new = ctx.run(seed=BASE_SEED, seeded_churn=False, n_samples=n)
    from diffusion.continuous import samplers as cur_mod
    ctx.sampler = orig.DDIMSampler(ctx.model, ctx.proc, ctx.cfg)
    old = ctx.run(seed=BASE_SEED, seeded_churn=False, n_samples=n)
    ctx.sampler = cur_mod.DDIMSampler(ctx.model, ctx.proc, ctx.cfg)
    new2 = ctx.run(seed=BASE_SEED, seeded_churn=False, n_samples=n)
    res = dict(
        n_samples=n,
        probs_bit_identical=bool(torch.equal(new["probs"], old["probs"])),
        bits_identical=bool(torch.equal(new["bits"], old["bits"])),
        max_abs_prob_diff=float((new["probs"] - old["probs"]).abs().max()),
        rerun_probs_identical=bool(torch.equal(new["probs"], new2["probs"])),
        nfe_calls_new=new["nfe_calls"], nfe_calls_old=old["nfe_calls"],
        wall_new=new["wall"], wall_old=old["wall"],
    )
    total = time.perf_counter() - t_proc
    res["meta"] = _meta(ctx, process_wall_sec=total)
    _atomic_json(results_root / "identity_check.json", res)
    _ledger_add(results_root, "identity-check", total)
    print(json.dumps({k: v for k, v in res.items() if k != "meta"}, indent=1))


def cmd_analyze(args) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analysis

    analysis.main(Path(args.out_root), Path(args.results_root), batches=args.batches)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--results-root", default=str(DEFAULT_RESULTS))
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("reference")
    r.add_argument("--batch", type=int, default=0)
    r.add_argument("--log-steps", action="store_true", help="per-step logs (always on for batch 0)")
    r.add_argument("--no-compile", action="store_true")
    r.add_argument("--force", action="store_true")
    r.add_argument("--step-logs-only", action="store_true",
                   help="rerun the trajectory to rewrite step logs; checks bits match the saved reference")

    p = sub.add_parser("probe")
    p.add_argument("--kind", nargs="+", choices=["refine", "nochurn", "null"], default=[])
    p.add_argument("--segment", type=int, nargs="+", default=[])
    p.add_argument("--all", action="store_true", help="null + refine[0..7] + nochurn[0..7]")
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--force", action="store_true")

    ic = sub.add_parser("identity-check")
    ic.add_argument("--n", type=int, default=512)
    ic.add_argument("--no-compile", action="store_true")

    a = sub.add_parser("analyze")
    a.add_argument("--batches", type=int, nargs="+", default=None)

    args = ap.parse_args()
    {"reference": cmd_reference, "probe": cmd_probe, "identity-check": cmd_identity_check,
     "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()

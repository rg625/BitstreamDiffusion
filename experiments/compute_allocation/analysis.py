"""
CPU analyses for the compute-allocation diagnostic (PLAN §5, §7, §8).

Called by `diagnose.py analyze`. Reads per-branch score files
(results/batch{b}/{name}.json), branch artefacts (runs/compute_allocation/batch{b}/{name}.pt)
and the Run 1 step logs; writes results/analysis/*.json and *.csv.

Definitions (fixed before Run 2 data existed; see NOTEBOOK.md):
  ΔNLL/token   ratio-of-sums change: Σnll_b/Σtok_b − Σnll_r/Σtok_r (= Δ log GenPPL)
  Δentropy     mean over samples of (entropy_b − entropy_r)
  tokens changed  fraction of the 1024×128 token ids differing from the reference;
                  "net" subtracts the null branch's fraction (same batch, else batch 0)
  CI           95% percentile bootstrap, 10,000 paired resamples of samples, rng seed 20261002
  material     CI excludes 0 and |ΔNLL/token| ≥ 0.01
  entropy flag |Δentropy| > 0.02  (ΔNLL then reported, not interpreted on its own)
  borderline   CI contains +0.01 or −0.01
  concentration  per-sample contribution c_i = (nll_b,i − nll_r,i) − r̄_r·(tok_b,i − tok_r,i)
                 (linearised ΔNLL·Σtok); share = Σ of the top-10% c_i in the direction of
                 the total / Σ c_i; also the |c| version.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

N_SEG = 8
SEG_LEN = 32
BITS = 15
SEQ = 128
MATERIAL = 0.01
ENT_FLAG = 0.02
N_BOOT = 10_000
BOOT_SEED = 20261002
SETTLE_WINDOW = 8  # steps a guess must stay unchanged to count as provisionally settled
SPECIAL_IDS = {0, 100, 101, 102, 103}  # BERT [PAD],[UNK],[CLS],[SEP],[MASK]


def _load_json(p: Path):
    return json.loads(p.read_text())


def _scores(results_root: Path, batch: int, name: str) -> Optional[Dict[str, np.ndarray]]:
    p = results_root / f"batch{batch}" / f"{name}.json"
    if not p.exists():
        return None
    d = _load_json(p)
    ps = d["per_sample"]
    return dict(nll=np.asarray(ps["nll"], dtype=np.float64), ntok=np.asarray(ps["ntok"], dtype=np.float64),
                ent=np.asarray(ps["entropy"], dtype=np.float64), genppl=d["genppl"],
                mean_entropy=d["mean_entropy"], meta=d["meta"], path=str(p))


def _artefact(out_root: Path, batch: int, name: str):
    p = out_root / f"batch{batch}" / f"{name}.pt"
    return torch.load(p, map_location="cpu") if p.exists() else None


# -----------------------------------------------------------------------------
# paired statistics
# -----------------------------------------------------------------------------
def _boot_idx(n: int) -> np.ndarray:
    rng = np.random.default_rng(BOOT_SEED)
    return rng.integers(0, n, size=(N_BOOT, n))


def paired_stats(ref: Dict, br: Dict) -> Dict:
    nll_r, t_r, e_r = ref["nll"], ref["ntok"], ref["ent"]
    nll_b, t_b, e_b = br["nll"], br["ntok"], br["ent"]
    n = nll_r.size
    d_nll = nll_b.sum() / t_b.sum() - nll_r.sum() / t_r.sum()
    d_ent = float(np.mean(e_b - e_r))
    idx = _boot_idx(n)
    bs_nll = nll_b[idx].sum(1) / t_b[idx].sum(1) - nll_r[idx].sum(1) / t_r[idx].sum(1)
    bs_ent = (e_b - e_r)[idx].mean(1)
    lo, hi = np.percentile(bs_nll, [2.5, 97.5])
    elo, ehi = np.percentile(bs_ent, [2.5, 97.5])
    excl0 = bool(lo > 0 or hi < 0)
    material = bool(excl0 and abs(d_nll) >= MATERIAL)
    borderline = bool((lo < MATERIAL < hi) or (lo < -MATERIAL < hi))

    rbar = nll_r.sum() / t_r.sum()
    c = (nll_b - nll_r) - rbar * (t_b - t_r)
    tot = c.sum()
    k = max(1, int(round(0.10 * n)))
    if tot != 0:
        top = np.sort(c)[::-1][:k] if tot > 0 else np.sort(c)[:k]
        share_signed = float(top.sum() / tot)
    else:
        share_signed = float("nan")
    ac = np.abs(c)
    share_abs = float(np.sort(ac)[::-1][:k].sum() / ac.sum()) if ac.sum() > 0 else float("nan")
    return dict(
        n=int(n),
        d_nll_tok=float(d_nll), ci_lo=float(lo), ci_hi=float(hi), ci_excludes_0=excl0,
        material=material, borderline=borderline,
        d_entropy=d_ent, d_entropy_ci=[float(elo), float(ehi)], entropy_flag=bool(abs(d_ent) > ENT_FLAG),
        genppl_ref=float(math.exp(nll_r.sum() / t_r.sum())), genppl_branch=float(math.exp(nll_b.sum() / t_b.sum())),
        d_genppl=float(math.exp(nll_b.sum() / t_b.sum()) - math.exp(nll_r.sum() / t_r.sum())),
        top10_share_signed=share_signed, top10_share_abs=share_abs,
        frac_samples_with_nll_change=float(np.mean(np.abs(nll_b - nll_r) > 0)),
    )


def _concat(a: Dict, b: Dict) -> Dict:
    return {k: np.concatenate([a[k], b[k]]) for k in ("nll", "ntok", "ent")}


# -----------------------------------------------------------------------------
# token-level breakdowns
# -----------------------------------------------------------------------------
def token_breakdown(ref_art, br_art, freq_rank: np.ndarray) -> Dict:
    rt = ref_art["tokens"].numpy().astype(np.int64)
    bt = br_art["tokens"].numpy().astype(np.int64)
    rb = ref_art["bits"].numpy().reshape(rt.shape[0], SEQ, BITS)
    bb = br_art["bits"].numpy().reshape(rt.shape[0], SEQ, BITS)
    ch = rt != bt
    pos_bins = [ch[:, i * 16:(i + 1) * 16].mean() for i in range(8)]
    ranks = freq_rank[rt]
    edges = [(1, 10), (11, 100), (101, 1000), (1001, 10**9)]
    freq = [float(ch[(ranks >= a) & (ranks <= b)].mean()) if ((ranks >= a) & (ranks <= b)).any() else float("nan")
            for a, b in edges]
    special = np.isin(rt, list(SPECIAL_IDS))
    bitflip = (rb != bb).mean(axis=(0, 1))
    return dict(
        frac_changed=float(ch.mean()),
        frac_samples_any_change=float(ch.any(1).mean()),
        by_position_bin16=[float(v) for v in pos_bins],
        by_ref_freq_rank={f"{a}-{b if b < 10**9 else 'inf'}": v for (a, b), v in zip(edges, freq)},
        special_vs_regular=[float(ch[special].mean()) if special.any() else float("nan"),
                            float(ch[~special].mean())],
        bitflip_by_position_msb_first=[float(v) for v in bitflip],
    )


def _freq_rank(ref_tokens: np.ndarray, vocab: int = 30522) -> np.ndarray:
    counts = np.bincount(ref_tokens.reshape(-1), minlength=vocab)
    order = np.argsort(-counts, kind="stable")
    rank = np.empty(vocab, dtype=np.int64)
    rank[order] = np.arange(1, vocab + 1)
    rank[counts == 0] = 10**8  # unseen in the reference corpus -> rarest bucket
    return rank


# -----------------------------------------------------------------------------
# Run 1 settling analyses
# -----------------------------------------------------------------------------
def _segment_of_step(t: np.ndarray) -> np.ndarray:
    return np.minimum(t // SEG_LEN, N_SEG - 1)


def _last_change(seq: torch.Tensor) -> torch.Tensor:
    """seq [T, ...]; index of the last t>=1 with seq[t]!=seq[t-1], else 0."""
    ch = seq[1:] != seq[:-1]
    T = ch.size(0)
    tidx = torch.arange(1, T + 1, dtype=torch.int32).view(-1, *([1] * (ch.dim() - 1)))
    return (ch.to(torch.int32) * tidx).amax(0)


def settling(step_logs: Dict) -> Dict:
    tok = step_logs["tok"].to(torch.int32)  # [T=255, N, 128]; index 254 = final denoise
    T = tok.size(0)
    n_steps = T - 1
    calls_per_seg = [SEG_LEN] * (N_SEG - 1) + [n_steps - SEG_LEN * (N_SEG - 1) + 1]
    uniform_share = [c / T for c in calls_per_seg]

    last = _last_change(tok).numpy()
    seg = _segment_of_step(last)
    hist_step = np.bincount(last.reshape(-1), minlength=T)
    hist_seg = np.bincount(seg.reshape(-1), minlength=N_SEG) / last.size

    # provisional settling: first t with no change in (t, t+W]; revision = changes later
    ch = (tok[1:] != tok[:-1])  # ch[t-1] = change at step t
    csum = torch.cat([torch.zeros(1, *ch.shape[1:], dtype=torch.int32), ch.to(torch.int32).cumsum(0)], 0)
    W = SETTLE_WINDOW
    # window_changes[t] = #changes at steps t+1..t+W, for t in 0..T-1-W
    win = csum[W:] - csum[:-W]  # shape [T-W, ...], index t -> changes in (t, t+W]
    quiet = win == 0
    has = quiet.any(0)
    first = torch.where(has, quiet.to(torch.int8).argmax(0), torch.full_like(has, T - 1, dtype=torch.int64))
    first = first.numpy()
    revised = last > first
    val_at_first = torch.gather(tok, 0, torch.from_numpy(first).unsqueeze(0).long()).squeeze(0).numpy()
    final = tok[-1].numpy()
    changed_value = val_at_first != final
    fseg = _segment_of_step(first)
    per_seg = []
    for s in range(N_SEG):
        m = fseg == s
        per_seg.append(dict(
            segment=s,
            frac_tokens_provisionally_settled_here=float(m.mean()),
            frac_revised_after=float(revised[m].mean()) if m.any() else float("nan"),
            frac_final_value_differs=float(changed_value[m].mean()) if m.any() else float("nan"),
        ))

    # changes per step (all tokens)
    change_rate = ch.float().mean(dim=(1, 2)).numpy()

    # bit-position settling (MSB first)
    bit_rows = []
    for k in range(BITS):
        b = (tok >> (BITS - 1 - k)) & 1
        lb = _last_change(b).numpy()
        bit_rows.append(dict(
            bit=k,
            mean_settle_step=float(lb.mean()),
            median_settle_step=float(np.median(lb)),
            by_segment=[float(v) for v in np.bincount(_segment_of_step(lb).reshape(-1), minlength=N_SEG) / lb.size],
            frac_never_changed=float((lb == 0).mean()),
        ))

    bit_ent = step_logs["bit_ent"].float().mean(dim=(1, 2)).numpy()
    sig_hat = step_logs["sigma"].numpy()
    gam = step_logs["gamma"].numpy()
    sig_cur = sig_hat / (1 + gam)
    delta = np.sqrt(np.clip(sig_hat ** 2 - sig_cur ** 2, 0, None))
    eps_rms = step_logs["eps_rms"].numpy().mean(1)
    return dict(
        n_model_calls=int(T), n_euler_steps=int(n_steps), calls_per_segment=calls_per_seg,
        uniform_share_per_segment=uniform_share,
        settle_share_per_segment=[float(v) for v in hist_seg],
        settle_hist_by_step=[int(v) for v in hist_step],
        mean_settle_step=float(last.mean()), median_settle_step=float(np.median(last)),
        frac_never_changed_after_step0=float((last == 0).mean()),
        provisional_window=W, provisional_by_segment=per_seg,
        overall_frac_revised_after_provisional=float(revised.mean()),
        change_rate_by_step=[float(v) for v in change_rate],
        bit_position=bit_rows,
        mean_bit_entropy_by_step=[float(v) for v in bit_ent],
        sigma_hat_by_step=[float(v) for v in sig_hat],
        gamma_by_step=[float(v) for v in gam],
        churn_sigma_delta_by_step=[float(v) for v in delta],
        churn_noise_rms_by_step=[float(v) for v in (delta * eps_rms)],
    )


# -----------------------------------------------------------------------------
# §8 decision rule (mechanical; operationalisations in NOTEBOOK.md)
# -----------------------------------------------------------------------------
def decide(rows: Dict[str, Dict], settle_share: Optional[List[float]]) -> Dict:
    def usable_material(r):
        return r is not None and r["material"] and r["d_nll_tok"] > 0 and not r["entropy_flag"]

    ref_mat = [s for s in range(N_SEG) if usable_material(rows.get(f"refine_{s}"))]
    # refine helping means ΔNLL < 0 (better); "material" for refine is an improvement
    ref_help = [s for s in range(N_SEG)
                if rows.get(f"refine_{s}") and rows[f"refine_{s}"]["material"]
                and rows[f"refine_{s}"]["d_nll_tok"] < 0 and not rows[f"refine_{s}"]["entropy_flag"]]
    nc = {s: rows.get(f"nochurn_{s}") for s in range(N_SEG)}
    nc_vals = [nc[s]["d_nll_tok"] for s in range(N_SEG) if nc[s] is not None]
    med = float(np.median(nc_vals)) if nc_vals else float("nan")
    nc_standout = [s for s in range(N_SEG) if usable_material(nc[s]) and nc[s]["d_nll_tok"] - med >= MATERIAL]
    both = sorted(set(ref_help) & set(nc_standout))
    ref_any = sorted(set(ref_help) | set(ref_mat))

    if both:
        row, segs = "both", both
    elif ref_help:
        row, segs = "step_size", ref_help
    elif nc_standout and not ref_any:
        row, segs = "uncorrected_errors", nc_standout
    elif not ref_any and not nc_standout:
        row, segs = "model", []
    else:
        row, segs = "unclassified", sorted(set(ref_any) | set(nc_standout))

    mismatch = None
    if settle_share is not None and segs:
        top2 = list(np.argsort(settle_share)[::-1][:2])
        mismatch = dict(top2_settling_segments=[int(v) for v in top2],
                        bottleneck_outside_top2=[s for s in segs if s not in top2])
    return dict(row=row, segments=segs, refine_material_improve=ref_help, refine_material_worse=ref_mat,
                nochurn_median_d_nll=med, nochurn_standout=nc_standout, settling_mismatch=mismatch)


# -----------------------------------------------------------------------------
def main(out_root: Path, results_root: Path, batches: Optional[List[int]] = None) -> None:
    adir = results_root / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    if batches is None:
        batches = sorted(int(p.name[5:]) for p in results_root.glob("batch*") if p.is_dir())

    # ---- Run 1: anchor + settling
    ref0 = _scores(results_root, 0, "reference")
    out = {}
    if ref0 is not None:
        out["anchor"] = dict(genppl=ref0["genppl"], mean_entropy=ref0["mean_entropy"], source=ref0["path"],
                             readme_single_seed_range=[59.0, 60.0], readme_entropy=4.31,
                             gate_genppl=bool(58.5 <= ref0["genppl"] <= 60.5),
                             gate_entropy=bool(abs(ref0["mean_entropy"] - 4.31) <= 0.015))
    logs_p = out_root / "batch0" / "step_logs.pt"
    settle = None
    if logs_p.exists():
        settle = settling(torch.load(logs_p, map_location="cpu"))
        (adir / "run1_settling.json").write_text(json.dumps(settle, indent=1))
        out["settling_source"] = str(logs_p)

    # ---- Run 2: per-branch paired stats
    names = ["null"] + [f"{k}_{s}" for k in ("refine", "nochurn") for s in range(N_SEG)]
    ref_art0 = _artefact(out_root, 0, "reference")
    freq_rank = _freq_rank(ref_art0["tokens"].numpy().astype(np.int64)) if ref_art0 is not None else None
    null_art0 = _artefact(out_root, 0, "null")
    null_floor0 = token_breakdown(ref_art0, null_art0, freq_rank) if null_art0 is not None else None

    rows, rows_by_batch, breakdowns = {}, {}, {}
    for name in names:
        per_batch = {}
        pooled_ref, pooled_br = None, None
        for b in batches:
            ref = _scores(results_root, b, "reference")
            br = _scores(results_root, b, name)
            if ref is None or br is None:
                continue
            st = paired_stats(ref, br)
            ra, ba = _artefact(out_root, b, "reference"), _artefact(out_root, b, name)
            tb = token_breakdown(ra, ba, freq_rank)
            na = _artefact(out_root, b, "null")
            nf = token_breakdown(ra, na, freq_rank)["frac_changed"] if na is not None else (
                null_floor0["frac_changed"] if null_floor0 else float("nan"))
            st["frac_tokens_changed"] = tb["frac_changed"]
            st["null_frac_tokens_changed"] = nf
            st["frac_tokens_changed_net"] = tb["frac_changed"] - nf
            st["nfe_per_sample"] = br["meta"].get("nfe_per_sample")
            st["source"] = br["path"]
            per_batch[b] = st
            if b == 0:
                breakdowns[name] = tb
            pooled_ref = ref if pooled_ref is None else _concat(pooled_ref, ref)
            pooled_br = br if pooled_br is None else _concat(pooled_br, br)
        if not per_batch:
            continue
        rows_by_batch[name] = per_batch
        if len(per_batch) > 1:
            pooled = paired_stats(pooled_ref, pooled_br)
            pooled["batches"] = sorted(per_batch)
            for k in ("frac_tokens_changed", "frac_tokens_changed_net"):
                pooled[k] = float(np.mean([per_batch[b][k] for b in per_batch]))
            pooled["nfe_per_sample"] = per_batch[min(per_batch)]["nfe_per_sample"]
            rows[name] = pooled
        else:
            rows[name] = dict(per_batch[min(per_batch)], batches=sorted(per_batch))

    # net-of-null token breakdowns
    if null_floor0 is not None:
        for name, tb in breakdowns.items():
            tb["net_by_position_bin16"] = [a - b for a, b in zip(tb["by_position_bin16"], null_floor0["by_position_bin16"])]
            tb["net_bitflip_by_position"] = [a - b for a, b in zip(tb["bitflip_by_position_msb_first"],
                                                                     null_floor0["bitflip_by_position_msb_first"])]

    out["null"] = rows.get("null")
    out["rows"] = rows
    out["rows_by_batch"] = {k: {str(b): v for b, v in d.items()} for k, d in rows_by_batch.items()}
    out["token_breakdowns_batch0"] = breakdowns
    if any(n.startswith("refine_") or n.startswith("nochurn_") for n in rows):
        out["decision"] = decide(rows, settle["settle_share_per_segment"] if settle else None)
    out["definitions"] = __doc__
    (adir / "run2_analysis.json").write_text(json.dumps(out, indent=1))

    with (adir / "run2_table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["branch", "batches", "nfe_per_sample", "d_nll_tok", "ci_lo", "ci_hi", "material",
                    "borderline", "d_entropy", "entropy_flag", "frac_tokens_changed", "frac_tokens_changed_net",
                    "genppl_ref", "genppl_branch", "top10_share_signed", "top10_share_abs"])
        for name in names:
            r = rows.get(name)
            if r is None:
                continue
            w.writerow([name, "+".join(map(str, r["batches"])), r.get("nfe_per_sample"),
                        f"{r['d_nll_tok']:.5f}", f"{r['ci_lo']:.5f}", f"{r['ci_hi']:.5f}", r["material"],
                        r["borderline"], f"{r['d_entropy']:.5f}", r["entropy_flag"],
                        f"{r['frac_tokens_changed']:.5f}", f"{r['frac_tokens_changed_net']:.5f}",
                        f"{r['genppl_ref']:.3f}", f"{r['genppl_branch']:.3f}",
                        f"{r['top10_share_signed']:.3f}", f"{r['top10_share_abs']:.3f}"])
    print(f"[analyze] wrote {adir}")
    if "anchor" in out:
        print("[analyze] anchor:", json.dumps(out["anchor"]))
    if "decision" in out:
        print("[analyze] decision:", json.dumps(out["decision"]))

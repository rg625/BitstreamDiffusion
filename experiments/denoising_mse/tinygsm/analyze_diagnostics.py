#!/usr/bin/env python
"""
Analysis of the GSM8K diagnostics (diagnostics.py; raw runs in results_tinygsm/diagnostics/raw/).

  Maj@N / pass@N   base runs, seeds 0 .. N-1 (and the mean over disjoint seed groups); the vote is over executed
                   answers (ints, floats rounded to 1e-3 as the scorer's tolerance), ties -> the earliest seed's answer
  variants         accuracy per variant (mean over its seeds), paired delta vs base (same seeds, same noise) with a
                   95% bootstrap CI over problems, failure-status breakdown
  clamping         when tokens get clamped vs their relative position in the answer
  failures         taxonomy of wrong answers (base runs): no runnable code (exec error / no function / timeout /
                   no answer), invalid token ids (>= vocab: a bit-level artifact), runs but wrong; for the latter,
                   copy errors: numbers in the code that are one digit edit away from a number in the question, with
                   the token-id Hamming distance between the generated and the question's number tokens; plus the
                   ASCII Hamming distance between predicted and gold answer strings (as requested; see the caveat)
Outputs: results_tinygsm/diagnostics/{summary.json, variants.csv, majn.csv, fig_*.png}
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results_tinygsm/diagnostics"
RAW = RES / "raw"
RNG = np.random.default_rng(0)
NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")
NSTATUS = ("correct", "wrong", "exec_error", "no_function", "no_answer", "timeout")


def style(ax):
    ax.grid(True, color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8984")
    ax.tick_params(colors="#52514e", labelsize=8)


def load():
    runs = {}
    for f in sorted(RAW.glob("*_seed*.json")):
        if re.search(r"_seed\d+$", f.stem) is None:
            continue
        r = json.loads(f.read_text())
        runs.setdefault(r["model"], {}).setdefault(r["variant"], {})[r["seed"]] = r
    return runs


def vkey(a):
    if a is None:
        return None
    if isinstance(a, float):
        if abs(a - round(a)) < 1e-9 and abs(a) < 1e15:
            return int(round(a))
        return round(a, 3)
    return int(a)


def gold_answers():
    import common as C
    import sandbox_gsm8k as SB

    recs = C.GSM8KTest().records
    return [SB._extract_gold_answer(r["response_ground_truth"]) for r in recs], recs


def maj_at_n(base, seeds, gold):
    import sandbox_gsm8k as SB

    n = len(gold)
    maj, anyc = np.zeros(n), np.zeros(n)
    for i in range(n):
        votes = [vkey(base[s]["answer"][i]) for s in seeds]
        c = Counter(v for v in votes if v is not None)
        if c:
            top = max(c.values())
            win = next(v for v in votes if v is not None and c[v] == top)
            maj[i] = SB._numbers_equal(win, gold[i])
        anyc[i] = max(base[s]["correct"][i] for s in seeds)
    return maj, anyc


def majn_part(runs, gold, summary):
    rows, out = [], {}
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for m, col in (("S", "#2a78d6"), ("M", "#eb6834")):
        base = runs.get(m, {}).get("base", {})
        seeds = sorted(base)
        if not seeds:
            continue
        res = {}
        Ns = [N for N in (1, 2, 4, 8, 16, 32) if N <= len(seeds)]
        for N in Ns:
            groups = [seeds[g * N:(g + 1) * N] for g in range(len(seeds) // N)]
            mj = [maj_at_n(base, g, gold) for g in groups]
            accm = [float(a.mean()) for a, _ in mj]
            accp = [float(p.mean()) for _, p in mj]
            res[N] = dict(maj_first=accm[0], pass_first=accp[0], maj_mean=float(np.mean(accm)),
                          maj_sd=float(np.std(accm)), pass_mean=float(np.mean(accp)), n_groups=len(groups))
            rows.append([m, N, len(groups), f"{100 * accm[0]:.2f}", f"{100 * np.mean(accm):.2f}",
                         f"{100 * np.std(accm):.2f}", f"{100 * accp[0]:.2f}", f"{100 * np.mean(accp):.2f}"])
        out[m] = dict(seeds=len(seeds), single=[float(np.mean(base[s]["correct"])) for s in seeds], by_N=res)
        ax.plot(Ns, [100 * res[N]["maj_mean"] for N in Ns], "o-", color=col, label=f"CoBit-{m} Maj@N")
        ax.plot(Ns, [100 * res[N]["pass_mean"] for N in Ns], "s--", color=col, alpha=0.6, label=f"CoBit-{m} pass@N")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N samples")
    ax.set_ylabel("GSM8K accuracy [%]")
    ax.set_title("Majority vote and oracle (pass@N) over N sampled trajectories", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(RES / "fig_majn.png", dpi=160)
    plt.close(fig)
    with open(RES / "majn.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "N", "groups", "maj_seeds0..N-1", "maj_mean_groups", "maj_sd_groups", "pass_seeds0..N-1",
                    "pass_mean_groups"])
        w.writerows(rows)
    summary["majn"] = out


def variants_part(runs, summary):
    out, rows = {}, []
    for m in sorted(runs):
        base = runs[m].get("base", {})
        out[m] = {}
        for v, by in sorted(runs[m].items()):
            seeds = sorted(set(by) & set(base)) if v != "base" else sorted(set(base) & {0, 1})
            if not seeds:
                continue
            pv = np.mean([by[s]["correct"] for s in seeds], 0)
            pb = np.mean([base[s]["correct"] for s in seeds], 0)
            d = pv - pb
            bs = d[RNG.integers(0, len(d), (4000, len(d)))].mean(1)
            st = Counter(x for s in seeds for x in by[s]["status"])
            tot = sum(st.values())
            e = dict(seeds=seeds, accuracy=float(pv.mean()), base=float(pb.mean()), delta=float(d.mean()),
                     lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                     status={k: st.get(k, 0) / tot for k in NSTATUS},
                     invalid_rate=float(np.mean([x > 0 for s in seeds for x in by[s]["invalid"]])))
            out[m][v] = e
            rows.append([m, v, " ".join(map(str, seeds)), f"{100 * e['accuracy']:.2f}", f"{100 * e['delta']:+.2f}",
                         f"{100 * e['lo']:+.2f}", f"{100 * e['hi']:+.2f}"] +
                        [f"{100 * e['status'][k]:.1f}" for k in NSTATUS] + [f"{100 * e['invalid_rate']:.1f}"])
    with open(RES / "variants.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "variant", "seeds", "acc", "delta_vs_base", "lo", "hi"] + [f"%{k}" for k in NSTATUS]
                   + ["%with_invalid_ids"])
        w.writerows(rows)
    summary["variants"] = out


def clamp_part(runs, summary):
    import common as C

    tok = C.get_tokenizer()
    out = {}
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, m in zip(axs, ("S", "M")):
        base = runs.get(m, {}).get("base", {})
        for v, col in (("clampS0.99", "#9a9994"), ("clampP0.99", "#1b1b1a"), ("clampS0.99s1.0", "#2a78d6"),
                       ("clampP0.99s1.0", "#eb6834")):
            by = runs.get(m, {}).get(v, {})
            if not by or 0 not in by or 0 not in base:
                continue
            r = by[0]
            cs = np.array(r["clamp_step"])
            pl = np.array(r["plen"])
            # answer length (to EOS) from the base run's decoded text, same seed
            alen = np.array([len(tok(t, add_special_tokens=False).input_ids) + 1 for t in base[0]["text"]])
            rel = np.arange(cs.shape[1])[None] - pl[:, None]
            inans = (rel >= 0) & (rel < alen[:, None])
            frac = (rel / np.maximum(alen[:, None], 1))
            bins = np.linspace(0, 1, 11)
            mean_step, clamped = [], []
            for a, b in zip(bins[:-1], bins[1:]):
                sel = inans & (frac >= a) & (frac < b)
                c = cs[sel]
                clamped.append(float((c >= 0).mean()))
                mean_step.append(float(np.where(c >= 0, c, 255).mean()))
            out.setdefault(m, {})[v] = dict(frac_answer_clamped=float((cs[inans] >= 0).mean()),
                                            frac_clamped_step0=float((cs[inans] == 0).mean()),
                                            mean_commit_step_by_relpos=mean_step, clamped_by_relpos=clamped)
            ax.plot(0.5 * (bins[1:] + bins[:-1]), mean_step, "o-", color=col, ms=3, label=v)
        ax.set_xlabel("relative position in the answer (base run's length)")
        ax.set_title(f"CoBit-{m}: mean step at which a token is committed (255 = never clamped)", loc="left",
                     fontsize=9)
        style(ax)
    axs[0].set_ylabel("step (0 = pure noise)")
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "fig_clamp_commit.png", dpi=160)
    plt.close(fig)
    summary["clamping"] = out


def edit1(a: str, b: str) -> bool:
    if a == b:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if abs(len(a) - len(b)) == 1:
        s, t = (a, b) if len(a) < len(b) else (b, a)
        return any(t[:k] + t[k + 1:] == s for k in range(len(t)))
    return False


CONST = re.compile(r"^\s*[A-Za-z_]\w*\s*=\s*(-?\d+(?:\.\d+)?)\s*(?:#.*)?$")


def _fval(x):
    try:
        return round(float(x), 6)
    except ValueError:
        return None


def exec_kind(text):
    """Why a generated solution does not give a number: exception class name, or the type of the returned value."""
    import sandbox_gsm8k as SB

    try:
        code = SB._extract_code(text)
        with SB._time_limit(5.0):
            ns = SB._safe_exec_no_timer(code)
            fn = ns.get("simple_math_problem", None)
            if fn is None:
                return "no_function"
            with contextlib.redirect_stdout(io.StringIO()):
                out = fn()
    except SB._Timeout:
        return "timeout"
    except Exception as e:  # noqa: BLE001
        return f"raise:{type(e).__name__}"
    return f"returns:{type(out).__name__}"


def failures_part(runs, gold, recs, summary):
    """Wrong answers of the base runs. Copy errors: constants assigned directly in the code (x = 45) whose value is
    not in the question, have >= 2 digits and are one digit edit from a number in the question. Digits are single
    tokens with ids 32..41, so a single-digit substitution has a small token-id Hamming distance by construction:
    the observed distribution is compared with that of a uniformly random digit substitution."""
    import itertools

    import common as C

    tok = C.get_tokenizer()
    dig = {str(d): tok(str(d), add_special_tokens=False).input_ids[0] for d in range(10)}
    null = np.bincount([bin(dig[a] ^ dig[b]).count("1") for a, b in itertools.permutations(dig, 2)], minlength=17)
    out = dict(null_digit_substitution_hamming=(null / null.sum()).tolist())
    for m in sorted(runs):
        base = runs[m].get("base", {})
        if not base:
            continue
        cnt, kinds = Counter(), Counter()
        ham, ham_ascii = [], []
        n_wrong = n_const = 0
        for s, r in base.items():
            for i, (stt, txt) in enumerate(zip(r["status"], r["text"])):
                if stt == "correct":
                    continue
                n_wrong += 1
                cnt[stt] += 1
                if r["invalid"][i] > 0:
                    cnt["has_invalid_ids"] += 1
                if stt != "wrong":
                    kinds[exec_kind(txt)] += 1
                    continue
                qstr = set(NUM.findall(recs[i]["prompt"].replace(",", "")))
                qval = {_fval(q) for q in qstr}
                body = txt.split("\'\'\'")[-1] if txt.count("\'\'\'") >= 2 else txt
                consts = [mm.group(1) for line in body.splitlines() if (mm := CONST.match(line))]
                n_const += len(consts)
                foreign = [c for c in consts if _fval(c) not in qval]
                near = [(c, q) for c in foreign for q in qstr if len(c.replace(".", "")) >= 2 and edit1(c, q)]
                if foreign:
                    cnt["wrong_with_foreign_constant"] += 1
                if near:
                    cnt["wrong_with_near_copy_error"] += 1
                    for c, q in near[:1]:
                        if len(c) == len(q):
                            k = next(j for j in range(len(c)) if c[j] != q[j])
                            if c[k] in dig and q[k] in dig:
                                ham.append(bin(dig[c[k]] ^ dig[q[k]]).count("1"))
                        else:
                            cnt["near_copy_digit_dropped_or_inserted"] += 1
                p, g = r["answer"][i], gold[i]
                if p is not None and g is not None:
                    ps, gs = str(vkey(p)), str(vkey(g))
                    ham_ascii.append(sum(bin(ord(u) ^ ord(v)).count("1") for u, v in zip(ps, gs)) if len(ps) == len(gs) else -1)
                    try:
                        if g and abs(float(p) - float(g)) <= 0.01 * abs(float(g)):
                            cnt["wrong_within_1pct"] += 1
                    except OverflowError:  # a huge integer answer
                        pass
        h, ha = np.array(ham, int), np.array(ham_ascii, int)
        obs = np.bincount(h, minlength=17)
        out[m] = dict(
            runs=len(base), n_wrong=n_wrong, counts=dict(cnt), shares={k: v / n_wrong for k, v in cnt.items()},
            constants_in_wrong_code=n_const,
            not_runnable_kinds={k: v / n_wrong for k, v in kinds.most_common(12)},
            near_copy_substitution_hamming=dict(n=int(len(h)), observed=(obs / max(1, obs.sum())).tolist(),
                                                le1_observed=float((h <= 1).mean()) if len(h) else None,
                                                le1_chance=float(null[:2].sum() / null.sum()),
                                                le2_observed=float((h <= 2).mean()) if len(h) else None,
                                                le2_chance=float(null[:3].sum() / null.sum())),
            ascii_hamming=dict(n=int(len(ha)), le2_same_length=float(((ha >= 0) & (ha <= 2)).mean()) if len(ha) else None,
                               diff_length=float((ha < 0).mean()) if len(ha) else None))
    summary["failures"] = out


ORDER = ["screset4.0", "screset2.5", "screset1.3", "scdecay0.5", "scdecay0.8", "scdecay0.9", "clampS0.99",
         "clampP0.99", "clampS0.99s1.0", "clampP0.99s1.0", "stagger0.0", "stagger0.05", "stagger0.1", "stagger0.2"]
LABEL = {"screset4.0": "SC reset σ≤4.0", "screset2.5": "SC reset σ≤2.5", "screset1.3": "SC reset σ≤1.3",
         "scdecay0.5": "SC decay α=0.5", "scdecay0.8": "SC decay α=0.8", "scdecay0.9": "SC decay α=0.9",
         "clampS0.99": "clamp scattered τ=0.99", "clampP0.99": "clamp prefix τ=0.99",
         "clampS0.99s1.0": "clamp scattered, σ≤1", "clampP0.99s1.0": "clamp prefix, σ≤1",
         "stagger0.0": "stagger Δt=0 (control)", "stagger0.05": "stagger Δt=0.05", "stagger0.1": "stagger Δt=0.10",
         "stagger0.2": "stagger Δt=0.20"}


def variant_figures(summary):
    V = summary["variants"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, (xl, clip) in zip(axs, (("full range", None), ("zoom ±5 pp", 5))):
        for j, (m, col) in enumerate((("S", "#2a78d6"), ("M", "#eb6834"))):
            ys = np.arange(len(ORDER)) + (0.18 if m == "S" else -0.18)
            for y, v in zip(ys, ORDER):
                e = V.get(m, {}).get(v)
                if e is None:
                    continue
                d, lo, hi = 100 * e["delta"], 100 * e["lo"], 100 * e["hi"]
                ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", color=col, ms=4, capsize=2,
                            label=f"CoBit-{m}" if v == ORDER[0] or (v == ORDER[-4] and j == 1 and False) else None)
        ax.axvline(0, color="#8a8984", lw=0.8)
        if clip:
            ax.set_xlim(-clip, clip)
        ax.set_xlabel("Δ GSM8K accuracy vs base, same seeds [pp] (95% CI)")
        ax.set_title(xl, loc="left", fontsize=9)
        style(ax)
    axs[0].set_yticks(range(len(ORDER)))
    axs[0].set_yticklabels([LABEL[v] for v in ORDER], fontsize=8)
    axs[0].invert_yaxis()
    h = [plt.Line2D([], [], color=c, marker="o", ls="", label=f"CoBit-{m}") for m, c in (("S", "#2a78d6"), ("M", "#eb6834"))]
    axs[1].legend(handles=h, frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RES / "fig_variants.png", dpi=160)
    plt.close(fig)

    F = summary["failures"]
    cats = [("raise:NameError", "undefined variable (NameError)"), ("returns:NoneType", "no return value"),
            ("raise:SyntaxError", "syntax error"), ("raise:UnboundLocalError", "used before assignment"),
            ("other_unrunnable", "other non-runnable"), ("wrong_near", "runs, near-copy of a question number"),
            ("wrong_other", "runs, other wrong answer")]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for j, (m, col) in enumerate((("S", "#2a78d6"), ("M", "#eb6834"))):
        f = F.get(m)
        if not f:
            continue
        k = f["not_runnable_kinds"]
        named = sum(k.get(c, 0) for c, _ in cats[:4])
        unrun = 1 - f["shares"].get("wrong", 0)
        vals = [k.get(c, 0) for c, _ in cats[:4]] + [max(0.0, unrun - named),
                f["shares"].get("wrong_with_near_copy_error", 0),
                f["shares"].get("wrong", 0) - f["shares"].get("wrong_with_near_copy_error", 0)]
        y = np.arange(len(cats)) + (0.2 if m == "S" else -0.2)
        ax.barh(y, [100 * v for v in vals], height=0.38, color=col, label=f"CoBit-{m} ({f['runs']} runs)")
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([c[1] for c in cats], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("share of wrong answers [%]")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(RES / "fig_failures.png", dpi=160)
    plt.close(fig)


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})
    runs = load()
    gold, recs = gold_answers()
    summary = {}
    majn_part(runs, gold, summary)
    variants_part(runs, summary)
    clamp_part(runs, summary)
    failures_part(runs, gold, recs, summary)
    variant_figures(summary)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    for m, e in summary["majn"].items():
        print("MajN", m, {N: (round(100 * v["maj_mean"], 1), round(100 * v["pass_mean"], 1)) for N, v in e["by_N"].items()})
    for m, d in summary["variants"].items():
        for v, e in d.items():
            print(f"{m} {v:16s} acc {100 * e['accuracy']:5.1f}  Δ {100 * e['delta']:+5.1f} [{100 * e['lo']:+.1f},{100 * e['hi']:+.1f}]"
                  f"  exec_err {100 * e['status']['exec_error']:.0f}%  no_ans {100 * e['status']['no_answer']:.0f}%")
    print(json.dumps(summary["failures"], indent=1))


if __name__ == "__main__":
    main()

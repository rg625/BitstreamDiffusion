"""TASK 3: failure taxonomy of generated GSM8K programs (non-FKC cells only).

What is on disk limits what can be classified:

  * every problem (1319 per cell) has `correct` and the executed `answer`
    (None = the program did not run, or returned nothing usable). That gives
    the coarse split correct / wrong answer / non-executing for everything;
  * program TEXT exists only for the first 100 problems of each cell, cut at
    400 characters by gsm8k_eval. A record shorter than 400 chars is the whole
    completion and can be re-executed with the TASK 0 executor to get the fine
    status (syntax / name / runtime / timeout-memory / no return). A record of
    exactly 400 chars was cut by LOGGING; its failure, if any, is unknowable
    from the text and is reported as such, never guessed.

Truncation (model-side, complete records only):
  * incomplete_tail : the code does not compile and the error is at the last
                      line or is an end-of-input error, i.e. the program stops
                      mid-statement (ran into padding/EOS);
  * missing_return  : simple_math_problem parses but has no return statement;
  * canvas_end      : the completion fills the whole 512-token canvas.

Lengths: characters (censored at 400) for the 100 records, by recorded class.
"""
from __future__ import annotations

import ast
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.cpu_reanalysis.common import (  # noqa: E402
    REPO, cpu_guard, gold_answers, load_dir, per_problem, write_json,
)

cpu_guard()
os.environ.setdefault("HF_HOME", str(REPO / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from evaluation.tasks.safe_exec import execute_many  # noqa: E402
from evaluation.tasks.sandbox_gsm8k import _extract_code  # noqa: E402

CELLS = "results/tokens/eval_gh200/gsm8k_results_*.json"
LOG_CAP = 400
EOF_MSGS = ("unexpected eof", "was never closed", "expected an indented block",
            "incomplete input", "unterminated", "eol while scanning", "eof while scanning")


def arm_of(run):
    if run.startswith("tok_"):
        return "token"
    if run == "prod_anchor":
        return "production"
    return "bits_control"


def truncation_flags(text, n_tokens, capacity):
    code = _extract_code(text)
    flags = {"incomplete_tail": False, "missing_return": False,
             "canvas_end": bool(n_tokens >= capacity - 1)}
    raw = text[text.find("def "):] if "def " in text else text
    try:
        compile(raw.strip(), "<s>", "exec")
    except SyntaxError as e:
        last = len(raw.strip().splitlines())
        msg = (e.msg or "").lower()
        flags["incomplete_tail"] = bool((e.lineno or 0) >= last or any(m in msg for m in EOF_MSGS))
    except ValueError:
        pass
    try:
        tree = ast.parse(code)
        fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "simple_math_problem"]
        if fns:
            flags["missing_return"] = not any(isinstance(n, ast.Return) and n.value is not None
                                              for n in ast.walk(fns[0]))
    except (SyntaxError, ValueError):
        pass
    flags["truncated"] = flags["incomplete_tail"] or flags["canvas_end"] or flags["missing_return"]
    return flags


def quantiles(xs):
    if not xs:
        return None
    a = np.asarray(xs, float)
    cens = float((a >= LOG_CAP).mean())
    q = {f"q{int(p * 100)}": (float(np.quantile(a, p)) if np.quantile(a, p) < LOG_CAP else ">=400")
         for p in (0.1, 0.25, 0.5, 0.75, 0.9)}
    return {"n": len(xs), "frac_at_400_cap": cens, **q}


def main():
    from data.tinygsm import SEP, _special_ids, get_tokenizer
    tok = get_tokenizer("HuggingFaceTB/SmolLM-135M")
    bos, _, _ = _special_ids(tok)
    sep = tok(SEP, add_special_tokens=False).input_ids
    gold, recs = gold_answers()
    capacity = {i: 512 - (1 + len(tok(r["prompt"].strip(), add_special_tokens=False).input_ids) + len(sep))
                for i, r in enumerate(recs)}

    cells = load_dir(CELLS)
    per_cell, agg = [], defaultdict(lambda: defaultdict(Counter))
    lengths = defaultdict(lambda: defaultdict(list))
    trunc_agg = defaultdict(lambda: defaultdict(Counter))
    for d in cells:
        run, step, seed = d["_run"], d["_step"], d["_seed"]
        key = f"{arm_of(run)}|{run}|{step}"
        pp = per_problem(d)
        coarse = Counter("correct" if c else ("wrong_answer" if a is not None else "non_executing")
                         for c, a in pp.values())
        rs = d.get("sample_records") or []
        texts = [r["response"] for r in rs]
        res = execute_many(texts, float(d.get("timeout_s", 5.0)), max_workers=4)
        fine = Counter()
        for r, t, ex in zip(rs, texts, res):
            i = int(r["idx"])
            c, a = pp[i]
            cls = "correct" if c else ("wrong_answer" if a is not None else "non_executing")
            lengths[key][cls].append(len(t))
            if cls != "non_executing":
                fine[cls] += 1
                continue
            if len(t) >= LOG_CAP:
                fine["unknown_cut_by_logging"] += 1
                continue
            st = {"memory": "timeout_memory", "timeout": "timeout_memory"}.get(ex.status, ex.status)
            if st == "ok":                     # should not happen: recorded None, executes now
                st = "MISMATCH_executes_now"
            fine[st] += 1
            ntok = len(tok(t, add_special_tokens=False).input_ids)
            fl = truncation_flags(t, ntok, capacity[i])
            tc = trunc_agg[key]
            tc["non_exec_complete"]["n"] += 1
            for k, v in fl.items():
                tc["non_exec_complete"][k] += int(v)
            tc["by_status"][f"{st}|truncated={fl['truncated']}"] += 1
        for k, v in coarse.items():
            agg[key]["coarse"][k] += v
        for k, v in fine.items():
            agg[key]["fine_100"][k] += v
        per_cell.append({"file": d["_path"], "run": run, "step": step, "seed": seed,
                         "coarse": dict(coarse), "fine_first100": dict(fine)})
        print(f"{run} {step} s{seed}: {dict(coarse)} | {dict(fine)}", flush=True)

    summary = {}
    for key in sorted(agg):
        arm, run, step = key.split("|")
        co = agg[key]["coarse"]; n = sum(co.values())
        tc = trunc_agg[key]["non_exec_complete"]
        summary[key] = {
            "arm": arm, "run": run, "step": int(step), "n_problem_seed": n,
            "coarse_frac": {k: co[k] / n for k in ("correct", "wrong_answer", "non_executing")},
            "fine_first100_counts": dict(agg[key]["fine_100"]),
            "non_exec_complete_records": tc["n"],
            "truncated_frac_of_complete_non_exec": (tc["truncated"] / tc["n"]) if tc["n"] else None,
            "truncation_flags": {k: tc[k] for k in ("incomplete_tail", "missing_return", "canvas_end", "truncated")},
            "status_x_truncation": dict(trunc_agg[key]["by_status"]),
            "length_chars_by_class": {c: quantiles(v) for c, v in lengths[key].items()},
        }
    out = {"scope": CELLS, "excluded": "FKC (none in scope); guidance confirmation cells: per-problem/text files "
           "not on this cluster (CSD3 runs/guidance/)", "log_cap_chars": LOG_CAP,
           "summary": summary, "cells": per_cell}
    print("->", write_json("t3_taxonomy.json", out))
    for key, s in summary.items():
        print(key, {k: round(v, 3) for k, v in s["coarse_frac"].items()}, s["fine_first100_counts"],
              "trunc", s["truncation_flags"], "of", s["non_exec_complete_records"])


if __name__ == "__main__":
    main()

"""TASK 0 verification: the subprocess executor scores exactly like the old sandbox.

Full program text was never saved (gsm8k_eval keeps only the first 100 records,
cut at 400 chars), so a saved evaluation cannot be re-executed end to end.
What can be checked:

  (a) the gold comparison: re-score the stored per-problem answers through
      `_numbers_equal` and reproduce `num_correct` / `accuracy` exactly;
  (b) the executor on complete programs: records whose text is < 400 chars are
      the full completion, so re-executing them must reproduce the recorded
      answer and `correct` flag;
  (c) old vs new executor on every stored text, complete or not.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.cpu_reanalysis.common import (  # noqa: E402
    cpu_guard, gold_answers, load_dir, parse_answer, per_problem, write_json,
)

cpu_guard()

from evaluation.tasks.safe_exec import execute_many  # noqa: E402
from evaluation.tasks.sandbox_gsm8k import _numbers_equal, predict_answer  # noqa: E402

PRIMARY = "results/tokens/eval_gh200/gsm8k_results_stochastic_g0.0_w0.0_s256_sd0.3998_ema1_prod_anchor__ck000500000__s0.json"


def check(d, gold):
    pp = per_problem(d)
    n_ok = sum(_numbers_equal(parse_answer(a), gold[i]) for i, (_, a) in pp.items())
    flags_match = sum(int(_numbers_equal(parse_answer(a), gold[i])) == c for i, (c, a) in pp.items())
    recs = d.get("sample_records") or []
    texts = [r["response"] for r in recs]
    new = execute_many(texts, float(d.get("timeout_s", 5.0)), max_workers=4)
    old = [predict_answer(t, float(d.get("timeout_s", 5.0))) for t in texts]
    complete = [k for k, t in enumerate(texts) if len(t) < 400]
    comp_ans = sum(str(new[k].value) == str(pp[int(recs[k]["idx"])][1]) for k in complete)
    comp_cor = sum(bool(_numbers_equal(new[k].value, gold[int(recs[k]["idx"])])) == bool(recs[k]["correct"])
                   for k in complete)
    mism = [{"idx": int(recs[k]["idx"]), "new": str(new[k].value), "old": str(old[k])}
            for k in range(len(texts)) if str(new[k].value) != str(old[k])]
    return {
        "file": d["_path"], "n": len(pp),
        "recorded_num_correct": int(d["num_correct"]), "rescored_num_correct": int(n_ok),
        "recorded_accuracy": float(d["accuracy"]), "rescored_accuracy": n_ok / len(pp),
        "correct_flags_match": f"{flags_match}/{len(pp)}",
        "records": len(texts), "complete_records": len(complete),
        "complete_answer_match": f"{comp_ans}/{len(complete)}",
        "complete_correct_match": f"{comp_cor}/{len(complete)}",
        "old_vs_new_mismatches": mism,
    }


def main():
    gold, _ = gold_answers()
    evals = load_dir("results/tokens/eval_gh200/gsm8k_results_*.json")
    rows = [check(d, gold) for d in evals]
    primary = next(r for r in rows if r["file"] == PRIMARY)
    allexact = all(r["recorded_num_correct"] == r["rescored_num_correct"] for r in rows)
    out = {"primary": primary, "all_files_exact": allexact,
           "total_records": sum(r["records"] for r in rows),
           "total_complete": sum(r["complete_records"] for r in rows),
           "total_old_new_mismatch": sum(len(r["old_vs_new_mismatches"]) for r in rows),
           "files": rows}
    p = write_json("t0_rescore_check.json", out)
    print(f"primary: {primary['recorded_num_correct']} recorded vs {primary['rescored_num_correct']} "
          f"rescored ({primary['recorded_accuracy']:.6f} / {primary['rescored_accuracy']:.6f}); "
          f"complete records answer {primary['complete_answer_match']}, "
          f"correct {primary['complete_correct_match']}")
    print(f"{len(rows)} files, all exact={allexact}; records {out['total_records']} "
          f"(complete {out['total_complete']}); old/new mismatches {out['total_old_new_mismatch']}")
    for r in rows:
        if r["recorded_num_correct"] != r["rescored_num_correct"] or r["old_vs_new_mismatches"] \
                or r["complete_answer_match"].split("/")[0] != r["complete_answer_match"].split("/")[1]:
            print("  !", r["file"], r["recorded_num_correct"], r["rescored_num_correct"],
                  r["complete_answer_match"], r["complete_correct_match"], r["old_vs_new_mismatches"][:3])
    print("->", p)


if __name__ == "__main__":
    main()

"""The GSM8K evaluator must never crash on a degenerate generation.

A diffusion model decoding raw bits can emit anything, and guidance makes
degenerate output MORE likely -- strong guidance saturates the bit posterior.
An evaluator that raises on such a sample takes the whole sweep cell with it,
which is exactly what happened to cfg_coarse cells 2 and 5: a generated
sequence containing a NUL byte made `compile()` raise ValueError (not
SyntaxError), and nothing caught it.

The correct semantics are uniform across methods -- an unparseable or
unexecutable sample is a WRONG ANSWER, never an exception. Uniformity matters
here beyond robustness: if degenerate output crashed for some methods and
scored zero for others, the comparison itself would be biased.
"""
from __future__ import annotations

import re

import pytest

from evaluation.tasks.sandbox_gsm8k import _extract_code, evaluate_samples, predict_answer

GOLD = "The answer is 42\n#### 42"

NUL = chr(0)

DEGENERATE = [
    pytest.param("def simple_math_problem():\n    return " + NUL + " 42\n", id="nul_byte"),
    pytest.param(NUL * 3, id="only_nul"),
    pytest.param("", id="empty"),
    pytest.param("   \n\t ", id="whitespace"),
    pytest.param("def simple_math_problem(:\n  return", id="syntax_error"),
    pytest.param("def simple_math_problem():\n    return 1/0\n", id="raises"),
    pytest.param("def simple_math_problem():\n    while True: pass\n", id="infinite_loop"),
    pytest.param("not python at all, just prose", id="prose"),
    pytest.param("def simple_math_problem():\n\treturn 1\n  return 2\n", id="bad_indent"),
    pytest.param(NUL + "a" * 5000, id="long_with_nul"),
    pytest.param("\ufeff\ufffd  def x", id="odd_unicode"),
    pytest.param("```python\n" + NUL + "\n```", id="nul_in_fence"),
]


@pytest.mark.parametrize("sample", DEGENERATE)
def test_extract_code_never_raises(sample):
    _extract_code(sample)


@pytest.mark.parametrize("sample", DEGENERATE)
def test_predict_answer_never_raises(sample):
    predict_answer(sample, 1.0)


@pytest.mark.parametrize("sample", DEGENERATE)
def test_evaluate_samples_scores_degenerate_output_as_wrong(sample):
    assert evaluate_samples(sample, GOLD, 1.0) is False


def test_valid_sample_still_scores_correct():
    """The robustness fix must not turn every sample into a wrong answer."""
    ok = "def simple_math_problem():\n    return 42\n"
    assert evaluate_samples(ok, GOLD, 5.0) is True


def test_fkc_result_filename_honours_the_tag():
    """Two FKC cells differing only in --checkpoint must not collide.

    The FKC branch builds its filename from sampler parameters alone, so
    `aud_A_collab` (base checkpoint) and `aud_B_ourckpt` (CFG checkpoint) --
    identical in every sampler argument -- wrote to the same path and the
    second silently overwrote the first, losing a replication-audit cell.
    """
    import re
    from pathlib import Path

    src = Path("evaluation/tasks/gsm8k_eval.py").read_text()
    # The FKC tag assembly must consume args.tag before the path is built.
    fkc = src[src.index('prop_tag = ('):src.index('out_path = out_dir / f"gsm8k_results_{tag}.json"')]
    assert "args.tag" in fkc, "FKC filename ignores --tag; cells will collide"


def test_both_result_paths_record_the_regime():
    """Regime A and Regime B results must never be pooled by accident.

    The eval has two independent result dicts (FKC and non-FKC). If only one
    records `regime`, half the results become unattributable the moment both
    regimes are in flight.
    """
    from pathlib import Path

    src = Path("evaluation/tasks/gsm8k_eval.py").read_text()
    assert src.count('"regime": args.regime') == 2, (
        "both the FKC and non-FKC result dicts must record --regime"
    )


def test_hpc_scripts_do_not_rely_on_conda_being_on_path():
    """srun steps must not invoke bare `torchrun`.

    env.sh exports COBIT_PYTHON but never touches PATH, and unlike the
    production launcher we do not `conda activate`. A bare `torchrun` under srun
    therefore dies with "execve(): torchrun: No such file or directory" -- which
    is exactly how the first throughput probe failed, 38 s in.
    """
    from pathlib import Path

    for p in Path("scripts/hpc").rglob("*.slurm"):
        src = p.read_text()
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "srun torchrun" not in stripped, (
                f"{p}: use `srun \"$COBIT_PYTHON\" -m torch.distributed.run` instead"
            )


def test_hpc_scripts_have_no_sigpipe_races_under_pipefail():
    """`cmd | head` with `set -o pipefail` fails intermittently with exit 141.

    head closes the pipe, the writer gets SIGPIPE, pipefail propagates 141 and
    `set -e` kills the job -- but only when the writer is still writing, so it
    is a race. It killed one of two otherwise-identical throughput probes.
    """
    from pathlib import Path

    offenders = []
    for p in Path("scripts/hpc").rglob("*.slurm"):
        src = p.read_text()
        if "pipefail" not in src:
            continue
        for n, line in enumerate(src.splitlines(), 1):
            st = line.strip()
            if st.startswith("#") or "|" not in st:
                continue
            if re.search(r"\|\s*head\b", st) and "|| true" not in st:
                offenders.append(f"{p}:{n}")
    assert not offenders, f"unguarded `| head` under pipefail: {offenders}"

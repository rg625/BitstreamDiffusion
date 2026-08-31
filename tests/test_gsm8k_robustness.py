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

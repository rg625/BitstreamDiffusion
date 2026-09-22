from evaluation.tasks.sandbox_gsm8k import evaluate_samples, _extract_gold_answer, _to_number

GOOD = """def simple_math_problem():
    eggs = 16 - 3 - 4
    return eggs * 2
"""
FENCED = "Here is the code:\n```python\ndef simple_math_problem():\n    return 18\n```\nDone."
WRONG = "def simple_math_problem():\n    return 17\n"
NOFN = "def other():\n    return 18\n"
BADIMPORT = "def simple_math_problem():\n    import os\n    return os.getpid()\n"

def test_gold_parse():
    assert _extract_gold_answer("blah\n#### 18") == 18
    assert _extract_gold_answer("#### 1,024") == 1024
    assert _to_number("the answer is $42.00") == 42

def test_correct():
    assert evaluate_samples(GOOD, "#### 18", 5.0) is True

def test_fenced():
    assert evaluate_samples(FENCED, "#### 18", 5.0) is True

def test_wrong():
    assert evaluate_samples(WRONG, "#### 18", 5.0) is False

def test_missing_fn():
    assert evaluate_samples(NOFN, "#### 18", 5.0) is False

def test_blocked_import():
    # os import inside the fn is blocked -> execution fails -> False
    assert evaluate_samples(BADIMPORT, "#### 18", 5.0) is False

if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
    print("ALL SANDBOX TESTS PASSED")


def test_absurd_integer_answer_does_not_kill_the_run():
    """One generated program returning 10**999999 must not discard the cell.

    Python raises above 4,300 digits on int->str. That crash took out the 500k
    control at seed 2 after 768 of 1,319 problems, so the l2r seed-2 comparison
    had nothing to pair against. The value is certainly wrong for GSM8K -- the
    targets are small integers -- so it is recorded, not raised.
    """
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from evaluation.tasks.gsm8k_eval import _answer_to_str

    assert _answer_to_str(None) is None
    assert _answer_to_str(42) == "42"
    assert _answer_to_str(-7) == "-7"

    huge = 10 ** 5000          # 5,001 digits, over the 4,300 limit
    with __import__("pytest").raises(ValueError):
        str(huge)              # the behaviour being guarded against
    assert _answer_to_str(huge) == "<overflow>"

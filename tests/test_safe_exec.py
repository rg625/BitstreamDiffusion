"""CPU tests for the out-of-process GSM8K executor (CODEBASE.md S1, S2)."""
import time

import pytest

from evaluation.tasks.safe_exec import (
    execute_many, execute_program, predict_answer_subprocess,
)
from evaluation.tasks.sandbox_gsm8k import predict_answer

GOOD = "def simple_math_problem():\n    eggs = 16 - 3 - 4\n    return eggs * 2\n"
FENCED = "Here is the code:\n```python\ndef simple_math_problem():\n    return 18\n```\nDone."


def test_ok_and_value():
    r = execute_program(GOOD, 5.0)
    assert r.status == "ok" and r.value == 18 and r.exception_type is None


def test_fenced():
    assert execute_program(FENCED, 5.0).value == 18


def test_syntax_error():
    r = execute_program("def simple_math_problem(:\n    return 1\n", 5.0)
    assert r.status == "syntax_error" and r.exception_type == "SyntaxError"


def test_nul_bytes_are_a_syntax_error():
    r = execute_program("def simple_math_problem():\n    return 1\x00\n", 5.0)
    assert r.status == "syntax_error" and r.value is None


def test_name_error_at_call():
    r = execute_program("def simple_math_problem():\n    return undefined_total + 1\n", 5.0)
    assert r.status == "name_error" and r.phase == "call" and r.exception_type == "NameError"


def test_non_whitelisted_builtin_is_a_name_error():
    # `list` is not in the whitelist; the old sandbox fails it the same way.
    r = execute_program("def simple_math_problem():\n    return len(list(range(3)))\n", 5.0)
    assert r.status == "name_error"


def test_runtime_error():
    r = execute_program("def simple_math_problem():\n    return 1 / 0\n", 5.0)
    assert r.status == "runtime_error" and r.exception_type == "ZeroDivisionError"


def test_top_level_value_error_is_runtime_not_syntax():
    r = execute_program("def simple_math_problem():\n    return 1\nx = int('abc')\n", 5.0)
    assert r.status == "runtime_error" and r.exception_type == "ValueError"


def test_blocked_import():
    r = execute_program("def simple_math_problem():\n    import os\n    return os.getpid()\n", 5.0)
    assert r.status == "runtime_error" and r.exception_type == "ImportError"


def test_no_function():
    r = execute_program("def other():\n    return 18\n", 5.0)
    assert r.status == "no_return" and r.no_function


def test_returns_none():
    r = execute_program("def simple_math_problem():\n    x = 3\n", 5.0)
    assert r.status == "no_return" and not r.no_function and r.raw_type == "none"


def test_stdout_fallback_matches_old_sandbox():
    prog = "def simple_math_problem():\n    print('the answer is 42')\n"
    assert execute_program(prog, 5.0).value == 42 == predict_answer(prog, 5.0)


def test_bare_except_infinite_loop_is_killed():
    # Defeats the old SIGALRM timeout (S2): _Timeout is an Exception subclass.
    prog = ("def simple_math_problem():\n"
            "    while True:\n"
            "        try:\n"
            "            while True:\n"
            "                pass\n"
            "        except:\n"
            "            pass\n")
    t0 = time.time()
    r = execute_program(prog, 1.0)
    assert r.status == "timeout" and r.value is None
    assert time.time() - t0 < 6.0


def test_huge_allocation_is_a_memory_error():
    t0 = time.time()
    r = execute_program("def simple_math_problem():\n    a = [0] * 10**10\n    return len(a)\n", 5.0)
    assert r.status == "memory" and r.value is None
    assert time.time() - t0 < 10.0


def test_memory_error_swallowed_by_bare_except_still_bounded():
    prog = ("def simple_math_problem():\n"
            "    try:\n"
            "        a = [0] * 10**10\n"
            "    except:\n"
            "        return 7\n")
    r = execute_program(prog, 5.0)
    assert r.status == "ok" and r.value == 7


def test_cannot_write_files(tmp_path):
    # Even via the classic builtins escape, writes are blocked by RLIMIT_FSIZE
    # and the child's cwd is a throwaway directory.
    target = tmp_path / "pwned.txt"
    prog = ("def simple_math_problem():\n"
            "    for c in ().__class__.__base__.__subclasses__():\n"
            "        if c.__name__ == 'FileIO':\n"
            f"            c({str(target)!r}, 'w').write(b'x' * 100)\n"
            "    return 1\n")
    execute_program(prog, 5.0)
    assert not target.exists() or target.stat().st_size == 0


def test_cannot_fork():
    prog = ("def simple_math_problem():\n"
            "    for c in ().__class__.__base__.__subclasses__():\n"
            "        if c.__name__ == '_wrap_close':\n"
            "            return c.__init__.__globals__['fork']()\n"
            "    return -1\n")
    r = execute_program(prog, 5.0)
    assert r.status in ("runtime_error", "ok")
    if r.status == "ok":
        assert r.value == -1          # escape route absent in this interpreter


def test_huge_int_return_is_recorded_not_raised():
    r = execute_program("def simple_math_problem():\n    return 10**5000\n", 5.0)
    assert r.status == "ok" and r.value == 10 ** 5000
    assert r.to_dict()["value"] == "<overflow>"


@pytest.mark.parametrize("prog", [
    GOOD, FENCED,
    "def simple_math_problem():\n    return 2.5\n",
    "def simple_math_problem():\n    return 3.0000001\n",
    "def simple_math_problem():\n    return float('nan')\n",
    "def simple_math_problem():\n    return '$1,024'\n",
    "def simple_math_problem():\n    return True\n",
    "def simple_math_problem():\n    return [1, 2]\n",
    "def simple_math_problem():\n    return math.sqrt(16)\n",
    "def simple_math_problem():\n    import math\n    return math.floor(7.9)\n",
    "def simple_math_problem():\n    x = 5\n    return x *\n",          # truncated mid-statement
    "def simple_math_problem():\n    a = 3\n    b = 4\n    return a + b\nprint(oops",  # trailing junk
    "def simple_math_problem():\n    print(12)\n",
    "",
])
def test_answer_identical_to_old_sandbox(prog):
    assert predict_answer_subprocess(prog, 5.0) == predict_answer(prog, 5.0)


def test_execute_many_preserves_order():
    progs = [f"def simple_math_problem():\n    return {i}\n" for i in range(12)]
    assert [r.value for r in execute_many(progs, 5.0, max_workers=4)] == list(range(12))

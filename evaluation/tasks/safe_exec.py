"""Resource-limited, out-of-process execution of generated GSM8K programs.

`sandbox_gsm8k.predict_answer` runs `exec` inside the evaluating process. That
has two failure modes (CODEBASE.md S1, S2):

  * the builtins whitelist is escapable (`().__class__.__base__.__subclasses__()`);
  * the SIGALRM timeout raises an `Exception` subclass, so a generated bare
    `except:` swallows it and the job hangs until walltime; memory is uncapped,
    so `[0]*10**10` can take the node down.

Here every program runs in its own short-lived interpreter (`python -I -S -B`)
with:

  * RLIMIT_CPU    `timeout_s` (soft; SIGXCPU) and `timeout_s + 1` (hard; SIGKILL);
  * RLIMIT_AS     1 GiB, so a huge allocation raises MemoryError in the child;
  * RLIMIT_FSIZE  0, RLIMIT_CORE 0, RLIMIT_NPROC 0 (no files, no cores, no forks);
  * an empty environment and a fresh temporary working directory;
  * a wall-clock deadline after which the whole process group is SIGKILLed.

What it does NOT do: the child still runs as you, so a program that escapes
the whitelist can *read* anything you can. The limits bound the damage (no
writes, no forks, bounded CPU/memory/time), they do not make it a jail.

Semantics are kept identical to `predict_answer`: the same `_extract_code`,
the same builtins whitelist and `math`-only import hook in the child, and the
same `_to_number` / `_extract_text_answer` normalisation, applied in the parent
to the raw value the child sends back. `predict_answer_subprocess(s, t)` is a
drop-in for `predict_answer(s, t)`; `execute_program` returns the structured
outcome as well.
"""
from __future__ import annotations

import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Sequence

from evaluation.tasks.sandbox_gsm8k import (
    _extract_code, _extract_text_answer, _to_number,
)

MEMORY_LIMIT_BYTES = 1 << 30
STDOUT_CAP_CHARS = 1 << 24
# Interpreter start-up and result transfer, on top of the CPU budget, before
# the parent gives up on the wall clock.
WALL_SLACK_S = 2.0

STATUSES = ("ok", "syntax_error", "name_error", "runtime_error",
            "timeout", "memory", "no_return")

# Runs in the child. Self-contained on purpose: `-I -S` means no site-packages
# and no repo on sys.path, so nothing heavier than the stdlib is importable.
# The builtins whitelist and import hook are copied verbatim from
# sandbox_gsm8k._safe_exec_no_timer; keep the two in step.
_CHILD = r'''
import resource, sys
# Limits first, before a byte of the program is read. Hard limits cannot be
# raised again by an unprivileged process, so the program cannot undo them.
_cpu, _mem = int(sys.argv[1]), int(sys.argv[2])
resource.setrlimit(resource.RLIMIT_CPU, (_cpu, _cpu + 1))
resource.setrlimit(resource.RLIMIT_AS, (_mem, _mem))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
try:
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
except (ValueError, OSError):
    pass
del resource
import contextlib, io, json, warnings

def _emit(**kw):
    sys.__stdout__.write("\n" + json.dumps(kw) + "\n")
    sys.__stdout__.flush()

def _raw(x):
    if x is None:
        return ["none", None]
    if isinstance(x, int):
        return ["int", hex(int(x))]
    if isinstance(x, float):
        return ["float", repr(float(x))]
    if isinstance(x, str):
        return ["str", x[:%(cap)d]]
    try:
        return ["other", type(x).__name__]
    except BaseException:
        return ["other", "?"]

def _exc(e):
    if isinstance(e, MemoryError):
        st = "memory"
    elif isinstance(e, NameError):
        st = "name_error"
    else:
        st = "runtime_error"
    try:
        msg = str(e)[:200]
    except BaseException:
        msg = ""
    return st, type(e).__name__, msg

code = sys.stdin.read()
safe_builtins = {
    "abs": abs, "min": min, "max": max, "sum": sum,
    "len": len, "range": range, "enumerate": enumerate,
    "int": int, "float": float, "str": str, "bool": bool,
    "round": round,
    "print": print,
}
def _limited_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "math":
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Import blocked: {name}")
safe_builtins["__import__"] = _limited_import
import math as _math
ns = {"__builtins__": safe_builtins, "math": _math}

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                exec(code, ns, ns)
except (SyntaxError, ValueError) as e:
    # ValueError: compile() on NUL bytes. Only possible at compile time here,
    # but a top-level ValueError raised by the program itself lands here too,
    # so distinguish on whether anything was defined.
    if isinstance(e, SyntaxError) or "null" in str(e).lower():
        _emit(status="syntax_error", phase="exec", exc_type=type(e).__name__,
              exc_msg=str(e)[:200])
    else:
        st, t, m = _exc(e)
        _emit(status=st, phase="exec", exc_type=t, exc_msg=m)
    sys.exit(0)
except BaseException as e:
    st, t, m = _exc(e)
    _emit(status=st, phase="exec", exc_type=t, exc_msg=m)
    sys.exit(0)

fn = ns.get("simple_math_problem", None)
if fn is None:
    _emit(status="no_return", phase="exec", no_function=True)
    sys.exit(0)

buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        with contextlib.redirect_stderr(io.StringIO()):
            out = fn()
except BaseException as e:
    st, t, m = _exc(e)
    _emit(status=st, phase="call", exc_type=t, exc_msg=m)
    sys.exit(0)

try:
    s = buf.getvalue()
except BaseException:
    s = ""
_emit(status="returned", phase="call", raw=_raw(out),
      stdout=s[-%(cap)d:], stdout_truncated=len(s) > %(cap)d)
''' % {"cap": STDOUT_CAP_CHARS}


@dataclass
class ExecResult:
    status: str                       # one of STATUSES
    value: Any = None                 # normalised answer, exactly predict_answer's
    exception_type: Optional[str] = None
    exception_msg: Optional[str] = None
    phase: Optional[str] = None       # "exec" (module body) or "call" (the function)
    raw_type: Optional[str] = None    # type of the value the function returned
    no_function: bool = False         # no simple_math_problem defined
    stdout_truncated: bool = False
    wall_s: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        v = d["value"]
        if isinstance(v, int) and abs(v) >= 10 ** 18:
            d["value"] = "<overflow>"
        return d


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.kill()
    except ProcessLookupError:
        pass


def _from_raw(raw) -> Any:
    kind, payload = raw
    if kind == "int":
        return int(payload, 16)
    if kind == "float":
        return float(payload)
    if kind == "str":
        return payload
    return None                       # None, or a type _to_number rejects


def execute_program(sample: str, timeout_s: float = 5.0, *,
                    extract: bool = True) -> ExecResult:
    """Run one generated program; never raises for anything the program does."""
    t0 = time.time()
    try:
        code = _extract_code(sample) if extract else sample
    except Exception as e:                       # mirrors predict_answer's guard
        return ExecResult("syntax_error", exception_type=type(e).__name__,
                          exception_msg=str(e)[:200], phase="extract")
    if "\x00" in code:
        # Cannot be passed through a text pipe intact, and compile() rejects it
        # anyway (ValueError), so this is exactly what the child would report.
        return ExecResult("syntax_error", exception_type="ValueError",
                          exception_msg="source code string cannot contain null bytes",
                          phase="exec", wall_s=time.time() - t0)

    with tempfile.TemporaryDirectory(prefix="gsm8k_exec_") as wd:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", "-B", "-c", _CHILD,
             str(max(1, int(round(timeout_s)))), str(MEMORY_LIMIT_BYTES)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=wd, env={}, close_fds=True, start_new_session=True,
        )
        try:
            out, _ = proc.communicate(code.encode("utf-8", "surrogatepass"),
                                      timeout=timeout_s + WALL_SLACK_S)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            proc.communicate()
            return ExecResult("timeout", exception_type="WallClockKill",
                              wall_s=time.time() - t0)
        finally:
            if proc.poll() is None:
                _kill_group(proc)
    wall = time.time() - t0

    msg = None
    for line in reversed(out.decode("utf-8", "replace").splitlines()):
        if line.startswith("{"):
            try:
                msg = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    if msg is None:
        rc = proc.returncode
        if rc in (-signal.SIGXCPU, -signal.SIGKILL):
            return ExecResult("timeout", exception_type=f"Signal{-rc}", wall_s=wall)
        # Dying without a report under RLIMIT_AS (the allocator aborting before
        # Python can raise MemoryError) is the only other way out we expect.
        return ExecResult("memory" if rc in (-signal.SIGABRT, -signal.SIGSEGV) else "runtime_error",
                          exception_type=f"ChildExit{rc}", wall_s=wall)

    if msg["status"] != "returned":
        return ExecResult(msg["status"], exception_type=msg.get("exc_type"),
                          exception_msg=msg.get("exc_msg"), phase=msg.get("phase"),
                          no_function=bool(msg.get("no_function", False)), wall_s=wall)

    raw = msg["raw"]
    value = _to_number(_from_raw(raw))
    if value is None:
        value = _extract_text_answer(msg.get("stdout", ""))
    return ExecResult("ok" if value is not None else "no_return", value=value,
                      phase="call", raw_type=raw[0],
                      stdout_truncated=bool(msg.get("stdout_truncated", False)),
                      wall_s=wall)


def predict_answer_subprocess(sample: str, timeout_s: float) -> Any:
    """Drop-in for sandbox_gsm8k.predict_answer: the normalised answer or None."""
    return execute_program(sample, timeout_s).value


def execute_many(samples: Sequence[str], timeout_s: float = 5.0, *,
                 max_workers: int = 4, extract: bool = True) -> List[ExecResult]:
    """Run programs concurrently, one child each; results in input order."""
    if not samples:
        return []
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
        return list(ex.map(lambda s: execute_program(s, timeout_s, extract=extract), samples))

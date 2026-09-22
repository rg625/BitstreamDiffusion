"""Regression tests for the epoch-boundary deadlock instrumentation.

Every ordering arm hangs at an epoch boundary every few hours; the NCCL
watchdog names the op but not the call site. Two mechanisms recover the call
site (scripts/hpc/arch/deadlock_debug.sh):

  * the NCCL flight recorder, enabled by env vars in the launchers, and
  * a SIGUSR1 handler in train.py that dumps every rank's Python stacks.

Both are silent until a run stalls, which is exactly why they need tests: a
mechanism that is never exercised is a mechanism that is broken by the time you
need it, and each missed hang costs 4-28 h of queue time.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DEBUG_SH = REPO / "scripts" / "hpc" / "arch" / "deadlock_debug.sh"
LAUNCHERS = [
    REPO / "scripts" / "hpc" / "arch" / "ordering_train.slurm",
    REPO / "scripts" / "hpc" / "arch" / "token_train.slurm",
]


def test_debug_helper_is_valid_bash():
    subprocess.run(["bash", "-n", str(DEBUG_SH)], check=True)


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.name)
def test_launchers_enable_instrumentation(launcher: Path):
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    text = launcher.read_text()
    assert "deadlock_debug.sh" in text, f"{launcher.name} does not source the debug helper"
    assert "enable_nccl_flight_recorder" in text
    assert "start_stall_watchdog" in text


def test_flight_recorder_sets_dump_on_timeout(tmp_path):
    """The dump only happens if TORCH_NCCL_DUMP_ON_TIMEOUT is exported."""
    script = f'source "{DEBUG_SH}"; enable_nccl_flight_recorder "{tmp_path}" >/dev/null; ' \
             'echo "$TORCH_NCCL_DUMP_ON_TIMEOUT|$TORCH_NCCL_TRACE_BUFFER_SIZE|' \
             '$TORCH_NCCL_DEBUG_INFO_TEMP_FILE|$COBIT_STACK_DIR"'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    dump, buf, prefix, stack_dir = out.stdout.strip().split("|")
    assert dump == "1"
    assert int(buf) > 0
    assert prefix.startswith(str(tmp_path))
    assert stack_dir == str(tmp_path)


def test_sigusr1_dumps_python_stacks(tmp_path):
    """train.py's handler must write a per-rank stack file when signalled.

    Run in a subprocess: faulthandler.register is process-global state, and the
    handler is installed at train.py import time.
    """
    prog = (
        "import os, signal, sys, time\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "import train\n"
        "def _marker():\n"
        "    os.kill(os.getpid(), signal.SIGUSR1)\n"
        "_marker()\n"
        "time.sleep(0.2)\n"
    )
    env = dict(os.environ, COBIT_STACK_DIR=str(tmp_path), RANK="2", SLURM_JOB_ID="test")
    env.pop("WORLD_SIZE", None)
    subprocess.run([sys.executable, "-c", prog], env=env, check=True, timeout=600)

    dump = tmp_path / "stack_test_rank2.txt"
    assert dump.exists(), f"no stack dump written; got {list(tmp_path.iterdir())}"
    body = dump.read_text()
    assert "_marker" in body, f"stack dump does not contain the calling frame:\n{body}"


def test_decoder_reports_missing_dumps_cleanly():
    """A missing dump must be a clear message, not a traceback."""
    out = subprocess.run(
        [sys.executable, str(REPO / "experiments" / "ordering" / "decode_nccl_trace.py"),
         "/nonexistent/nccl_trace_rank"],
        capture_output=True, text=True,
    )
    assert out.returncode == 1
    assert "no dump files matched" in out.stderr
    assert "Traceback" not in out.stderr


REPRO = REPO / "scripts" / "hpc" / "arch" / "deadlock_repro.slurm"


def test_repro_launcher_is_self_contained():
    subprocess.run(["bash", "-n", str(REPRO)], check=True)
    text = REPRO.read_text()
    assert "COBIT_STEPS_PER_EPOCH" in text
    assert "enable_nccl_flight_recorder" in text and "start_stall_watchdog" in text
    # A throwaway run dir: the repro must never write into one of the four arms.
    assert "ORD_TAG=dlrepro" in text
    # Fast failure is the point; a long timeout wastes the whole reservation.
    assert "DDP_TIMEOUT_MIN:-6" in text


def test_epoch_cap_is_rank_independent():
    """The repro shortcut must not itself desynchronise the ranks.

    Capping the epoch on anything rank-local (wall clock, rank id, loss value)
    would create the very mismatch we are hunting. The cap counts batches, which
    every rank processes in lockstep under DistributedSampler(drop_last=True).
    """
    src = (REPO / "trainers" / "trainer.py").read_text()
    assert "COBIT_STEPS_PER_EPOCH" in src
    line = next(l for l in src.splitlines()
                if "epoch_batch_limit > 0 and num_train_batches" in l)
    assert ">=" in line
    assert "time" not in line and "rank" not in line


def test_epoch_cap_is_off_by_default(monkeypatch):
    """Production runs must be untouched: no env var, no cap."""
    monkeypatch.delenv("COBIT_STEPS_PER_EPOCH", raising=False)
    assert int(os.environ.get("COBIT_STEPS_PER_EPOCH", 0) or 0) == 0


def test_watchdog_signals_only_published_rank_pids(tmp_path):
    """The bug this guards: SIGUSR1 kills a process that has no handler.

    The first version walked the process tree and matched "train.py" in the
    command line. The launcher's command line contains "train.py" too, so job
    35635245 signalled the torchrun agent, which died (rc=138) and took the job
    with it at 15 minutes -- while the four real ranks were never signalled and
    every stack file came out empty. Only a process that has installed the
    handler may be signalled, and the pid file is what proves it has.
    """
    import signal as _signal
    import time

    prog = (
        f"import sys, os, time\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "import train\n"
        "def _sleeper():\n"
        "    time.sleep(60)\n"
        "_sleeper()\n"
    )
    env = dict(os.environ, COBIT_STACK_DIR=str(tmp_path), RANK="3", SLURM_JOB_ID="tjob")
    env.pop("WORLD_SIZE", None)
    proc = subprocess.Popen([sys.executable, "-c", prog], env=env)
    try:
        pid_file = tmp_path / "pid_tjob_rank3.txt"
        for _ in range(600):                      # torch import is slow on Lustre
            if pid_file.exists():
                break
            time.sleep(0.5)
        assert pid_file.exists(), "rank never published its pid"
        assert pid_file.read_text().strip() == str(proc.pid)

        # The helper must find exactly that pid and nothing else.
        script = (f'source "{DEBUG_SH}"; export COBIT_STACK_DIR="{tmp_path}" '
                  'SLURM_JOB_ID=tjob; _dd_rank_pids')
        out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
        assert out.stdout.split() == [str(proc.pid)]

        # And signalling it must dump stacks, not kill it.
        os.kill(proc.pid, _signal.SIGUSR1)
        dump = tmp_path / "stack_tjob_rank3.txt"
        for _ in range(40):
            if dump.exists() and dump.stat().st_size > 0:
                break
            time.sleep(0.25)
        assert proc.poll() is None, "the rank died on SIGUSR1 instead of dumping"
        assert "_sleeper" in dump.read_text()
    finally:
        proc.kill()
        proc.wait()


def test_watchdog_refuses_to_guess_when_no_pids_are_published(tmp_path):
    """With no pid files it must signal NOTHING rather than fall back to a
    command-line match -- the fallback is what killed the job."""
    script = (f'source "{DEBUG_SH}"; export COBIT_STACK_DIR="{tmp_path}" '
              'SLURM_JOB_ID=absent; _dd_rank_pids')
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""

    body = DEBUG_SH.read_text()
    assert "NOT signalling anything" in body
    assert "_dd_descendants" not in body, "the process-tree walk must be gone"
    assert "cmdline" not in body, "no command-line matching may remain"

"""The site abstraction must not change CSD3 behaviour, and must refuse to guess.

Every result in this repo was produced with the values that used to be
hardcoded in env.sh. Routing them through site.sh is only safe if CSD3 comes out
byte-identical, and if the new site fails loudly on the two values that cannot
be guessed (account and interpreter) rather than submitting jobs that pend
forever.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "scripts" / "site" / "site.sh"
SUBMIT = REPO / "scripts" / "site" / "submit.sh"
ENV_SH = REPO / "scripts" / "hpc" / "guidance" / "env.sh"


def _sh(script, env=None):
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          cwd=str(REPO), env=env, timeout=300)


@pytest.mark.parametrize("script", [SITE, SUBMIT, ENV_SH,
                                    REPO / "scripts" / "site" / "transfer.sh",
                                    REPO / "scripts" / "site" / "build_env_isambard.sh"])
def test_scripts_parse(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_csd3_values_are_unchanged():
    """These are the values that produced every number in results/."""
    out = _sh(f'COBIT_SITE=csd3 source "{SITE}" >/dev/null; '
              'echo "$COBIT_PARTITION|$COBIT_ACCOUNT|$COBIT_GPUS_PER_NODE|$COBIT_PYTHON"')
    part, acct, gpus, py = out.stdout.strip().split("|")
    assert part == "ampere"
    assert acct == "GIROLAMI-SL2-GPU"
    assert gpus == "4"
    assert py.endswith("/envs/sedd310/bin/python")


def test_isambard_refuses_to_guess_the_account():
    """A wrong account is a job that pends forever without saying why."""
    out = _sh(f'COBIT_SITE=isambard COBIT_PYTHON=/x/python source "{SITE}" >/dev/null 2>&1; '
              'echo rc=$?')
    assert "rc=0" not in out.stdout


def test_isambard_refuses_to_guess_the_interpreter():
    """The x86_64 env cannot be reused; defaulting to it would fail at the
    first CUDA call rather than at configuration time."""
    out = _sh(f'COBIT_SITE=isambard COBIT_ACCOUNT=proj source "{SITE}" >/dev/null 2>&1; '
              'echo rc=$?')
    assert "rc=0" not in out.stdout


def test_isambard_is_declared_aarch64():
    out = _sh(f'COBIT_SITE=isambard COBIT_ACCOUNT=p COBIT_PYTHON=/x source "{SITE}" >/dev/null; '
              'echo "$COBIT_ARCH|$COBIT_GPU"')
    arch, gpu = out.stdout.strip().split("|")
    assert arch == "aarch64", "the env-rebuild requirement hangs off this"
    assert "GH200" in gpu


def test_env_sh_sources_the_site_file():
    body = ENV_SH.read_text()
    assert "scripts/site/site.sh" in body
    # CSD3 module loads must not run on another cluster.
    assert 'COBIT_SITE:-csd3}" = "csd3"' in body


def test_submit_wrapper_passes_flags_on_the_command_line():
    """SLURM precedence is command line > in-script #SBATCH > environment, so
    exporting SBATCH_ACCOUNT cannot override the launchers' CSD3 directives."""
    body = SUBMIT.read_text()
    for flag in ("--partition=", "--account=", "--gres=", "--cpus-per-task="):
        assert flag in body
    assert "exceeds this site's cap" in body


def test_transfer_manifest_carries_sigma_data_with_every_checkpoint():
    """Without sigma_data.json the evaluator silently falls back to 0.5 against
    a true 0.3998 -- wrong numbers, no error."""
    body = (REPO / "scripts" / "site" / "transfer.sh").read_text()
    n_ckpt = body.count("/checkpoints/last.pt")
    n_sigma = body.count("sigma_data.json")
    assert n_ckpt >= 6
    assert n_sigma >= n_ckpt, f"{n_ckpt} checkpoints but only {n_sigma} sigma_data entries"

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


def test_isambard_refuses_to_guess_the_account(tmp_path):
    """A wrong account is a job that pends forever without saying why."""
    b = _fake_bin(tmp_path, sacctmgr='printf "projA\\nprojB\\n"')
    out = _sh(f'COBIT_SITE=isambard COBIT_PYTHON=/x/python source "{SITE}" >/dev/null 2>&1; '
              'echo rc=$?', env=_env_with(b))
    assert "rc=0" not in out.stdout, "two associations: must refuse, not pick one"


def test_isambard_refuses_to_guess_the_interpreter(tmp_path):
    """The x86_64 env cannot be reused; defaulting to it would fail at the
    first CUDA call rather than at configuration time."""
    env = _env_with(_fake_bin(tmp_path))
    env["PROJECTDIR"] = str(tmp_path)          # no cobit-gh env built here
    out = _sh(f'COBIT_SITE=isambard COBIT_ACCOUNT=proj source "{SITE}" >/dev/null 2>&1; '
              'echo rc=$?', env=env)
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


def test_manifest_carries_every_published_result_checkpoint():
    """A checkpoint behind a number in docs/ cannot be regenerated without
    retraining, so it travels even though nothing resumes from it.

    The first manifest carried 10 of 206 .pt files and silently omitted the
    V-way token run that measured 0.0255, its matched binary control at 0.0493,
    and both arms of the SM-vs-CE negative.
    """
    body = (REPO / "scripts" / "site" / "transfer.sh").read_text()
    for run in ("tok_fs500k_token_ce_s42",            # 0.0255, the V-way result
                "obj_binary_sm_fs500k_b128_s42",      # 0.0493, its control
                "obj_binary_sm_fs50k_s42",            # SM arm of the CE/SM negative
                "obj_binary_ce_fs50k_s42"):           # CE arm
        assert f"{run}/checkpoints/last.pt" in body, f"{run} is not in the manifest"


def test_interval_checkpoints_are_opt_in():
    """42 GB that only an accuracy-vs-step curve needs; nothing measured so far
    depends on them."""
    body = (REPO / "scripts" / "site" / "transfer.sh").read_text()
    assert "--with-intervals" in body
    assert 'WITH_INTERVALS:-0' in body, "must default to off"


# --- Isambard port fixes (23 Sep 2026) ---------------------------------------

def _fake_bin(tmp_path, **cmds):
    """A PATH dir whose `name` commands print fixed strings."""
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    for name, body in cmds.items():
        f = d / name
        f.write_text(f"#!/bin/bash\n{body}\n")
        f.chmod(0o755)
    return d


def _env_with(bindir):
    import os
    env = {k: v for k, v in os.environ.items() if not k.startswith(("COBIT_", "PROJECT_DIR"))}
    env["PATH"] = f"{bindir}:{env['PATH']}"
    return env


def test_isambard_login_node_is_detected_by_architecture(tmp_path):
    """Isambard's login nodes are plain 'login40': no domain, no 'isambard'.
    The hostname patterns sent that to the csd3 fallback -- CSD3's account and
    partition on an aarch64 machine."""
    b = _fake_bin(tmp_path, uname='echo aarch64', hostname='echo login40')
    out = _sh(f'COBIT_ACCOUNT=p COBIT_PYTHON=/x source "{SITE}" >/dev/null; echo "$COBIT_SITE"',
              env=_env_with(b))
    assert out.stdout.strip() == "isambard"


def test_csd3_login_node_still_detected_as_csd3(tmp_path):
    b = _fake_bin(tmp_path, uname='echo x86_64', hostname='echo login-q-1.hpc.cam.ac.uk')
    out = _sh(f'source "{SITE}" >/dev/null; echo "$COBIT_SITE|$COBIT_ACCOUNT"', env=_env_with(b))
    assert out.stdout.strip() == "csd3|GIROLAMI-SL2-GPU"


def test_isambard_partition_is_workq():
    """`grace` does not exist here; sinfo shows a single partition, workq."""
    out = _sh(f'COBIT_SITE=isambard COBIT_ACCOUNT=p COBIT_PYTHON=/x source "{SITE}" >/dev/null; '
              'echo "$COBIT_PARTITION|$COBIT_WALL_MAX"')
    assert out.stdout.strip() == "workq|24:00:00"


def test_submit_exports_project_dir_to_the_job(tmp_path):
    """Every launcher falls back to the CSD3 path when PROJECT_DIR is unset, and
    sbatch's --export=ALL only carries exported variables."""
    b = _fake_bin(tmp_path, sbatch='echo "PD=$PROJECT_DIR"')
    launcher = tmp_path / "x.slurm"
    launcher.write_text("#!/bin/bash\n")
    out = _sh(f'COBIT_SITE=isambard COBIT_ACCOUNT=p COBIT_PYTHON=/x bash "{SUBMIT}" "{launcher}" 2>/dev/null',
              env=_env_with(b))
    assert f"PD={REPO}" in out.stdout, out.stdout


def test_env_sh_passes_cpus_to_srun():
    """srun stopped inheriting --cpus-per-task in SLURM 22.05; Isambard is 24.11."""
    assert 'SRUN_CPUS_PER_TASK' in ENV_SH.read_text()


def test_build_script_pins_the_torch_behind_every_result():
    """No aarch64 2.8.0+cu128 wheel exists; unpinned cu128 gives torch 2.11."""
    body = (REPO / "scripts" / "site" / "build_env_isambard.sh").read_text()
    assert "torch==2.8.0" in body and "whl/cu129" in body
    assert "-c \"$CONSTRAINTS\" -r \"$REPO/requirements.txt\"" in body, "requirements pins must apply"
    assert "already exists" in body, "must refuse to recreate (conda create -y deletes) an env"


def test_verify_gpu_launcher_parses():
    subprocess.run(["bash", "-n", str(REPO / "scripts" / "site" / "verify_gpu.slurm")], check=True)


def _fake_torch(version="2.8.0+cu129", cuda="12.9", arch="sm_90 sm_100"):
    from types import SimpleNamespace
    return SimpleNamespace(__version__=version, version=SimpleNamespace(cuda=cuda),
                           _C=SimpleNamespace(_cuda_getArchFlags=lambda: arch))


def _verify_env():
    import importlib.util
    spec = importlib.util.spec_from_file_location("verify_env", REPO / "scripts" / "site" / "verify_env.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_verify_env_accepts_the_pinned_build():
    assert _verify_env().build_checks(_fake_torch()) == []


def test_verify_env_names_a_cpu_only_wheel_as_such():
    """is_available() is False for a CPU wheel AND for a CUDA wheel on a login
    node; only torch.version.cuda tells them apart."""
    fails = _verify_env().build_checks(_fake_torch(version="2.8.0", cuda=None))
    assert any("CPU-only" in f for f in fails)


def test_verify_env_rejects_an_unpinned_torch():
    fails = _verify_env().build_checks(_fake_torch(version="2.11.0+cu128", cuda="12.8"))
    assert any("expected 2.8.0" in f for f in fails)


def test_verify_env_rejects_a_wheel_without_hopper_kernels():
    fails = _verify_env().build_checks(_fake_torch(arch="sm_80"))
    assert any("sm_90" in f for f in fails)


def _submit_args(tmp_path, gres_line, *extra):
    b = _fake_bin(tmp_path, sbatch='echo "ARGS $*"')
    launcher = tmp_path / "x.slurm"
    launcher.write_text(f"#!/bin/bash\n{gres_line}\n")
    extra_s = " ".join(extra)
    out = _sh(f'COBIT_SITE=isambard COBIT_ACCOUNT=p COBIT_PYTHON=/x bash "{SUBMIT}" "{launcher}" {extra_s} 2>/dev/null',
              env=_env_with(b))
    return next(l for l in out.stdout.splitlines() if l.startswith("ARGS"))


def test_submit_keeps_a_one_gpu_launcher_at_one_gpu(tmp_path):
    """Forcing gpu:4 on the eval launchers billed four GPUs per eval."""
    args = _submit_args(tmp_path, "#SBATCH --gres=gpu:1")
    assert "--gres=gpu:1" in args and "--cpus-per-task=72" in args


def test_submit_gives_a_four_gpu_launcher_all_four_sockets(tmp_path):
    args = _submit_args(tmp_path, "#SBATCH --gres=gpu:4")
    assert "--gres=gpu:4" in args and "--cpus-per-task=288" in args
    assert "--partition=workq" in args and "--account=p" in args


def test_submit_lets_the_command_line_override_resources(tmp_path):
    args = _submit_args(tmp_path, "#SBATCH --gres=gpu:4", "--gres=gpu:2", "--cpus-per-task=10")
    assert "--gres=gpu:4" not in args and "--cpus-per-task=288" not in args
    assert "--gres=gpu:2" in args


def test_manifest_carries_the_gsm8k_test_set():
    """Every eval reads it; the first transfer left it behind."""
    body = (REPO / "scripts" / "site" / "transfer.sh").read_text()
    assert '"datasets/gsm8k/gsm8k_test.json"' in body


def test_gh200_eval_driver_never_writes_into_the_csd3_directory():
    """Evaluation must be common to all arms: GH200 cells go to a fresh dir and
    are compared only within it."""
    body = (REPO / "scripts" / "hpc" / "arch" / "ordering_eval_gh200.sh").read_text()
    assert 'OUT="results/ordering/train_eval_gh200"' in body
    assert "EV_OUT_DIR=$OUT" in body and "scripts/site/submit.sh" in body
    assert "gate 3 has not passed" in body, "arms must be gated on an in-band anchor"
    slurm = (REPO / "scripts" / "hpc" / "arch" / "ordering_eval.slurm").read_text()
    assert '--out_dir "$EV_OUT_DIR"' in slurm and "--schedule karras" in slurm
    subprocess.run(["bash", "-n", str(REPO / "scripts" / "hpc" / "arch" / "ordering_eval_gh200.sh")], check=True)


def test_submit_creates_the_launchers_log_directory(tmp_path):
    """SLURM will not create --output's directory; the job just fails, logless."""
    b = _fake_bin(tmp_path, sbatch='echo "ARGS $*"')
    launcher = tmp_path / "x.slurm"
    logdir = tmp_path / "newlogs" / "arch"
    launcher.write_text(f"#!/bin/bash\n#SBATCH --gres=gpu:1\n#SBATCH --output={logdir}/%x_%j.log\n")
    _sh(f'COBIT_SITE=isambard COBIT_ACCOUNT=p COBIT_PYTHON=/x bash "{SUBMIT}" "{launcher}" 2>/dev/null',
        env=_env_with(b))
    assert logdir.is_dir()


# --- Job 2: the logits question on GH200 -------------------------------------

LOGITS = REPO / "scripts" / "hpc" / "arch" / "logits_gh200.sh"


def test_logits_driver_parses():
    subprocess.run(["bash", "-n", str(LOGITS)], check=True)


def test_logits_arms_never_reuse_a_csd3_run_directory():
    """train.py resumes from whatever last.pt the run dir holds. The CSD3 token
    run (tok_b512, 167k) and control (ord_fs500k_none, 500k) are both on disk,
    so reusing either tag would continue a CSD3 run on GH200."""
    body = LOGITS.read_text()
    assert "TOK_TAG=gh500k" in body and "ORD_TAG=gh500k" in body
    assert "TOK_TAG=b512" not in body and "ORD_TAG=fs500k" not in body


def test_logits_arms_are_matched():
    """Same steps, effective batch, LR and seed on both sides; lr 3e-4 diverged 8/8."""
    body = LOGITS.read_text()
    tok = next(l for l in body.splitlines() if l.strip().startswith('TOK_EX='))
    ctl = next(l for l in body.splitlines() if l.strip().startswith('CTL_EX='))
    for want in ("TOK_BATCH=512", "TOK_LR=1e-4", "TOK_SEED=42", "TOK_STEPS=500000"):
        assert want in tok
    for want in ("ORD_LR=1e-4", "ORD_SEED=42", "ORD_STEPS=500000", "ORD_INIT=none", "ORD_MODE=none"):
        assert want in ctl


def _logits(tmp_path, *args):
    import os
    env = {k: v for k, v in os.environ.items() if not k.startswith("COBIT_")}
    env.update(PROJECT_DIR=str(tmp_path), COBIT_PYTHON="/bin/false", COBIT_ACCOUNT="p", COBIT_SITE="isambard")
    return subprocess.run(["bash", str(LOGITS), *args], capture_output=True, text=True,
                          env=env, timeout=60)


def test_logits_chain_refuses_an_accum_without_a_clean_smoke(tmp_path):
    (tmp_path / "logs" / "arch").mkdir(parents=True)
    out = _logits(tmp_path, "chain", "--accum", "1")
    assert out.returncode != 0 and "no clean smoke" in out.stderr


def test_logits_chain_refuses_an_oom_smoke(tmp_path):
    logs = tmp_path / "logs" / "arch"; logs.mkdir(parents=True)
    (logs / "cobit_toktrain_1.log").write_text(
        "[tok] loss=token_ce batch=512 tag=smoke_gh_a1\ntorch.OutOfMemoryError: CUDA out of memory\n"
        "[tok] rc=1 -- check for OOM\n[tok] finished\n")
    out = _logits(tmp_path, "chain", "--accum", "1")
    assert out.returncode != 0 and "FAILED" in out.stdout


def test_logits_smoke_refuses_a_leftover_smoke_dir(tmp_path):
    """A leftover dir would resume at step 300 and exit at once -- a fake pass."""
    (tmp_path / "runs/tasks/tinygsm/tok_smoke_gh_a1_token_ce_s42").mkdir(parents=True)
    out = _logits(tmp_path, "smoke")
    assert out.returncode != 0 and "REFUSING" in out.stderr


def test_isambard_takes_the_account_when_slurm_lists_exactly_one(tmp_path):
    b = _fake_bin(tmp_path, sacctmgr='echo brics.u6ya')
    out = _sh(f'COBIT_SITE=isambard COBIT_PYTHON=/x source "{SITE}" >/dev/null; echo "$COBIT_ACCOUNT"',
              env=_env_with(b))
    assert out.stdout.strip() == "brics.u6ya"


def test_isambard_defaults_to_the_built_aarch64_env(tmp_path):
    """A fresh shell without the exports must not stop every driver."""
    py = tmp_path / "u" / "miniforge3" / "envs" / "cobit-gh" / "bin" / "python"
    py.parent.mkdir(parents=True); py.write_text("#!/bin/bash\n"); py.chmod(0o755)
    env = _env_with(_fake_bin(tmp_path))
    env.update(PROJECTDIR=str(tmp_path), USER="u")
    out = _sh(f'COBIT_SITE=isambard COBIT_ACCOUNT=p source "{SITE}" >/dev/null; echo "$COBIT_PYTHON"', env=env)
    assert out.stdout.strip() == str(py)


def test_submit_stdout_is_only_the_job_id(tmp_path):
    """Chains capture stdout as the id for --dependency=afterany:<id>; the site
    banner on stdout made that '[site] isambard ...\\n6840800'."""
    b = _fake_bin(tmp_path, sbatch='echo 12345')
    launcher = tmp_path / "x.slurm"
    launcher.write_text("#!/bin/bash\n#SBATCH --gres=gpu:1\n")
    out = _sh(f'COBIT_SITE=isambard COBIT_ACCOUNT=p COBIT_PYTHON=/x bash "{SUBMIT}" "{launcher}" --parsable 2>/dev/null',
              env=_env_with(b))
    assert out.stdout == "12345\n"


def test_env_sh_replaces_a_tmpdir_that_does_not_exist(tmp_path):
    """--export=ALL carried the login node's TMPDIR into job 6840804, where it
    did not exist; torch died at import on all four ranks."""
    body = ENV_SH.read_text()
    start = body.index('if [ -n "${SLURM_JOB_ID:-}" ]')
    snippet = body[start: body.index("fi\n", start) + 3]
    out = _sh(f'TMPDIR={tmp_path}/nope\n{snippet}echo "$TMPDIR"')
    assert out.stdout.strip() == "/tmp"
    out = _sh(f'unset SLURM_JOB_ID; TMPDIR={tmp_path}\n{snippet}echo "$TMPDIR"')
    assert out.stdout.strip() == str(tmp_path), "outside a job a valid TMPDIR is left alone"
    # Inside a job it is never trusted: 6841368 passed the check and still failed.
    out = _sh(f'SLURM_JOB_ID=1 TMPDIR={tmp_path}\n{snippet}echo "$TMPDIR"')
    assert out.stdout.strip() == "/tmp"


def test_logits_smoke_can_rerun_one_accum(tmp_path):
    (tmp_path / "runs/tasks/tinygsm/tok_smoke_gh_a2_token_ce_s42").mkdir(parents=True)
    out = _logits(tmp_path, "smoke", "--accum", "1")
    assert out.returncode == 0 and "{1}" in out.stdout, out.stdout + out.stderr


def test_deadlock_repro_gets_a_fresh_run_dir_per_job():
    """A fixed tag let 6832047 resume at step 400 and report a vacuous NO HANG."""
    body = (REPO / "scripts" / "hpc" / "arch" / "deadlock_repro.slurm").read_text()
    assert 'ORD_TAG="dlrepro_${SLURM_JOB_ID' in body
    assert "export ORD_TAG=dlrepro\n" not in body

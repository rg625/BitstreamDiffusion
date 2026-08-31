"""Load the pre-refactor sampler straight out of git, as a behavioural oracle.

Phase-5 regression work needs the *old* `diffusion/continuous/samplers.py` --
the `tasks/fkc-temperature` implementation that shipped the original CFG -- to
compare against the refactored one. Rather than vendoring a copy (which would
rot silently), we materialise it from git at a pinned revision and import it
under its own module name.

`REFERENCE_REV` is the last commit before guidance became pluggable.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pytest

REFERENCE_REV = "8c83392"
_REPO = Path(__file__).resolve().parents[1]
_cached: Optional[object] = None


def _git_show(rev: str, path: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "show", f"{rev}:{path}"],
            cwd=_REPO, capture_output=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.decode()


def load_legacy_samplers():
    """Import the pinned pre-refactor samplers module (cached), or skip."""
    global _cached
    if _cached is not None:
        return _cached

    src = _git_show(REFERENCE_REV, "diffusion/continuous/samplers.py")
    if src is None:
        pytest.skip(
            f"reference revision {REFERENCE_REV} unavailable (no git checkout); "
            "cannot run the pre-refactor regression"
        )

    tmp = Path(tempfile.mkdtemp(prefix="cobit_legacy_")) / "legacy_samplers.py"
    tmp.write_text(src)

    spec = importlib.util.spec_from_file_location("cobit_legacy_samplers", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cobit_legacy_samplers"] = mod
    spec.loader.exec_module(mod)
    _cached = mod
    return mod

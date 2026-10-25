"""Training must be importable and runnable without optional extras.

Both of these were real failures that cost GPU jobs:
  * a disabled MAUVE metric made `mauve` a hard dependency of ALL training;
  * diffusion/discrete/losses.py used PEP 604 without postponed annotations,
    which is a runtime TypeError on this project's Python 3.9.
"""
import pathlib
import re
import sys

import pytest


def test_trainer_imports_without_mauve_installed():
    """`mauve` is not in this environment and must not be needed to train."""
    import trainers.trainer  # noqa: F401
    from utils.callbacks import MauveCallback  # noqa: F401

    assert "mauve" not in sys.modules, (
        "importing the trainer pulled in `mauve`; it must stay lazy so a "
        "disabled metric cannot break training"
    )


def test_no_pep604_without_postponed_annotations_on_the_training_path():
    """`X | None` in a signature is a TypeError on Python 3.9 at import time."""
    offenders = []
    for p in list(pathlib.Path("diffusion").rglob("*.py")) + \
             list(pathlib.Path("trainers").rglob("*.py")) + \
             list(pathlib.Path("models").rglob("*.py")) + \
             list(pathlib.Path("data").rglob("*.py")):
        src = p.read_text()
        if "from __future__ import annotations" in src:
            continue
        if re.search(r":\s*[A-Za-z_][\w\.]*\s*\|\s*None\s*[,)=]", src):
            offenders.append(str(p))
    assert not offenders, f"PEP 604 without postponed annotations: {offenders}"


def test_both_loss_arms_run_on_real_data_shapes():
    """The Branch 1 comparison is one config line; both arms must actually run."""
    import torch
    from diffusion.continuous.losses import binary_score_interpolation_loss

    class _C:
        pass

    cfg = _C(); cfg.train = _C(); cfg.diffusion = _C(); cfg.diffusion.continuous = _C()
    cfg.diffusion.continuous.sigma_data = 0.3998
    B, S = 2, 8192
    x0 = (torch.rand(B, S) > 0.5).float()
    logits = torch.randn(B, S)
    sigma = torch.rand(B) * 2 + 0.1
    out = {}
    for lt in ("binary_sm", "binary_ce"):
        cfg.train.loss_type = lt
        loss = binary_score_interpolation_loss(logits, x0, sigma, cfg)
        assert torch.isfinite(loss), lt
        out[lt] = float(loss)
    # They are different objectives, so they must not coincide numerically.
    assert out["binary_sm"] != pytest.approx(out["binary_ce"])

"""NFE must count denoiser forward passes, not sampler steps.

These numbers are not asserted from the solver's source: they were measured by
wrapping `model.forward` and counting calls on a real checkpoint (DDIM 1.00
per step at 4/8/16 steps; Heun 7/15/31 forwards = 2N-1). The test pins the
accounting function against those measurements.
"""
import pytest

from evaluation.tasks.gsm8k_eval import _nfe_per_sample, _solver_evals_per_step


class _G:
    def __init__(self, cfg=False, ag=False, sg=False, variant="prev"):
        self.cfg_enabled, self.ag_enabled = cfg, ag
        self.sg_enabled, self.sg_variant = sg, variant


def test_ddim_is_one_forward_per_step():
    assert _solver_evals_per_step("ddim", 16) == 1.0
    assert _nfe_per_sample(512, _G(), "ddim") == 512.0


@pytest.mark.parametrize("steps,measured", [(4, 7), (8, 15), (16, 31)])
def test_heun_matches_the_measured_forward_counts(steps, measured):
    """Heun is a predictor+corrector per step, with a final Euler step: 2N-1."""
    assert _nfe_per_sample(steps, _G(), "heun") == pytest.approx(float(measured))


def test_the_solver_control_grid_was_compute_matched():
    """Heun-256 vs DDIM-512 is the comparison the SG-vs-solver question needs."""
    assert _nfe_per_sample(256, _G(), "heun") == pytest.approx(511.0)
    assert _nfe_per_sample(512, _G(), "ddim") == 512.0
    assert _nfe_per_sample(512, _G(), "heun") == pytest.approx(1023.0)
    assert _nfe_per_sample(1024, _G(), "ddim") == 1024.0


def test_solver_order_and_guidance_branches_multiply():
    # CFG doubles branches; on Heun that stacks on top of the solver's two.
    assert _nfe_per_sample(512, _G(cfg=True), "ddim") == 1024.0
    assert _nfe_per_sample(256, _G(cfg=True), "heun") == pytest.approx(1022.0)
    # SG-prev is free; SG-exact doubles.
    assert _nfe_per_sample(512, _G(sg=True, variant="prev"), "ddim") == 512.0
    assert _nfe_per_sample(512, _G(sg=True, variant="exact"), "ddim") == 1024.0

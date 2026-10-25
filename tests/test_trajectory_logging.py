"""Per-problem trajectory geometry: correctness, masking, and no-op guarantee.

The batch-aggregated per_step trace cannot support causal claims -- a population
mean cannot say whether the trajectory that diverged is the one that got the
answer wrong. These tests cover the per-row replacement.
"""
import math

import pytest
import torch

from diffusion.continuous.guidance import GuidedDenoiser
from evaluation.guidance_metrics import summarise_trace


def rs(g, s, mask=None, cond_enabled=False):
    return GuidedDenoiser._row_stats(g, s, mask, cond_enabled)


def test_row_stats_matches_hand_computation():
    # Row 0: g exactly parallel to s. Row 1: g exactly orthogonal to s.
    s = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    g = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    r = rs(g, s)
    # norms are RMS over coordinates, so ||[2,0]|| = sqrt(4/2) = sqrt(2)
    assert r["g_norm"][0] == pytest.approx(math.sqrt(2.0))
    assert r["s_norm"][0] == pytest.approx(math.sqrt(0.5))
    assert r["cos"][0] == pytest.approx(1.0, abs=1e-5)
    assert r["cos"][1] == pytest.approx(0.0, abs=1e-5)
    assert r["orthogonal"][0] == pytest.approx(0.0, abs=1e-5)
    assert r["parallel"][1] == pytest.approx(0.0, abs=1e-5)


def test_parallel_orthogonal_decomposition_is_pythagorean():
    """||g||^2 = parallel^2 + orthogonal^2 must hold for arbitrary vectors."""
    torch.manual_seed(0)
    g, s = torch.randn(8, 32), torch.randn(8, 32)
    r = rs(g, s)
    lhs = r["g_norm"] ** 2
    rhs = r["parallel"] ** 2 + r["orthogonal"] ** 2
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_ratio_is_invariant_to_the_shared_dspace_to_score_rescaling():
    """g and s are both D-space; the map to score-space is a shared 1/sigma^2.

    Every reported quantity must therefore be unchanged by that rescaling --
    this is what makes the ratio comparable across sigma and across mechanisms.
    """
    torch.manual_seed(1)
    g, s = torch.randn(4, 16), torch.randn(4, 16)
    a, b = rs(g, s), rs(g * 1e4, s * 1e4)
    assert torch.allclose(a["ratio"], b["ratio"], atol=1e-4)
    assert torch.allclose(a["cos"], b["cos"], atol=1e-4)


def test_prompt_coordinates_are_excluded():
    """Prompt positions are clamped and carry no drift; including them would
    dilute every norm toward zero."""
    g = torch.tensor([[5.0, 5.0, 1.0, 1.0]])
    s = torch.tensor([[5.0, 5.0, 1.0, 1.0]])
    mask = torch.tensor([[True, True, False, False]])   # first half is prompt
    free = rs(g, s, mask, cond_enabled=True)
    allc = rs(g, s, None, cond_enabled=False)
    # Masked: only the two 1.0s count, and the RMS normalises by the FREE count
    # (2), not the total (4) -- so sqrt(2/2) = 1.0. Normalising by the total
    # would make every norm shrink as the prompt grows.
    assert free["g_norm"][0] == pytest.approx(1.0)
    assert allc["g_norm"][0] == pytest.approx(math.sqrt(13.0))


def test_summarise_trace_ignores_the_per_row_payload():
    """_rows holds tensors; the aggregator averages scalars and must skip it."""
    trace = [{"step": 0, "sigma": 1.0, "ratio": 0.5,
              "_rows": {"ratio": torch.tensor([0.1, 0.9])}}]
    out = summarise_trace([trace])
    assert "ratio_mean" in out or "ratio" in str(out)
    assert "_rows" not in str(out.get("per_step", [{}])[0])


def test_per_problem_logging_defaults_off():
    """A default run must be unchanged: the flag is opt-in."""
    import inspect
    src = inspect.getsource(GuidedDenoiser.__init__)
    assert "self.per_problem_diagnostics = False" in src

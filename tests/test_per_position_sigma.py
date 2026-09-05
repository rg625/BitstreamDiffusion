"""Per-position sigma: the refactor must be a no-op at uniform sigma.

CoBit conditions on ONE sigma per example, broadcast to all positions, so it has
no temporal ordering. Ordering requires sigma [B,S]. This gates that refactor:
with the same sigma at every position the model must behave exactly as before.

`torch.equal` matters here, not just `allclose`. An intermediate version of this
refactor emitted (B, B, S, 1) instead of (B, S, 1); the values broadcast-compared
as equal so max|diff| was 0.0 and allclose passed, while torch.equal caught it.
"""
import torch

from models.sdt import AdaLNZero, PreNormBlockAda, _SinTimeSigma


def test_sigma_embedding_accepts_both_ranks():
    torch.manual_seed(0)
    B, S, E = 3, 7, 32
    f = _SinTimeSigma(E)
    sig = torch.rand(B) * 5 + 0.1
    e_global = f(sig)
    e_pos = f(sig[:, None].expand(B, S))
    assert e_global.shape == (B, E)
    assert e_pos.shape == (B, S, E)
    # every position column must equal the global embedding, exactly
    for j in range(S):
        assert torch.equal(e_pos[:, j], e_global)


def test_adaln_per_token_conditioning_matches_global():
    torch.manual_seed(0)
    B, n, d = 3, 7, 32
    a = AdaLNZero(d)
    h, t = torch.randn(B, n, d), torch.randn(B, d)
    hm_g, g_g = a(h, t)
    hm_p, g_p = a(h, t[:, None, :].expand(B, n, d))
    assert hm_g.shape == (B, n, d) and hm_p.shape == (B, n, d)
    assert torch.allclose(hm_g, hm_p, rtol=0, atol=1e-6)
    # the gate must come back broadcast-ready in BOTH cases; callers must not
    # unsqueeze it again (doing so is what produced the (B,B,S,1) bug)
    assert g_g.dim() == 3 and g_p.dim() == 3


def test_block_output_is_shape_preserving_under_both_conditionings():
    torch.manual_seed(0)
    B, n, d = 3, 7, 32
    blk = PreNormBlockAda(d, 4, dim_ff=64).eval()
    h, t = torch.randn(B, n, d), torch.randn(B, d)
    with torch.no_grad():
        o_g = blk(h, t)
        o_p = blk(h, t[:, None, :].expand(B, n, d))
    assert o_g.shape == (B, n, d), f"global conditioning changed shape: {o_g.shape}"
    assert o_p.shape == (B, n, d), f"per-token conditioning changed shape: {o_p.shape}"
    assert torch.allclose(o_g, o_p, rtol=0, atol=1e-6)


def test_no_double_unsqueeze_of_the_adaln_gate():
    """AdaLNZero already broadcasts its gate; a caller that unsqueezes again
    silently adds a batch dimension instead of failing."""
    import inspect
    import models.sdt as sdt

    src = inspect.getsource(sdt)
    assert "gate.unsqueeze(1)" not in src, "gate is already broadcast-ready"
    assert "gate_ff.unsqueeze(1)" not in src, "gate is already broadcast-ready"

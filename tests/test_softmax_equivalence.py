"""Is a 2-class softmax CE a genuinely different objective from sigmoid/BCE?

Under the current parameterization: NO, and this file is the proof, so the arm
is not run.

The model emits ONE logit per bit (cfg.model.out_dim = 1, data.vocab_size = 2).
A 2-class softmax over (z0, z1) is shift-invariant -- softmax(z0+c, z1+c) is
unchanged -- so it has exactly one degree of freedom, the difference d = z1-z0,
and

    softmax(z0,z1)[1] = e^{z1}/(e^{z0}+e^{z1}) = 1/(1+e^{-(z1-z0)}) = sigmoid(d)

Cross-entropy against a one-hot target is then identical to
BCEWithLogits(d, y). The families are the same; only the redundant
parameterization differs.
"""
import torch
import torch.nn.functional as F


def test_two_class_softmax_probability_equals_sigmoid_of_the_difference():
    torch.manual_seed(0)
    z = torch.randn(5000, 2, dtype=torch.float64)
    assert torch.allclose(torch.softmax(z, dim=-1)[:, 1],
                          torch.sigmoid(z[:, 1] - z[:, 0]), atol=1e-12)


def test_two_class_softmax_CE_equals_BCEWithLogits():
    torch.manual_seed(0)
    z = torch.randn(5000, 2, dtype=torch.float64)
    y = (torch.rand(5000) > 0.5).long()
    ce = F.cross_entropy(z, y, reduction="none")
    bce = F.binary_cross_entropy_with_logits(
        z[:, 1] - z[:, 0], y.to(torch.float64), reduction="none")
    assert torch.allclose(ce, bce, atol=1e-12)


def test_softmax_is_shift_invariant_so_the_extra_logit_adds_no_expressivity():
    torch.manual_seed(0)
    z = torch.randn(1000, 2, dtype=torch.float64)
    c = torch.randn(1000, 1, dtype=torch.float64)
    assert torch.allclose(torch.softmax(z, -1), torch.softmax(z + c, -1), atol=1e-12)


def test_gradients_wrt_the_effective_logit_coincide():
    """The 2-class head splits the same gradient across two logits: dL/dz1 =
    -dL/dz0 = dL/dd. Same update direction in the one direction that matters."""
    torch.manual_seed(0)
    z = torch.randn(64, 2, dtype=torch.float64, requires_grad=True)
    y = (torch.rand(64) > 0.5).long()
    F.cross_entropy(z, y, reduction="sum").backward()
    gz = z.grad.clone()
    d = (z.detach()[:, 1] - z.detach()[:, 0]).requires_grad_(True)
    F.binary_cross_entropy_with_logits(d, y.to(torch.float64), reduction="sum").backward()
    assert torch.allclose(gz[:, 1], d.grad, atol=1e-12)
    assert torch.allclose(gz[:, 0], -d.grad, atol=1e-12)


def test_the_repo_really_uses_the_one_logit_parameterization():
    """If this ever changes, the equivalence argument above must be revisited."""
    import importlib.util
    s = importlib.util.spec_from_file_location("c", "configs/tasks/tinygsm_bits_cfg.py")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    cfg = m.get_config()
    assert int(cfg.model.out_dim) == 1
    assert int(cfg.data.vocab_size) == 2
    assert str(cfg.data.representation) == "binary"

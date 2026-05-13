import torch

from src.models import FunctionalCausalModel
from src.utils import set_global_seed


def test_forward_shape_and_range():
    set_global_seed(0)
    B, D, L, d_p, d_m, H = 2, 6, 5, 4, 64, 32
    fcm = FunctionalCausalModel(D=D, L=L, d_p=d_p, d_m=d_m, hidden=H)
    P = torch.randn(B, D, L, d_p)
    C = torch.randn(B, D, L, 10, d_m)
    G = torch.rand(L, D, D)
    G_hat = torch.randn(L, D, D)
    y = fcm(P, C, G, G_hat)
    assert y.shape == (B, D)
    assert ((y > 0) & (y < 1)).all()  # Sigmoid open interval


def test_sparsity_respected():
    """If G[:, j, i] is all zero, then ∂f_i/∂P_{:,j,:} should also be zero."""
    set_global_seed(0)
    B, D, L, d_p, d_m, H = 1, 4, 3, 4, 16, 8
    fcm = FunctionalCausalModel(D=D, L=L, d_p=d_p, d_m=d_m, hidden=H)
    P = torch.randn(B, D, L, d_p, requires_grad=True)
    C = torch.randn(B, D, L, 10, d_m)
    G = torch.ones(L, D, D)
    G[:, 1, :] = 0  # source j=1 has no outgoing edges
    G_hat = torch.ones(L, D, D)
    y = fcm(P, C, G, G_hat)
    y.sum().backward()
    grad_j1 = P.grad[0, 1].abs().sum().item()
    grad_j0 = P.grad[0, 0].abs().sum().item()
    assert grad_j1 == 0.0
    assert grad_j0 > 0.0


def test_use_news_false_skips_psi():
    fcm = FunctionalCausalModel(D=4, L=3, d_p=4, d_m=64, hidden=16, use_news=False)
    assert fcm.psi_net is None
    P = torch.randn(2, 4, 3, 4)
    G = torch.rand(3, 4, 4)
    G_hat = torch.randn(3, 4, 4)
    y = fcm(P, None, G, G_hat)
    assert y.shape == (2, 4)


def test_softplus_noise_positive():
    fcm = FunctionalCausalModel(D=4, L=3, d_p=4, d_m=64, hidden=16)
    assert (fcm.sigma_noise > 0).all()


def test_per_node_head_independence():
    """Gradient of ζ_0's weights should NOT depend on output i=1."""
    fcm = FunctionalCausalModel(D=3, L=2, d_p=4, d_m=8, hidden=8)
    P = torch.randn(1, 3, 2, 4)
    C = torch.randn(1, 3, 2, 10, 8)
    G = torch.ones(2, 3, 3)
    G_hat = torch.ones(2, 3, 3)
    y = fcm(P, C, G, G_hat)
    # Backprop only y[:, 1] (target stock i=1).
    fcm.zero_grad()
    y[:, 1].sum().backward(retain_graph=True)
    g0 = next(fcm.zeta[0].parameters()).grad
    g1 = next(fcm.zeta[1].parameters()).grad
    assert g0 is None or g0.abs().sum() == 0
    assert g1 is not None and g1.abs().sum() > 0

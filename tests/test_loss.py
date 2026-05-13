import math

import torch

from src.training import CausalStockLoss
from src.utils import set_global_seed


def _toy_inputs(B=2, D=4, L=3, seed=0):
    set_global_seed(seed)
    f = (torch.rand(B, D) * 0.8 + 0.1).detach().requires_grad_(True)
    y = torch.randint(0, 2, (B, D))
    sigma_q = (torch.rand(L, D, D) * 0.5 + 0.25).detach().requires_grad_(True)
    G_sample = torch.rand(L, D, D).detach().requires_grad_(True)
    sigma_noise = torch.tensor([1.0] * D, requires_grad=True)
    return f, y, sigma_q, G_sample, sigma_noise


def test_loss_components_finite():
    f, y, sq, Gs, sn = _toy_inputs()
    loss = CausalStockLoss(bce_weight=0.01)
    out = loss(f, y, sq, Gs, sn)
    for name in ["total", "neg_elbo", "bce", "likelihood", "prior", "entropy"]:
        v = getattr(out, name)
        assert torch.isfinite(v).all(), f"{name} is non-finite: {v}"


def test_entropy_max_at_half():
    """H(Bernoulli) is maximized at p=0.5, value = log 2 per edge."""
    loss = CausalStockLoss()
    sq = torch.full((1, 2, 2), 0.5)
    h = loss._entropy(sq).item()
    assert math.isclose(h, 4 * math.log(2), abs_tol=1e-5)


def test_prior_zero_at_zero_graph():
    """log p(G) = -λ_s ||G||² → 0 when G=0."""
    loss = CausalStockLoss(lambda_s=1.0, lambda_d=0.0)
    p = loss._log_graph_prior(torch.zeros(3, 4, 4)).item()
    assert p == 0.0


def test_bce_zero_at_perfect_prediction():
    loss = CausalStockLoss()
    f = torch.tensor([[0.999, 0.001]])
    y = torch.tensor([[1, 0]])
    bce = loss._bce(f, y).item()
    assert bce < 1e-2


def test_gradient_flows_to_all_inputs():
    f, y, sq, Gs, sn = _toy_inputs()
    loss = CausalStockLoss(bce_weight=0.01)
    out = loss(f, y, sq, Gs, sn)
    out.total.backward()
    for name, t in [("f", f), ("sigma_q", sq), ("G_sample", Gs), ("sigma_noise", sn)]:
        assert t.grad is not None and t.grad.abs().sum() > 0, f"no grad in {name}"


def test_loss_decreases_on_toy_overfit():
    """Single toy sample: optimizer should drive total loss down."""
    set_global_seed(0)
    B, D, L = 1, 3, 2
    f_logits = torch.zeros(B, D, requires_grad=True)
    y = torch.tensor([[1, 0, 1]])
    sq = torch.full((L, D, D), 0.5, requires_grad=True)
    Gs = torch.full((L, D, D), 0.5, requires_grad=True)
    sn = torch.ones(D, requires_grad=True)
    loss = CausalStockLoss(bce_weight=0.01)
    opt = torch.optim.Adam([f_logits, sq, Gs, sn], lr=1e-1)
    losses = []
    for _ in range(80):
        f = torch.sigmoid(f_logits)
        out = loss(f, y, sq.clamp(1e-3, 1 - 1e-3), Gs.clamp(0, 1), sn.clamp_min(1e-2))
        opt.zero_grad()
        out.total.backward()
        opt.step()
        losses.append(out.total.item())
    assert losses[-1] < losses[0] - 0.05


def test_bernoulli_likelihood_form():
    f, y, sq, Gs, sn = _toy_inputs()
    loss = CausalStockLoss(bce_weight=0.0, likelihood_form="bernoulli")
    out = loss(f, y, sq, Gs, sn)
    # With bce_weight=0 and bernoulli likelihood, total ≈ -elbo / D and finite.
    assert torch.isfinite(out.total)

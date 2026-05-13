import torch

from src.models import LagDependentTCD
from src.utils import set_global_seed


def test_shapes_and_ranges():
    set_global_seed(0)
    tcd = LagDependentTCD(D=10, L=5, h_layers=1, lag_dep=True)
    tcd.train()
    out = tcd()
    assert out.sigma.shape == (5, 10, 10)
    assert out.G_sample.shape == (5, 10, 10)
    assert out.G_hat.shape == (5, 10, 10)
    assert ((out.sigma >= 0) & (out.sigma <= 1)).all()
    assert ((out.G_sample >= 0) & (out.G_sample <= 1)).all()


def test_gumbel_sample_is_differentiable():
    tcd = LagDependentTCD(D=4, L=3)
    tcd.train()
    out = tcd()
    out.G_sample.sum().backward()
    # All lag-dep MLPs and U/V should get gradients.
    assert tcd.U.grad is not None and tcd.U.grad.abs().sum() > 0
    assert tcd.V.grad is not None and tcd.V.grad.abs().sum() > 0


def test_eval_uses_hard_threshold():
    tcd = LagDependentTCD(D=4, L=3)
    tcd.eval()
    out = tcd()
    assert ((out.G_sample == 0) | (out.G_sample == 1)).all()


def test_lag_dep_off_skips_mlps():
    tcd = LagDependentTCD(D=4, L=3, lag_dep=False)
    assert not hasattr(tcd, "h_u")
    out = tcd()
    assert out.sigma.shape == (3, 4, 4)


def test_lag_boundary_at_l_eq_1_no_nan():
    """Eq. 6: u_{l-1} for l=1 is padded with zeros; should produce finite σ."""
    tcd = LagDependentTCD(D=4, L=5, lag_dep=True)
    tcd.train()
    out = tcd()
    assert torch.isfinite(out.sigma).all()
    assert torch.isfinite(out.G_sample).all()


def test_init_keeps_sigma_near_half():
    """Xavier init of U≈V → σ ≈ 0.5. Reasonable starting prior."""
    set_global_seed(0)
    tcd = LagDependentTCD(D=8, L=5, lag_dep=False)  # bypass h_u/h_v to test U,V directly
    out = tcd()
    mean = out.sigma.mean().item()
    assert 0.45 < mean < 0.55

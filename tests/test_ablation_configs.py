"""Verify that ablation flags actually flip the right knobs in CausalStockModel."""
from pathlib import Path

import torch

from src.models import CausalStockModel
from src.training import CausalStockLoss
from src.utils import load_yaml_config

ROOT = Path(__file__).resolve().parents[1]


def _model_from(cfg, D=4, F=6):
    return CausalStockModel(
        D=D, L=int(cfg.data.lag_L), price_in_dim=F,
        d_p=int(cfg.model.d_p), d_m=int(cfg.model.d_m), hidden=16,
        use_tcd=bool(cfg.model.use_tcd),
        use_news=bool(cfg.data.use_news),
        lag_dep=bool(cfg.model.lag_dep),
        h_uv_layers=int(cfg.model.h_uv_layers),
    )


def test_no_tcd_uses_constant_graph():
    cfg = load_yaml_config("experiments/configs/ablations/no_tcd.yaml")
    m = _model_from(cfg)
    assert m.use_tcd is False
    P = torch.randn(1, 4, 5, 6)
    C = torch.zeros(1, 4, 5, 10, 5)
    out = m(P, C)
    assert (out.G_sample == 1).all()


def test_no_news_skips_psi():
    cfg = load_yaml_config("experiments/configs/ablations/no_news.yaml")
    m = _model_from(cfg)
    assert m.fcm.psi_net is None
    assert m.use_news is False


def test_no_lag_dep_drops_h_mlps():
    cfg = load_yaml_config("experiments/configs/ablations/no_lag_dep.yaml")
    m = _model_from(cfg)
    assert not hasattr(m.tcd, "h_u")
    assert not hasattr(m.tcd, "h_v")


def test_lambda_0_zeroes_bce_term():
    cfg = load_yaml_config("experiments/configs/ablations/lambda_0.yaml")
    loss = CausalStockLoss(bce_weight=float(cfg.loss.bce_weight))
    assert loss.bce_weight == 0.0

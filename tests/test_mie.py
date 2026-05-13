import torch

from src.models import MarketInformationEncoder, NewsScoreEmbedding, PriceEncoder
from src.utils import set_global_seed


def test_price_encoder_shape():
    set_global_seed(0)
    enc = PriceEncoder(in_dim=6, d_p=4)
    P = torch.randn(2, 85, 5, 6)  # (B, D, L, F)
    out = enc(P)
    assert out.shape == (2, 85, 5, 4)
    out.sum().backward()  # gradient flows


def test_news_embed_shape_and_normalize():
    enc = NewsScoreEmbedding(d_m=64, normalize=True)
    raw = torch.tensor([[10.0, 1.0, 10.0, 10.0, 10.0]])  # max values
    # After normalize → [1, 1, 1, 1, 1]
    out = enc(raw)
    assert out.shape == (1, 64)
    # Sanity: zero scores → linear(0) = bias = 0 → output 0
    z = enc(torch.zeros(1, 5))
    assert torch.allclose(z, torch.zeros(1, 64))


def test_mie_combined():
    mie = MarketInformationEncoder(price_in_dim=6, d_p=4, d_m=64)
    P = torch.randn(2, 85, 5, 6)
    C = torch.rand(2, 85, 5, 10, 5) * 10  # roughly paper range
    P_emb, C_emb = mie(P, C)
    assert P_emb.shape == (2, 85, 5, 4)
    assert C_emb.shape == (2, 85, 5, 10, 64)
    # Gradient flows through both branches
    (P_emb.sum() + C_emb.sum()).backward()
    for p in mie.parameters():
        assert p.grad is not None


def test_news_embed_no_normalize():
    enc = NewsScoreEmbedding(d_m=8, normalize=False)
    raw = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])
    out = enc(raw)
    assert out.shape == (1, 8)

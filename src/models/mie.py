"""Market Information Encoder (Section 3 of docs/paper-summary.md).

    Price Encoder    : R^F (F=6 for ACL18, 11 for DTML datasets) → R^{d_p=4}
    News Score Embed : R^5 (Correlation, Sentiment, Importance, Impact,
                              Duration; see Table in §3.2) → R^{d_m=64}

Appendix C.4: d_p=4, d_m=64. Linear projection assumed (paper text only says
"Embedding layer"; see docs/reproduction-questions.md B.3).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PriceEncoder(nn.Module):
    def __init__(self, in_dim: int, d_p: int = 4):
        super().__init__()
        self.linear = nn.Linear(in_dim, d_p)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, P: torch.Tensor) -> torch.Tensor:
        # P: (..., F) -> (..., d_p)
        return self.linear(P)


class NewsScoreEmbedding(nn.Module):
    """5-dim GPT scores → d_m embedding.

    Score ranges (paper §3.2): Correlation, Importance, Impact, Duration ∈ [0,10];
    Sentiment ∈ [-1,1]. We rescale to roughly unit magnitude before the linear
    projection so the optimizer isn't biased by raw scale (docs/reproduction-
    questions.md B.5).
    """

    SCALE = torch.tensor([10.0, 1.0, 10.0, 10.0, 10.0])  # divisors per dim
    SHIFT = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])

    def __init__(self, d_m: int = 64, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        self.linear = nn.Linear(5, d_m)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        if normalize:
            self.register_buffer("scale", self.SCALE.clone())
            self.register_buffer("shift", self.SHIFT.clone())

    def forward(self, C_score: torch.Tensor) -> torch.Tensor:
        # C_score: (..., 5)
        if self.normalize:
            C_score = (C_score - self.shift) / self.scale
        return self.linear(C_score)


class MarketInformationEncoder(nn.Module):
    """Wraps both encoders. Returns embedded (P, C)."""

    def __init__(self, price_in_dim: int, d_p: int = 4, d_m: int = 64, normalize_news: bool = True):
        super().__init__()
        self.price_encoder = PriceEncoder(price_in_dim, d_p=d_p)
        self.news_embedding = NewsScoreEmbedding(d_m=d_m, normalize=normalize_news)
        self.d_p = d_p
        self.d_m = d_m

    def forward(self, P: torch.Tensor, C_score: torch.Tensor):
        """P: (B,D,L,F), C_score: (B,D,L,l,5) → (B,D,L,d_p), (B,D,L,l,d_m)."""
        return self.price_encoder(P), self.news_embedding(C_score)

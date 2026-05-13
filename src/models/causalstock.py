"""Top-level CausalStock model = MIE + TCD + FCM.

Integrates the three modules per Figure 2 (docs/paper-summary.md §2):
    [P, C_score] --MIE--> [P_emb, C_emb]
    (price only)  --TCD--> (G, σ, Ĝ)            # causal-stationary detach
    [P_emb, C_emb, G, Ĝ] --FCM--> f_i ∈ (0,1)

Forward returns f_i, the variational posterior σ, and the Gumbel sample G,
which are then consumed by `CausalStockLoss`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .fcm import FunctionalCausalModel
from .mie import MarketInformationEncoder
from .tcd import LagDependentTCD


@dataclass
class ModelOutput:
    f_i: torch.Tensor       # (B, D)
    sigma: torch.Tensor     # (L, D, D)
    G_sample: torch.Tensor  # (L, D, D)
    G_hat: torch.Tensor     # (L, D, D)


class CausalStockModel(nn.Module):
    def __init__(
        self,
        D: int,
        L: int = 5,
        price_in_dim: int = 6,
        d_p: int = 4,
        d_m: int = 64,
        hidden: int = 332,
        use_tcd: bool = True,
        use_news: bool = True,
        lag_dep: bool = True,
        h_uv_layers: int = 1,
        gumbel_tau: float = 1.0,
        normalize_news: bool = True,
    ):
        super().__init__()
        self.D, self.L = D, L
        self.use_tcd = use_tcd
        self.use_news = use_news
        self.mie = MarketInformationEncoder(
            price_in_dim=price_in_dim, d_p=d_p, d_m=d_m, normalize_news=normalize_news
        )
        self.tcd = LagDependentTCD(D=D, L=L, h_layers=h_uv_layers, gumbel_tau=gumbel_tau, lag_dep=lag_dep)
        self.fcm = FunctionalCausalModel(D=D, L=L, d_p=d_p, d_m=d_m, hidden=hidden, use_news=use_news)

    def forward(self, P: torch.Tensor, C_score: torch.Tensor) -> ModelOutput:
        # Section 6.3 — Causal Stationary: news C does not flow to TCD.
        # In this architecture TCD has no input path from C anyway, so the
        # detach is implicit. Recorded for future amortized-TCD variants.
        P_emb, C_emb = self.mie(P, C_score)
        if self.use_news:
            C_for_fcm = C_emb
        else:
            C_for_fcm = None

        if self.use_tcd:
            tcd_out = self.tcd()
            G = tcd_out.G_sample
            sigma = tcd_out.sigma
            G_hat = tcd_out.G_hat
        else:
            # Ablation: w/o TCD → G fixed to ones, weight to ones (paper Table 2)
            G = torch.ones(self.L, self.D, self.D, device=P.device, dtype=P.dtype)
            sigma = torch.full(
                (self.L, self.D, self.D), 0.5, device=P.device, dtype=P.dtype
            )
            G_hat = torch.ones(self.L, self.D, self.D, device=P.device, dtype=P.dtype)

        f_i = self.fcm(P_emb, C_for_fcm, G, G_hat)
        return ModelOutput(f_i=f_i, sigma=sigma, G_sample=G, G_hat=G_hat)

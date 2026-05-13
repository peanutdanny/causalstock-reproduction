"""Lag-dependent Temporal Causal Discovery (Section 4 of docs/paper-summary.md).

We learn per-lag, per-edge Bernoulli posteriors over the temporal causal graph
G ∈ {0,1}^{L×D×D}, with the *lag-dependent* factorization (Eq. 3):

    q_φ(G) = q_φ(G_1) · ∏_{l=2..L} q_φ(G_l | G_{l-1})

Each edge has two learnable likelihood scalars (existence u, non-existence v)
stored as parameters U, V ∈ R^{L×D×D}. Edge probability σ_{l,ji} follows Eq. 7
after passing through lag-dependency MLPs h_u, h_v (Eq. 6).

NOTE on q_φ(G | X) input-dependence: paper-summary writes the posterior as
conditional on X_{<T}, but the body only treats U, V as free parameters. We
follow the body and make U, V input-independent for now.
TODO(paper-ambiguity): docs/reproduction-questions.md C.7 (new).

Causal Weight Graph Ĝ ∈ R^{L×D×D} (Section 4.5): real-valued learnable tensor.
Causal strength = G ⊙ Ĝ.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TCDOutput:
    G_sample: torch.Tensor   # (L, D, D) — Gumbel-Softmax sample (or hard at eval)
    sigma:    torch.Tensor   # (L, D, D) — Bernoulli probabilities
    G_hat:    torch.Tensor   # (L, D, D) — causal weight (real-valued)


def _build_h_mlp(n_layers: int, hidden: int = 16) -> nn.Module:
    """h_u and h_v: map (u_{l,ji}, u_{l-1,ji}) ∈ R^2 → R^1.

    Appendix C.4 says 1-layer, body §4.3 says 3-layer (docs/reproduction-
    questions.md C.1). Default 1, configurable.
    """
    if n_layers <= 1:
        return nn.Linear(2, 1)
    layers = [nn.Linear(2, hidden), nn.ReLU()]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers += [nn.Linear(hidden, 1)]
    return nn.Sequential(*layers)


class LagDependentTCD(nn.Module):
    def __init__(
        self,
        D: int,
        L: int,
        h_layers: int = 1,
        gumbel_tau: float = 1.0,
        lag_dep: bool = True,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.D, self.L = D, L
        self.tau = gumbel_tau
        self.lag_dep = lag_dep
        # Edge existence / non-existence likelihoods (Section 4.4).
        self.U = nn.Parameter(torch.empty(L, D, D))
        self.V = nn.Parameter(torch.empty(L, D, D))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        with torch.no_grad():
            self.U.mul_(init_scale)
            self.V.mul_(init_scale)
        if lag_dep:
            self.h_u = _build_h_mlp(h_layers)
            self.h_v = _build_h_mlp(h_layers)
        # Causal Weight Graph (Section 4.5).
        self.G_hat = nn.Parameter(torch.empty(L, D, D))
        nn.init.xavier_uniform_(self.G_hat)

    @staticmethod
    def _shift_prev(t: torch.Tensor) -> torch.Tensor:
        """For each lag l, return the (l-1) slice padded with zeros at l=0."""
        prev = torch.zeros_like(t)
        prev[1:] = t[:-1]
        return prev

    def _apply_lag_dep(self, raw: torch.Tensor, mlp: nn.Module) -> torch.Tensor:
        """Eq. 6 applied elementwise to (L,D,D) parameter tensor."""
        prev = self._shift_prev(raw)
        pair = torch.stack([raw, prev], dim=-1)  # (L, D, D, 2)
        return mlp(pair).squeeze(-1)             # (L, D, D)

    def _edge_logits(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.lag_dep:
            u = self._apply_lag_dep(self.U, self.h_u)
            v = self._apply_lag_dep(self.V, self.h_v)
        else:
            u, v = self.U, self.V
        return u, v

    def _sigma(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Eq. 7: σ = softmax([u, v])[0] = exp(u) / (exp(u) + exp(v))
        return torch.sigmoid(u - v)

    def sample_gumbel(self, hard: bool = False) -> torch.Tensor:
        """Gumbel-Softmax sample of G ∈ [0,1]^{L,D,D}."""
        u, v = self._edge_logits()
        logits = torch.stack([u, v], dim=-1)  # (L, D, D, 2)
        sample = F.gumbel_softmax(logits, tau=self.tau, hard=hard, dim=-1)
        return sample[..., 0]                  # take existence channel

    def forward(self, hard: bool = False) -> TCDOutput:
        u, v = self._edge_logits()
        sigma = self._sigma(u, v)
        G = self.sample_gumbel(hard=hard) if self.training else (sigma > 0.5).float()
        return TCDOutput(G_sample=G, sigma=sigma, G_hat=self.G_hat)

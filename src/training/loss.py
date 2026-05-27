"""ELBO + BCE objective (Section 6 of docs/paper-summary.md).

Eq. 13 (ELBO):
    log p_θ(y_T | X_{<T}) ≥ E_q[log p(y_T|G,X)] + E_q[log p(G)] + H(q_φ(G))

Eq. 14 (full loss, minimized):
    L = (1/D) · ( -ELBO + λ · BCE(g_T, f_i) )

Open interpretation issues (see docs/reproduction-questions.md D.4–D.5):
- The paper writes y_T as continuous in Eq. 9 (f_i + z) but the labels are
  binary {0,1}. Two consistent reductions:
    (a) "gaussian": z = g_T - f_i, then log p(z) = log N(z; 0, σ²).
    (b) "bernoulli": treat f_i as p(y=1) and use BCE form for the likelihood.
  Default = "gaussian" (paper-literal Eq. 11–12); BCE auxiliary already covers (b).

- Graph prior Eq. 4 normalizing constant Z is dropped because λ_s, λ_d are
  fixed (constant in G, drops out of ELBO gradient).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossOutput:
    total: torch.Tensor
    neg_elbo: torch.Tensor
    bce: torch.Tensor
    likelihood: torch.Tensor
    prior: torch.Tensor
    entropy: torch.Tensor


class CausalStockLoss(nn.Module):
    """All-in-one objective combining ELBO + BCE per Eq. 14."""

    EPS = 1e-7

    def __init__(
        self,
        bce_weight: float = 0.01,
        lambda_s: float = 1.0,
        lambda_d: float = 0.0,
        G_prior: Optional[torch.Tensor] = None,
        likelihood_form: Literal["gaussian", "bernoulli"] = "gaussian",
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.lambda_s = lambda_s
        self.lambda_d = lambda_d
        self.likelihood_form = likelihood_form
        # KL warmup: scales the (prior + entropy) terms. Trainer can mutate this
        # per epoch via set_kl_weight(). Default 1.0 = paper-faithful full ELBO.
        # Setting to <1 prevents the posterior collapse where σ_q stays at 0.5
        # and the entropy bonus dominates the likelihood signal (observed in
        # 4/10 seeds during Phase 10b GPU sweep, 2026-05-27).
        self.kl_weight: float = 1.0
        if G_prior is not None:
            self.register_buffer("G_prior", G_prior)
        else:
            self.G_prior = None

    def set_kl_weight(self, beta: float) -> None:
        """Set the (prior + entropy) scale factor used in the next forward()."""
        self.kl_weight = float(beta)

    # ---- ELBO components -----------------------------------------------------

    def _log_likelihood(
        self, f_i: torch.Tensor, y: torch.Tensor, sigma_noise: torch.Tensor
    ) -> torch.Tensor:
        """E_q[log p(y|G,X)] for one Gumbel sample (Eq. 11–12).

        f_i : (B, D) ∈ (0, 1)
        y   : (B, D) binary
        sigma_noise : (D,) > 0
        """
        if self.likelihood_form == "gaussian":
            # z = y - f_i (paper-literal additive noise residual).
            z = y.float() - f_i
            var = (sigma_noise ** 2).clamp_min(self.EPS).unsqueeze(0)  # (1, D)
            log_pdf = -0.5 * (z ** 2 / var) - 0.5 * torch.log(2 * math.pi * var)
            return log_pdf.sum(dim=-1).mean()  # mean over batch, sum over D
        # Bernoulli interpretation: log p(y|f) = y log f + (1-y) log(1-f)
        f = f_i.clamp(self.EPS, 1 - self.EPS)
        ll = y.float() * torch.log(f) + (1 - y.float()) * torch.log(1 - f)
        return ll.sum(dim=-1).mean()

    def _log_graph_prior(self, G_sample: torch.Tensor) -> torch.Tensor:
        """E_q[log p(G)] via single sample (Eq. 4)."""
        sparseness = -self.lambda_s * (G_sample ** 2).sum()
        if self.lambda_d > 0 and self.G_prior is not None:
            domain = -self.lambda_d * ((G_sample - self.G_prior) ** 2).sum()
        else:
            domain = torch.zeros((), device=G_sample.device)
        return sparseness + domain

    def _entropy(self, sigma_q: torch.Tensor) -> torch.Tensor:
        """H(q_φ(G)) for independent Bernoullis."""
        p = sigma_q.clamp(self.EPS, 1 - self.EPS)
        return -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).sum()

    # ---- BCE auxiliary -------------------------------------------------------

    def _bce(self, f_i: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(
            f_i.clamp(self.EPS, 1 - self.EPS),
            y.float(),
            reduction="mean",  # mean over (B, D)
        )

    # ---- Top-level -----------------------------------------------------------

    def forward(
        self,
        f_i: torch.Tensor,           # (B, D)
        y: torch.Tensor,             # (B, D) long {0,1}
        sigma_q: torch.Tensor,       # (L, D, D)
        G_sample: torch.Tensor,      # (L, D, D)
        sigma_noise: torch.Tensor,   # (D,) > 0
    ) -> LossOutput:
        D = f_i.shape[-1]
        ll = self._log_likelihood(f_i, y, sigma_noise)
        prior = self._log_graph_prior(G_sample) / max(f_i.shape[0], 1)  # per-batch scale
        entropy = self._entropy(sigma_q) / max(f_i.shape[0], 1)
        # KL warmup scales the prior+entropy terms. ll always gets full weight
        # so the model has a clear gradient toward fitting the data first.
        elbo = ll + self.kl_weight * (prior + entropy)
        bce = self._bce(f_i, y)
        # Eq. 14 — 1/D normalization.
        total = (-elbo + self.bce_weight * bce * D) / D
        return LossOutput(
            total=total,
            neg_elbo=-elbo,
            bce=bce,
            likelihood=ll,
            prior=prior,
            entropy=entropy,
        )

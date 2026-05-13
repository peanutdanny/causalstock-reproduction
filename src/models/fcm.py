"""Functional Causal Model (Section 5 of docs/paper-summary.md).

Eq. 10:
    f_i(Pa^i_G(<T)) = Sigmoid( ζ_i( Σ_{l,j} G_{l,ji} · Ĝ_{l,ji}
                                    · [ℓ(P_{T-l}^j), ψ(C̄_{T-l}^j)] ) )

where:
    ℓ      : R^{d_p}             → R^{hidden=332}   shared across (j, l)
    ψ      : R^{d_m}             → R^{hidden=332}   shared, mean-pooled over l-news
    ζ_i    : R^{2·hidden}        → R^1              per-stock (D heads)
    G,Ĝ   : (L, D, D)            from TCD module

Notes
-----
- ψ takes C̄ = mean over the l news embeddings for the day (paper text doesn't
  specify reduction — docs/reproduction-questions.md B.4 default = mean).
- Causal stationary detach: paper Appendix B says C must not flow gradients to G.
  Here the FCM consumes G via the per-edge mask but G doesn't depend on C in the
  forward path, so detach is moot in this stack. Will revisit when TCD becomes
  amortized.
- Eq. 10 typo `h_ℓ` vs `ℓ` (docs/paper-summary line 169 vs 172) — treat as same.
- Output is Sigmoid → probability ∈ (0,1). The "+z_T^i additive noise" of Eq. 9
  is interpreted as a residual term used in the ELBO likelihood (Phase 6); we
  do NOT add noise to the predicted probability at forward time.
  TODO(paper-ambiguity): docs/reproduction-questions.md D.4.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(in_dim: int, out_dim: int, hidden: int, n_layers: int = 3) -> nn.Module:
    """3-layer MLP (input + 1 hidden + output), ReLU activations, Xavier init."""
    layers = []
    last = in_dim
    for k in range(n_layers - 1):
        layers += [nn.Linear(last, hidden), nn.ReLU()]
        last = hidden
    layers += [nn.Linear(last, out_dim)]
    mlp = nn.Sequential(*layers)
    for m in mlp:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return mlp


class FunctionalCausalModel(nn.Module):
    """Implements Eq. 10 with shared ℓ, ψ and per-node ζ_i."""

    def __init__(
        self,
        D: int,
        L: int,
        d_p: int = 4,
        d_m: int = 64,
        hidden: int = 332,
        n_layers: int = 3,
        use_news: bool = True,
    ):
        super().__init__()
        self.D, self.L = D, L
        self.use_news = use_news
        self.hidden = hidden
        self.l_net = _build_mlp(d_p, hidden, hidden, n_layers=n_layers)
        if use_news:
            self.psi_net = _build_mlp(d_m, hidden, hidden, n_layers=n_layers)
            zeta_in = 2 * hidden
        else:
            self.psi_net = None
            zeta_in = hidden
        # Per-stock head ζ_i: D parallel 3-layer MLPs.
        self.zeta = nn.ModuleList([_build_mlp(zeta_in, 1, hidden, n_layers=n_layers) for _ in range(D)])
        # Additive-noise scale (Eq. 9). Softplus-parameterized for positivity
        # (docs/reproduction-questions.md D.3).
        self.log_sigma = nn.Parameter(torch.zeros(D))

    @property
    def sigma_noise(self) -> torch.Tensor:
        return F.softplus(self.log_sigma)  # (D,)

    def forward(
        self,
        P_emb: torch.Tensor,    # (B, D, L, d_p)
        C_emb: Optional[torch.Tensor],  # (B, D, L, l, d_m) or None
        G: torch.Tensor,        # (L, D, D) — sample
        G_hat: torch.Tensor,    # (L, D, D) — weight
    ) -> torch.Tensor:
        """Return f_i(...) ∈ (0,1), shape (B, D)."""
        B = P_emb.shape[0]
        D, L, hidden = self.D, self.L, self.hidden

        # Shared per-(j, l) features.
        h_l = self.l_net(P_emb)                                # (B, D, L, H)
        if self.use_news and C_emb is not None and self.psi_net is not None:
            C_bar = C_emb.mean(dim=-2)                         # (B, D, L, d_m)
            h_psi = self.psi_net(C_bar)                        # (B, D, L, H)
            features = torch.cat([h_l, h_psi], dim=-1)         # (B, D, L, 2H)
        else:
            features = h_l                                     # (B, D, L, H)
        feat_dim = features.shape[-1]

        # Aggregate parents per target node i: sum_{l,j} G[l,j,i] · Ĝ[l,j,i] · features[B,j,l,:]
        # Mask = G ⊙ Ĝ → (L, D_src, D_tgt)
        mask = G * G_hat
        # Reshape features → (B, D_src=D, L, feat_dim); we want (B, D_tgt=D, feat_dim) result.
        # einsum: B j l F , l j i -> B i F
        agg = torch.einsum("bjlf,lji->bif", features, mask)    # (B, D, feat_dim)

        # Per-node ζ_i over the aggregated parents.
        logits = torch.empty(B, D, device=P_emb.device, dtype=P_emb.dtype)
        for i, head in enumerate(self.zeta):
            logits[:, i] = head(agg[:, i]).squeeze(-1)
        return torch.sigmoid(logits)

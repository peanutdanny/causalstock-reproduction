"""Causal graph visualizations — corresponds to paper Figure 3 + Table 5.

Inputs are trained model checkpoints (we extract σ, G, Ĝ from TCD module).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from ._style import apply_style


def _load_tcd(checkpoint_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load TCD parameters from a Trainer checkpoint.

    Returns
    -------
    sigma  : (L, D, D) Bernoulli probabilities
    G_hat  : (L, D, D) causal weight
    G_sample : (L, D, D) one Gumbel sample (eval mode = hard threshold)
    """
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.models import CausalStockModel

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # We don't know D, L from the checkpoint dict alone; infer from tensor shape.
    state = ckpt["model"]
    U = state["tcd.U"]
    L, D, _ = U.shape
    model = CausalStockModel(D=D, L=L, price_in_dim=6, d_p=4, d_m=64, hidden=332, h_uv_layers=1)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        out = model.tcd()
    return out.sigma.cpu().numpy(), out.G_hat.cpu().numpy(), out.G_sample.cpu().numpy()


def plot_sigma_heatmap(
    checkpoint_path: str | Path,
    out_path: str | Path,
    *,
    stock_labels: Optional[Sequence[str]] = None,
    title: str = "Causal edge probability σ_{l,j,i} per lag",
) -> Path:
    """Per-lag (L panels) heatmap of σ. Paper Figure 3a 대응."""
    apply_style()
    sigma, _, _ = _load_tcd(checkpoint_path)
    L, D, _ = sigma.shape

    cols = min(L, 5)
    rows = (L + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows + 0.5), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    for l in range(L):
        ax = axes[l]
        show_ticks = (D <= 30) and (stock_labels is not None)
        sns.heatmap(
            sigma[l], cmap="viridis", vmin=0, vmax=1, square=True, ax=ax,
            cbar=(l == L - 1),
            xticklabels=stock_labels if show_ticks else False,
            yticklabels=stock_labels if show_ticks else False,
        )
        ax.set_title(f"lag = {l + 1}")
    for l in range(L, len(axes)):
        axes[l].axis("off")
    fig.suptitle(title, fontsize=12)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_causal_strength(
    checkpoint_path: str | Path,
    out_path: str | Path,
    *,
    stock_labels: Optional[Sequence[str]] = None,
    title: str = "Causal strength  Σ_l G_l ⊙ Ĝ_l",
) -> Path:
    """Aggregated (D × D) causal strength. Paper Figure 3b 대응."""
    apply_style()
    sigma, G_hat, _ = _load_tcd(checkpoint_path)
    # Use σ as soft G (eval-mode hard sample available too).
    strength = (sigma * G_hat).sum(axis=0)
    D = strength.shape[0]

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    show_ticks = (D <= 30) and (stock_labels is not None)
    vmax = np.abs(strength).max() or 1.0
    sns.heatmap(
        strength, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax, square=True,
        xticklabels=stock_labels if show_ticks else False,
        yticklabels=stock_labels if show_ticks else False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Target stock i")
    ax.set_ylabel("Source stock j")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_market_cap_scatter(
    causal_strength_per_stock: np.ndarray,
    market_cap: np.ndarray,
    out_path: str | Path,
    *,
    stock_labels: Optional[Sequence[str]] = None,
    title: str = "Causal strength vs. market cap",
) -> Path:
    """Paper Table 5 visualization (Spearman corr)."""
    apply_style()
    from scipy.stats import spearmanr

    rho, p = spearmanr(causal_strength_per_stock, market_cap)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.scatter(market_cap, causal_strength_per_stock, alpha=0.6, s=30)
    ax.set_xscale("log")
    ax.set_xlabel("Market cap (USD, log scale)")
    ax.set_ylabel("Causal strength (Σ_j |G⊙Ĝ|_{ji})")
    ax.set_title(f"{title}  Spearman ρ = {rho:.3f}  (p = {p:.4f})")
    if stock_labels is not None and len(stock_labels) <= 30:
        for x, y, lbl in zip(market_cap, causal_strength_per_stock, stock_labels):
            ax.annotate(lbl, (x, y), fontsize=7, alpha=0.7)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out

"""Trading simulation visualizations — paper Figure 4 reproduction.

Inputs: arrays of (T, D) predictions + (T, D) realized returns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ._style import PALETTE, apply_style
from src.evaluation.trading import (
    accumulated_portfolio_value,
    sharpe_ratio,
    top_k_portfolio_returns,
)


def plot_apv_curve(
    runs: Dict[str, Dict[str, np.ndarray]],
    out_path: str | Path,
    *,
    k: int = 3,
    title: str = "Accumulated Portfolio Value (top-3 strategy)",
    paper_final_apv: float | None = None,
) -> Path:
    """Per-model APV over time.

    Args:
        runs: {label: {"probs": (T,D), "returns": (T,D)}}
        paper_final_apv: if given, draw a dashed horizontal at the paper's
            reported final APV for visual comparison (e.g. ACL18 = 1.32).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for label, d in runs.items():
        r_daily = top_k_portfolio_returns(d["probs"], d["returns"], k=k)
        apv = accumulated_portfolio_value(r_daily)
        color = PALETTE.get(label, None)
        ax.plot(apv, label=f"{label}  (final={apv[-1]:.3f})",
                color=color, linewidth=1.6)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, label="break-even")
    if paper_final_apv is not None:
        ax.axhline(paper_final_apv, color=PALETTE.get("paper", "#7f7f7f"),
                   linestyle="--", linewidth=1.0,
                   label=f"paper final APV = {paper_final_apv:.2f}")
    ax.set_xlabel("Trading day (test period)")
    ax.set_ylabel("APV   (= ∏(1 + r_t))")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_sharpe_bar(
    runs: Dict[str, Dict[str, np.ndarray]],
    out_path: str | Path,
    *,
    k: int = 3,
    risk_free: float = 0.0,
    title: str = "Sharpe ratio comparison",
    paper_sharpe: float | None = None,
) -> Path:
    """Bar chart of daily Sharpe ratio per model.

    Args:
        paper_sharpe: optional dashed reference line for the paper's reported
            value (e.g. ACL18 = 0.369).
    """
    apply_style()
    labels = list(runs.keys())
    srs = []
    for d in runs.values():
        r = top_k_portfolio_returns(d["probs"], d["returns"], k=k)
        srs.append(sharpe_ratio(r, risk_free=risk_free))
    colors = [PALETTE.get(lbl, "#7f7f7f") for lbl in labels]

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    bars = ax.bar(labels, srs, color=colors)
    for bar, sr in zip(bars, srs):
        ax.annotate(f"{sr:.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    if paper_sharpe is not None:
        ax.axhline(paper_sharpe, color=PALETTE.get("paper", "#7f7f7f"),
                   linestyle="--", linewidth=1.0,
                   label=f"paper SR = {paper_sharpe:.3f}")
        ax.legend(loc="best", frameon=False)
    ax.set_ylabel("Sharpe ratio (daily, R_f=0)")
    ax.set_title(title)
    ax.set_axisbelow(True)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out

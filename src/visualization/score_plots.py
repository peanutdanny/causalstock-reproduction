"""DNE 5-aspect score distribution / correlation visualizations.

Consumes `data/processed/dne_*.parquet` produced by Phase 3b scoring.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ._style import apply_style

DIMENSIONS = ["Correlation", "Sentiment", "Importance", "Impact", "Duration"]
PAPER_RANGES = {
    "Correlation": (0, 10),
    "Sentiment":   (-1, 1),
    "Importance":  (0, 10),
    "Impact":      (0, 10),
    "Duration":    (0, 10),
}


def _flat_scored(cache_path: str | Path) -> np.ndarray:
    """Return (n_scored_tweets, 5) array of nonzero score rows."""
    df = pd.read_parquet(cache_path)
    arr = np.array(df["scores"].tolist())
    arr = arr.reshape(len(df), -1, 5)
    mask = (arr.sum(axis=2) != 0)
    return arr[mask]


def plot_score_distribution(
    cache_path: str | Path,
    out_path: str | Path,
    *,
    title: str = "DNE GPT score distribution",
) -> Path:
    """5 histograms in one figure."""
    apply_style()
    scores = _flat_scored(cache_path)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3), constrained_layout=True)
    for i, (ax, name) in enumerate(zip(axes, DIMENSIONS)):
        lo, hi = PAPER_RANGES[name]
        bins = 21 if name == "Sentiment" else 11
        ax.hist(scores[:, i], bins=bins, range=(lo, hi), color="steelblue", edgecolor="white")
        mean = scores[:, i].mean()
        ax.axvline(mean, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{name}\n(μ={mean:+.2f})")
        ax.set_xlim(lo, hi)
        ax.set_xlabel("score")
    fig.suptitle(f"{title}  (n={len(scores):,} scored tweets)", fontsize=12)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_score_correlation(
    cache_path: str | Path,
    out_path: str | Path,
    *,
    title: str = "5-dim DNE score correlation",
) -> Path:
    """5x5 Pearson correlation heatmap."""
    apply_style()
    scores = _flat_scored(cache_path)
    corr = np.corrcoef(scores.T)
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    sns.heatmap(
        corr,
        annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        xticklabels=DIMENSIONS, yticklabels=DIMENSIONS, square=True, ax=ax,
    )
    ax.set_title(title)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_tweet_volume_heatmap(
    cache_path: str | Path,
    out_path: str | Path,
    *,
    title: str = "Tweets per (stock, day)",
) -> Path:
    """Heatmap: row=stock, col=date, value=#scored tweets that day."""
    apply_style()
    df = pd.read_parquet(cache_path)
    arr = np.array(df["scores"].tolist()).reshape(len(df), -1, 5)
    df = df.assign(n_scored=(arr.sum(axis=2) != 0).sum(axis=1))
    pivot = df.pivot_table(index="ticker", columns="date", values="n_scored", fill_value=0)
    # Truncate to a representative time slice if too wide
    if pivot.shape[1] > 400:
        pivot = pivot.iloc[:, :400]
    fig, ax = plt.subplots(figsize=(min(14, 0.05 * pivot.shape[1] + 5), 0.18 * pivot.shape[0] + 2),
                            constrained_layout=True)
    sns.heatmap(pivot, cmap="YlGnBu", cbar_kws={"label": "tweets/day"}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out

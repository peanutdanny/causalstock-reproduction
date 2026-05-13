"""Shared plotting style settings — paper-friendly look."""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


def apply_style() -> None:
    """Call once at the top of any plotting script."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "font.family": "sans-serif",
    })


PALETTE = {
    "full":       "#1f77b4",
    "no_tcd":     "#d62728",
    "no_news":    "#ff7f0e",
    "no_lag_dep": "#2ca02c",
    "lambda_0":   "#9467bd",
    "paper":      "#7f7f7f",
}

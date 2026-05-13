"""Reproduction-vs-paper comparison visualizations."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ._style import apply_style, PALETTE


def plot_reproduction_table(
    paper: Dict[str, float],
    ours: Dict[str, float],
    out_path: str | Path,
    *,
    tolerance: float = 0.5,   # percentage points for ACC
    title: str = "Paper vs. Reproduction",
) -> Path:
    """Heatmap-style comparison: each row is a (dataset, metric); color = within/outside tolerance.

    Args:
        paper: {"ACL18_ACC": 63.42, "ACL18_MCC": 0.2172, ...}
        ours:  same keys
    """
    apply_style()
    keys = sorted(set(paper) & set(ours))
    paper_vals = np.array([paper[k] for k in keys])
    ours_vals = np.array([ours[k] for k in keys])
    diff = ours_vals - paper_vals

    # Within tolerance? For ACC use percentage points; for MCC use ±0.015.
    within = np.array([
        abs(diff[i]) <= (tolerance if "ACC" in keys[i] else 0.015)
        for i in range(len(keys))
    ])
    status = np.array(["✓" if w else "✗" for w in within])

    table = np.stack([paper_vals, ours_vals, diff], axis=1)

    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(keys) + 1.5), constrained_layout=True)
    sns.heatmap(
        table, annot=True, fmt=".3f", cmap="RdYlGn_r",
        center=0, cbar=False,
        xticklabels=["paper", "ours", "Δ"],
        yticklabels=[f"{k}  {s}" for k, s in zip(keys, status)],
        ax=ax,
    )
    ax.set_title(title + f"  (ACC tol ±{tolerance}pp, MCC tol ±0.015)")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_confusion_matrix(
    preds: np.ndarray,
    targets: np.ndarray,
    out_path: str | Path,
    *,
    title: str = "Confusion matrix",
) -> Path:
    """Binary confusion matrix heatmap with counts + rates."""
    apply_style()
    from src.evaluation.classification import confusion
    c = confusion(preds, targets)
    cm = np.array([[c.tp, c.fn], [c.fp, c.tn]])
    total = cm.sum()
    rates = cm / total
    annot = np.array([[f"{cm[i, j]}\n({rates[i, j]:.1%})"
                       for j in range(2)] for i in range(2)])

    fig, ax = plt.subplots(figsize=(4.5, 4), constrained_layout=True)
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False,
                xticklabels=["Pred Up", "Pred Down"],
                yticklabels=["True Up", "True Down"], ax=ax)
    ax.set_title(f"{title}\nACC = {(c.tp + c.tn) / total:.4f}, MCC computed separately")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_ablation_bar(
    results: Dict[str, Tuple[float, float]],
    paper_results: Dict[str, Tuple[float, float]],
    out_path: str | Path,
    *,
    title: str = "Ablation comparison (ACL18)",
) -> Path:
    """Side-by-side bar: ACC (left) and MCC (right) per variant.

    Args:
        results: {"full": (acc, mcc), "no_tcd": ..., ...}
        paper_results: same shape, paper's reported numbers.
    """
    apply_style()
    variants = list(results.keys())
    ours_acc = [results[v][0] for v in variants]
    ours_mcc = [results[v][1] for v in variants]
    paper_acc = [paper_results.get(v, (np.nan, np.nan))[0] for v in variants]
    paper_mcc = [paper_results.get(v, (np.nan, np.nan))[1] for v in variants]

    x = np.arange(len(variants))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, ours, papers, label in [(axes[0], ours_acc, paper_acc, "ACC (%)"),
                                     (axes[1], ours_mcc, paper_mcc, "MCC")]:
        ax.bar(x - w / 2, papers, w, label="paper", color=PALETTE["paper"])
        ax.bar(x + w / 2, ours,    w, label="ours",  color=PALETTE["full"])
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=20, ha="right")
        ax.set_ylabel(label)
        ax.legend(loc="best", frameon=False)
    fig.suptitle(title, fontsize=12)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out

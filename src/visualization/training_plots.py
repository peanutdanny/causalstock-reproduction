"""Training-time visualizations.

Inputs are the JSON files produced by Trainer (`experiments/logs/<exp>/history.json`)
and the result summaries (`experiments/results/causalstock_*.json`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from ._style import PALETTE, apply_style


def _load_history(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    # train.py saves the full summary; history might be at "history" key or top-level
    if isinstance(data, dict) and "history" in data:
        return data["history"]
    return data  # already a list


def plot_training_curve(
    history_files: dict[str, Path | str],
    out_path: Path | str,
    *,
    title: str = "Training curves — ACL18",
) -> Path:
    """Plot train_loss, val_acc, val_mcc vs epoch for one or more runs.

    Args:
        history_files: {label: path_to_history_json}, e.g.
            {"full": ".../causalstock_acl18_seed0.json", "no_tcd": ...}
        out_path: PNG output.
    """
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    metrics = [("train_loss", "Train loss"), ("val_acc", "Val ACC"), ("val_mcc", "Val MCC")]
    for ax, (key, label) in zip(axes, metrics):
        for run_label, path in history_files.items():
            hist = _load_history(Path(path))
            epochs = [h["epoch"] for h in hist]
            vals = [h[key] for h in hist]
            color = PALETTE.get(run_label, None)
            ax.plot(epochs, vals, label=run_label, color=color, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        if key == "val_acc":
            ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        if key == "val_mcc":
            ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.legend(loc="best", frameon=False)
    fig.suptitle(title, fontsize=12)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_loss_components(
    history_files: dict[str, Path | str],
    out_path: Path | str,
    *,
    title: str = "ELBO decomposition over training",
) -> Path:
    """For a single run, plot likelihood/prior/entropy/BCE components if logged.

    Currently the Trainer only logs total `train_loss`. This plot is a
    placeholder until `LossOutput` components are persisted; for now we plot
    train_loss alongside negative-ELBO heuristic.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for run_label, path in history_files.items():
        hist = _load_history(Path(path))
        epochs = [h["epoch"] for h in hist]
        losses = [h["train_loss"] for h in hist]
        color = PALETTE.get(run_label, None)
        ax.plot(epochs, losses, label=f"{run_label} (total)", color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss value (lower = better)")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_sensitivity_curve(
    data: dict[str, list[float]],
    *,
    out_path: Path | str,
    x_label: str,
    y_label: str = "Test ACC",
    title: str = "Hyperparameter sensitivity",
) -> Path:
    """Generic sensitivity plot (e.g. Table 4: lr/L/λ vs ACC).

    Args:
        data: {dataset_label: [acc_at_each_setting]} where the x positions are
              implicit (use numeric labels for x_ticks separately if needed).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for label, ys in data.items():
        ax.plot(range(len(ys)), ys, "-o", label=label, linewidth=1.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out

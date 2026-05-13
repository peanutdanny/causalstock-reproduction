"""Visualization utilities for CausalStock reproduction reports.

Modules
-------
training_plots   : training curves, loss components, hyperparameter sensitivity
score_plots      : DNE 5-dim score distributions, correlations
causal_plots     : σ heatmap, Ĝ heatmap, G⊙Ĝ aggregated, NetworkX DAG
trading_plots    : APV curve (Figure 4), Sharpe bar, drawdown
comparison_plots : paper vs reproduction tables, confusion matrix, per-stock ACC
report           : HTML/Markdown report generator
"""
from .training_plots import plot_training_curve, plot_loss_components
from .score_plots import plot_score_distribution, plot_score_correlation
from .causal_plots import plot_sigma_heatmap, plot_causal_strength, plot_market_cap_scatter
from .trading_plots import plot_apv_curve, plot_sharpe_bar
from .comparison_plots import plot_reproduction_table, plot_confusion_matrix, plot_ablation_bar

__all__ = [
    "plot_training_curve", "plot_loss_components",
    "plot_score_distribution", "plot_score_correlation",
    "plot_sigma_heatmap", "plot_causal_strength", "plot_market_cap_scatter",
    "plot_apv_curve", "plot_sharpe_bar",
    "plot_reproduction_table", "plot_confusion_matrix", "plot_ablation_bar",
]

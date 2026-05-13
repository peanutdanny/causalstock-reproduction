"""Trading-simulation metrics (Section 8.2 of docs/paper-summary.md).

Top-k portfolio strategy: each trading day T, take the k stocks with the
highest predicted up-probability and weight them equally (long-only).

Eq. 17 (APV) and Eq. 18 (Sharpe ratio).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def _as_numpy(t) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def top_k_portfolio_returns(
    probs: np.ndarray | torch.Tensor,         # (T, D) up-probabilities
    returns: np.ndarray | torch.Tensor,       # (T, D) realised next-day returns
    k: int = 3,
) -> np.ndarray:
    """Per-day portfolio return r_t = mean over chosen top-k stocks."""
    P = _as_numpy(probs)
    R = _as_numpy(returns)
    assert P.shape == R.shape, f"shape mismatch {P.shape} vs {R.shape}"
    T, D = P.shape
    k = min(k, D)
    out = np.empty(T, dtype=np.float64)
    for t in range(T):
        idx = np.argpartition(P[t], -k)[-k:]
        out[t] = R[t, idx].mean()
    return out


def accumulated_portfolio_value(per_day_returns: Sequence[float]) -> np.ndarray:
    """Eq. 17: APV_t = ∏_{i=1..t} (1 + r_i)."""
    r = np.asarray(per_day_returns, dtype=np.float64)
    return np.cumprod(1.0 + r)


def sharpe_ratio(per_day_returns: Sequence[float], risk_free: float = 0.0) -> float:
    """Eq. 18: SR = E[r - R_f] / Std[r - R_f]. Daily, not annualized."""
    r = np.asarray(per_day_returns, dtype=np.float64) - risk_free
    std = r.std(ddof=1)
    if std == 0 or len(r) < 2:
        return 0.0
    return float(r.mean() / std)

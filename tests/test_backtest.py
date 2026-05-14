"""Smoke + correctness tests for scripts/run_backtest.py.

Trading metric correctness (top-k / APV / Sharpe) is already covered in
tests/test_metrics.py — here we test:
  * the realized-return helper uses StockNet's `movement` column,
  * APV/Sharpe wired through plot helpers don't crash on real backtest output,
  * make_plots integration picks up backtest NPZ.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_backtest import _realized_returns_for_target


class _StubDataset:
    """Mimics CausalStockDataset just enough for _realized_returns_for_target."""

    def __init__(self, stocks, dates, movements):
        self.stocks = stocks
        self.D = len(stocks)
        self._common_dates = list(dates)
        self._date_to_idx = {d: i for i, d in enumerate(self._common_dates)}
        self._stock_date_to_row = {
            s: {d: i for i, d in enumerate(self._common_dates)} for s in stocks
        }
        self.price_dfs = {
            s: pd.DataFrame({"date": self._common_dates, "movement": movements[s]})
            for s in stocks
        }


def test_realized_returns_reads_movement_column():
    dates = pd.to_datetime(["2015-10-01", "2015-10-02", "2015-10-05"])
    stocks = ["AAA", "BBB"]
    ds = _StubDataset(stocks, dates, {
        "AAA": [0.01, -0.02, 0.03],
        "BBB": [0.0, 0.05, -0.01],
    })
    r = _realized_returns_for_target(ds, dates[1])  # day 2015-10-02
    assert r.shape == (2,)
    assert r[0] == pytest.approx(-0.02)
    assert r[1] == pytest.approx(0.05)


def test_realized_returns_unknown_date_returns_zeros():
    dates = pd.to_datetime(["2015-10-01"])
    ds = _StubDataset(["AAA"], dates, {"AAA": [0.01]})
    r = _realized_returns_for_target(ds, pd.Timestamp("2099-01-01"))
    assert r.shape == (1,)
    assert r[0] == 0.0


@pytest.mark.skipif(
    not (ROOT / "experiments/results/backtest_acl18_full.npz").exists(),
    reason="ACL18 backtest NPZ not produced (run scripts/run_backtest.py first)",
)
def test_backtest_npz_shape_and_range():
    """Sanity-check the on-disk backtest output (if available)."""
    d = np.load(ROOT / "experiments/results/backtest_acl18_full.npz")
    probs, returns = d["probs"], d["returns"]
    assert probs.shape == returns.shape
    assert probs.ndim == 2
    # f_i is sigmoided in FCM → in [0, 1].
    assert (probs >= 0).all() and (probs <= 1).all()
    # ACL18 daily |movement| typically < 25%.
    assert np.abs(returns).max() < 0.5
    assert d["stocks"].shape[0] == probs.shape[1]
    assert d["dates"].shape[0] == probs.shape[0]


@pytest.mark.skipif(
    not (ROOT / "experiments/results/backtest_acl18_full.npz").exists(),
    reason="ACL18 backtest NPZ not produced",
)
def test_trading_plots_render_from_backtest(tmp_path):
    from src.visualization import plot_apv_curve, plot_sharpe_bar

    d = np.load(ROOT / "experiments/results/backtest_acl18_full.npz")
    runs = {"full": {"probs": d["probs"], "returns": d["returns"]}}
    apv_path = plot_apv_curve(runs, tmp_path / "apv.png", k=3,
                              paper_final_apv=1.32)
    sr_path = plot_sharpe_bar(runs, tmp_path / "sr.png", k=3,
                              paper_sharpe=0.369)
    assert apv_path.exists() and apv_path.stat().st_size > 0
    assert sr_path.exists() and sr_path.stat().st_size > 0

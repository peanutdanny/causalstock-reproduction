"""ACL18 (StockNet) dataset glue — chronological train/valid/test split.

Reference: docs/paper-summary.md §7.1
    Train: 2014/01/02–2015/08/02
    Valid: 2015/08/03–2015/09/30
    Test : 2015/10/01–2016/01/01
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from .dataset import CausalStockDataset, NewsScorer
from .stocknet import load_stocknet_prices, load_stocknet_tweets


def _date_range(start: str, end: str, all_trading_days: List[pd.Timestamp]) -> List[pd.Timestamp]:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return [d for d in all_trading_days if s <= d <= e]


def build_acl18_splits(
    price_root: str | Path,
    tweet_root: str | Path,
    train_range: Tuple[str, str],
    valid_range: Tuple[str, str],
    test_range: Tuple[str, str],
    lag_L: int = 5,
    news_per_day: int = 10,
    news_scorer: Optional[NewsScorer] = None,
    movement_threshold: float = 0.0,
    only_tickers: Optional[List[str]] = None,
) -> Tuple[CausalStockDataset, CausalStockDataset, CausalStockDataset]:
    prices = load_stocknet_prices(price_root)
    if only_tickers is not None:
        prices = {k: v for k, v in prices.items() if k in only_tickers}

    # Restrict to the intersection of price tickers and tweet folders.
    tweet_dir = Path(tweet_root)
    tweet_stocks = {p.name for p in tweet_dir.iterdir() if p.is_dir()}
    keep = sorted(set(prices.keys()) & tweet_stocks)

    # Drop tickers whose price history doesn't cover the full target window
    # (e.g. GMRE IPO'd 2016-07; BABA 2014-09; AGFS 2014-11). The paper claims
    # "88 stocks" but only 85 of the StockNet tickers cover 2013-12..2016-01.
    # TODO(paper-ambiguity): docs/reproduction-questions.md A.4 — exact ticker list.
    coverage_start = pd.Timestamp(train_range[0]) - pd.Timedelta(days=30)
    coverage_end = pd.Timestamp(test_range[1])
    keep = [
        k for k in keep
        if prices[k]["date"].min() <= coverage_start
        and prices[k]["date"].max() >= coverage_end
    ]
    prices = {k: prices[k] for k in keep}
    stocks = keep

    if not stocks:
        raise ValueError("No overlapping tickers between price and tweet directories")

    all_dates = sorted(set.intersection(*[set(df["date"]) for df in prices.values()]))

    train_dates = _date_range(*train_range, all_dates)
    valid_dates = _date_range(*valid_range, all_dates)
    test_dates = _date_range(*test_range, all_dates)

    # Drop the first L target dates of train (no prior lag available).
    if len(train_dates) > lag_L:
        train_dates = train_dates[lag_L:]

    all_needed_dates = sorted(set(train_dates) | set(valid_dates) | set(test_dates))
    # We also need dates in the L-window prior to each target.
    if all_needed_dates:
        earliest = min(all_needed_dates)
        idx0 = all_dates.index(earliest)
        prior = all_dates[max(0, idx0 - lag_L) : idx0]
        all_needed_dates = sorted(set(prior) | set(all_needed_dates))

    tweets = load_stocknet_tweets(tweet_root, stocks, all_needed_dates) if news_scorer else {}

    # Compute z-score normalization from the train range only, then share with
    # val/test (docs/reproduction-questions.md A.1).
    if train_dates:
        train_lo = pd.Timestamp(train_range[0])
        train_hi = pd.Timestamp(train_range[1])
        feature_mean, feature_std = CausalStockDataset.compute_feature_stats(
            prices, date_range=(train_lo, train_hi)
        )
    else:
        feature_mean = feature_std = None

    def _make(dates):
        return CausalStockDataset(
            price_dfs=prices,
            tweets=tweets,
            stocks=stocks,
            target_dates=dates,
            lag_L=lag_L,
            news_per_day=news_per_day,
            news_scorer=news_scorer,
            movement_threshold=movement_threshold,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )

    return _make(train_dates), _make(valid_dates), _make(test_dates)

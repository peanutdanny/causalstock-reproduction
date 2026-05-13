"""Generic CausalStock-style Dataset.

Per paper-summary §1, each sample is a target trading day T plus the past L
trading days of (price, news) for D stocks. We yield:

    P       : (D, L, F) float — numeric price features for [T-L .. T-1]
    news    : list[list[list[str]]] of shape (D, L) — raw news texts per
              (stock, day). Phase 3a/3b consume this via a news_scorer that
              returns (D, L, l, 5) scores. Stored as nested lists because
              collation depends on the scorer.
    y       : (D,) long — 1 iff close_T > close_{T-1}.
    meta    : dict with target_date, stocks, dates_window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .stocknet import NUM_PRICE_FEATURES, PRICE_FEATURE_COLUMNS


NewsScorer = Callable[[str, str, List[str]], np.ndarray]
"""Callable (ticker, date_str, raw_texts) -> ndarray of shape (l, 5)."""


@dataclass
class CausalStockSample:
    P: torch.Tensor       # (D, L, F)
    C: torch.Tensor       # (D, L, l, 5) if scorer provided, else zeros
    y: torch.Tensor       # (D,) long
    target_date: pd.Timestamp
    stocks: Tuple[str, ...]


class CausalStockDataset(Dataset):
    """Lazy index over valid target dates."""

    def __init__(
        self,
        price_dfs: Dict[str, pd.DataFrame],
        tweets: Dict[Tuple[str, str], List[str]],
        stocks: List[str],
        target_dates: List[pd.Timestamp],
        lag_L: int,
        news_per_day: int = 10,
        news_scorer: Optional[NewsScorer] = None,
        movement_threshold: float = 0.0,
    ):
        self.stocks = tuple(stocks)
        self.D = len(stocks)
        self.L = lag_L
        self.l = news_per_day
        self.scorer = news_scorer
        self.threshold = movement_threshold
        self.tweets = tweets
        self.price_dfs = price_dfs
        self._stock_date_to_row = {
            s: {d: i for i, d in enumerate(df["date"].tolist())}
            for s, df in price_dfs.items()
        }
        self._features = {
            s: df[PRICE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            for s, df in price_dfs.items()
        }
        self._close = {s: df["close"].to_numpy(dtype=np.float32) for s, df in price_dfs.items()}
        self._common_dates = sorted(
            set.intersection(*[set(df["date"].tolist()) for df in price_dfs.values()])
        )
        common = set(self._common_dates)
        self.samples = [d for d in target_dates if d in common]
        self._date_to_idx = {d: i for i, d in enumerate(self._common_dates)}

    @property
    def F(self) -> int:
        return NUM_PRICE_FEATURES

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> CausalStockSample:
        T = self.samples[idx]
        tT = self._date_to_idx[T]
        if tT < self.L:
            raise IndexError(f"target {T} lacks {self.L} prior trading days")

        window_dates = self._common_dates[tT - self.L : tT]  # T-L .. T-1 (length L)
        P = np.zeros((self.D, self.L, self.F), dtype=np.float32)
        y = np.zeros((self.D,), dtype=np.int64)
        C = np.zeros((self.D, self.L, self.l, 5), dtype=np.float32)

        for i, s in enumerate(self.stocks):
            rows = self._stock_date_to_row[s]
            feats = self._features[s]
            close = self._close[s]
            for j, d in enumerate(window_dates):
                row = rows.get(d)
                if row is None:
                    continue  # leave zero; intersection guarantees presence
                P[i, j] = feats[row]
                if self.scorer is not None:
                    ds = d.strftime("%Y-%m-%d")
                    raw = self.tweets.get((s, ds), [])
                    scores = self.scorer(s, ds, raw)  # (l, 5)
                    C[i, j] = scores
            row_T = rows.get(T)
            row_T_minus_1 = rows.get(window_dates[-1])
            if row_T is not None and row_T_minus_1 is not None:
                y[i] = int(close[row_T] > close[row_T_minus_1] + self.threshold)

        return CausalStockSample(
            P=torch.from_numpy(P),
            C=torch.from_numpy(C),
            y=torch.from_numpy(y),
            target_date=T,
            stocks=self.stocks,
        )


def collate_samples(batch: List[CausalStockSample]) -> Dict[str, torch.Tensor]:
    """Stack into a dict-of-tensors batch."""
    return {
        "P": torch.stack([s.P for s in batch], dim=0),
        "C": torch.stack([s.C for s in batch], dim=0),
        "y": torch.stack([s.y for s in batch], dim=0),
        "target_dates": [s.target_date for s in batch],
    }

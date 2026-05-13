"""Parquet-backed cache for DNE scores keyed by (ticker, date, news_hash).

Real GPT-3.5 calls are expensive and slow; we precompute scores once and reuse.
Mock and real scorers share this cache so they're drop-in replacements.

Key design: each (ticker, date) maps to one row containing the (l, 5) score
matrix flattened to a length-50 float array plus a hash of the concatenated
raw_texts so we can detect dataset changes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _news_digest(raw_texts: Sequence[str]) -> str:
    h = hashlib.sha1()
    for t in raw_texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


class DNECache:
    def __init__(self, path: str | Path, news_per_day: int = 10):
        self.path = Path(path)
        self.l = news_per_day
        self._data: Dict[Tuple[str, str], Tuple[np.ndarray, str]] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        df = pd.read_parquet(self.path)
        for _, row in df.iterrows():
            arr = np.asarray(row["scores"], dtype=np.float32).reshape(self.l, 5)
            self._data[(row["ticker"], row["date"])] = (arr, row["digest"])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for (ticker, date), (arr, digest) in self._data.items():
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "digest": digest,
                    "scores": arr.flatten().tolist(),
                }
            )
        pd.DataFrame(rows).to_parquet(self.path, index=False)

    def get(self, ticker: str, date_str: str) -> Optional[np.ndarray]:
        entry = self._data.get((ticker, date_str))
        return entry[0].copy() if entry else None

    def put(self, ticker: str, date_str: str, scores: np.ndarray, digest: str) -> None:
        assert scores.shape == (self.l, 5)
        self._data[(ticker, date_str)] = (scores.astype(np.float32), digest)

    def __len__(self) -> int:
        return len(self._data)


class CachedScorer:
    """Wraps a base scorer with the DNECache. Used by the Dataset.

    On miss, calls base; on hit, returns cached array.
    """

    def __init__(
        self,
        base: Callable[[str, str, Sequence[str]], np.ndarray],
        cache: DNECache,
    ):
        self.base = base
        self.cache = cache

    def __call__(self, ticker: str, date_str: str, raw_texts: Sequence[str]) -> np.ndarray:
        cached = self.cache.get(ticker, date_str)
        if cached is not None:
            return cached
        arr = self.base(ticker, date_str, raw_texts)
        self.cache.put(ticker, date_str, arr, _news_digest(raw_texts))
        return arr

"""Deterministic mock DNE scorer (Phase 3a).

Produces (l, 5) scores per (stock, date) without calling the real GPT-3.5 API,
so the rest of the pipeline (TCD, FCM, training loop) can be tested before any
API spend. Hash-based for reproducibility.

The score ranges match the paper (§3.2 of docs/paper-summary.md):
    Correlation, Importance, Impact, Duration ∈ [0, 10]
    Sentiment ∈ [-1, 1]
"""
from __future__ import annotations

import hashlib
from typing import List, Sequence

import numpy as np


def _hash_to_unit(s: str, salt: int) -> float:
    """Stable float in [0, 1) derived from a string and an integer salt."""
    h = hashlib.sha1(f"{salt}:{s}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / (1 << 64)


class MockDNEScorer:
    """Hash-deterministic scorer with the same call signature as the real one."""

    def __init__(self, news_per_day: int = 10):
        self.l = news_per_day

    def __call__(self, ticker: str, date_str: str, raw_texts: Sequence[str]) -> np.ndarray:
        scores = np.zeros((self.l, 5), dtype=np.float32)
        # Time-ordered top-l (docs/reproduction-questions.md A.2 default).
        texts = list(raw_texts)[: self.l]
        for k, txt in enumerate(texts):
            key = f"{ticker}|{date_str}|{k}|{txt[:200]}"
            scores[k, 0] = _hash_to_unit(key, 1) * 10.0   # Correlation
            scores[k, 1] = _hash_to_unit(key, 2) * 2 - 1  # Sentiment
            scores[k, 2] = _hash_to_unit(key, 3) * 10.0   # Importance
            scores[k, 3] = _hash_to_unit(key, 4) * 10.0   # Impact
            scores[k, 4] = _hash_to_unit(key, 5) * 10.0   # Duration
        # Empty days remain zero (paper Appendix A allows all-zero only when
        # truly impossible; mock returns zero for missing news).
        return scores

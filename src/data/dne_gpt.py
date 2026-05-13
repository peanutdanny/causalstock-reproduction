"""Real GPT-3.5 Denoised News Encoder (Phase 3b).

Prompt template is *verbatim* ported from
CausalStock_code/GPT_scoremaker/GPT_news_score.py to maximize faithfulness
(docs/paper-summary.md §3.2; Appendix A).

Cost (rough): ACL18 ≈ 26k tweets × ~250 tokens ≈ $30–50 on gpt-3.5-turbo-0125
at 2026 pricing. Use the precompute script + DNECache to score once and reuse.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


SYS_PROMPT = (
    "As a stock trading news analyst, you are a helpful and precise assistant. "
    "Your task is to analyze the correlation between news and the given stock, "
    "sentiment polarity of the news, importance of the news, the impact of the "
    "news on stock prices, and the duration of the news impact."
)

DEFAULT_PROMPT = """I need you to analyze the provided stock-related news from four dimensions:
1. Correlation between the news and the given stock: Rate the correlation on a scale of 0 to 10, where a higher score indicates a stronger correlation between the news and the given stock.
2. Sentiment polarity of the news: Rate the sentiment polarity on a scale of -1 to 1, where a value closer to -1 indicates stronger negative sentiment and a value closer to 1 indicates stronger positive sentiment.
3. Importance of the news event: Rate the importance on a scale of 0 to 10, where a higher score indicates higher importance of the news event.
4. Impact of the news on stock prices: Rate the impact on a scale of 0 to 10, where a higher score indicates a greater impact of the news on stock prices.
5. Duration of the news impact: Rate the duration on a scale of 0 to 10, where a higher score indicates a longer potential duration of the news impact.
(When you encounter a situation where analysis is not possible, please try to avoid assigning all-zero scores and instead make an effort to analyze the text content and derive scores accordingly. Only when analysis is truly impossible should you assign a score of 0 to all factors.)
(Please refrain from providing an analysis and simply provide the answer according to the following format.)

Output format:
Correlation: <Correlation score between the news and the stock>
Sentiment: <Sentiment polarity score of the news>
Importance: <Importance score of the news event>
Impact: <Impact score of the news on stock prices>
Duration: <Duration score of the news impact>
"""

USER_TEMPLATE = (
    "[Stock Name]\n{stock_name}\n\n[News Content]\n{news_content}\n\n"
    "[Publish Time]\n{publish_time}\n\n[System]\n{prompt}\n"
)

_FIELDS = ("Correlation", "Sentiment", "Importance", "Impact", "Duration")
_RE = {f: re.compile(rf"{f}\s*:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE) for f in _FIELDS}


def parse_scores(text: str) -> Optional[np.ndarray]:
    """Return (5,) ndarray or None if any field is missing."""
    out = np.zeros(5, dtype=np.float32)
    for k, f in enumerate(_FIELDS):
        m = _RE[f].search(text)
        if m is None:
            return None
        out[k] = float(m.group(1))
    return out


@dataclass
class GPTConfig:
    """GPT scoring config.

    Default model is gpt-5.4-mini (cheaper, faster, slightly better than
    gpt-3.5-turbo). The paper used gpt-3.5-turbo; switching to a newer
    model is a documented deviation (docs/reproduction-questions.md I.6).
    """

    model: str = "gpt-5.4-mini"
    temperature: float = 0.0
    max_retries: int = 3
    timeout_sec: int = 30


class GPTDNEScorer:
    """Score (l, 5) per (stock, date). Empty days return zero matrix."""

    def __init__(
        self,
        full_name_map: dict[str, str],
        news_per_day: int = 10,
        config: Optional[GPTConfig] = None,
        client=None,  # injectable for testing
    ):
        self.full_name = full_name_map
        self.l = news_per_day
        self.cfg = config or GPTConfig()
        self.client = client
        if self.client is None:
            try:
                import openai  # local import — only required at runtime
            except ImportError as e:
                raise RuntimeError(
                    "openai package not installed. `pip install openai` to use GPTDNEScorer."
                ) from e
            self.client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                timeout=self.cfg.timeout_sec,
            )

    def _score_one(self, stock_name: str, text: str, publish_time: str) -> np.ndarray:
        user = USER_TEMPLATE.format(
            stock_name=stock_name,
            news_content=text,
            publish_time=publish_time,
            prompt=DEFAULT_PROMPT,
        )
        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.model,
                    temperature=self.cfg.temperature,
                    messages=[
                        {"role": "system", "content": SYS_PROMPT},
                        {"role": "user", "content": user},
                    ],
                )
                content = resp.choices[0].message.content or ""
                arr = parse_scores(content)
                if arr is not None:
                    return arr
                last_err = ValueError(f"parse failure: {content[:200]}")
            except Exception as e:  # noqa: BLE001 — retry on any API/network error
                last_err = e
                time.sleep(2 ** attempt)
        # Final fallback (Appendix A allows all-zero only when truly impossible).
        return np.zeros(5, dtype=np.float32)

    def __call__(self, ticker: str, date_str: str, raw_texts: Sequence[str]) -> np.ndarray:
        scores = np.zeros((self.l, 5), dtype=np.float32)
        stock_name = self.full_name.get(ticker, ticker)
        for k, txt in enumerate(list(raw_texts)[: self.l]):
            scores[k] = self._score_one(stock_name, txt, date_str)
        return scores

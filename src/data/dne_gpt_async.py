"""Async GPT-3.5 DNE scorer.

Same prompt & parser as the sync GPTDNEScorer (paper-verbatim), but uses
openai.AsyncOpenAI + asyncio.Semaphore so many API calls can be in flight
simultaneously. Bound by OpenAI tier's TPM/RPM limits.

For Tier 1 (60K TPM gpt-3.5-turbo-0125):
    ~550 tokens/call → ~109 calls/min ceiling regardless of concurrency.
For Tier 3+ (1M TPM):
    ~1800 calls/min → concurrency=50 fully utilizable.
"""
from __future__ import annotations

import asyncio
import os
from typing import Optional, Sequence

import numpy as np

from .dne_gpt import (
    DEFAULT_PROMPT,
    GPTConfig,
    SYS_PROMPT,
    USER_TEMPLATE,
    parse_scores,
)


class AsyncGPTDNEScorer:
    """Score many (stock, date) pairs in parallel via asyncio.

    Public surface:
        await score_pair(ticker, date_str, raw_texts) -> np.ndarray (l, 5)
        await score_pairs(pairs_with_texts) -> list of np.ndarray
    """

    def __init__(
        self,
        full_name_map: dict[str, str],
        news_per_day: int = 20,
        config: Optional[GPTConfig] = None,
        concurrency: int = 20,
        client=None,
    ):
        self.full_name = full_name_map
        self.l = news_per_day
        self.cfg = config or GPTConfig()
        self.semaphore = asyncio.Semaphore(concurrency)
        self.client = client
        if self.client is None:
            try:
                import openai
            except ImportError as e:
                raise RuntimeError("openai package not installed") from e
            self.client = openai.AsyncOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                timeout=self.cfg.timeout_sec,
            )

    async def _score_one(self, stock_name: str, text: str, publish_time: str) -> tuple[np.ndarray, bool]:
        """Return (scores, ok). ok=False signals "do not cache" (transient failure).

        Strategy:
        - Parse failure (model returned bad output): up to max_retries, then ok=True
          with zeros (treat as model's fault — won't be retried).
        - Network/connection error: wait with exponential backoff *forever*.
          We never give up on these — better to wait through an internet outage
          than to permanently store wrong zeros.
        """
        import openai
        user = USER_TEMPLATE.format(
            stock_name=stock_name,
            news_content=text,
            publish_time=publish_time,
            prompt=DEFAULT_PROMPT,
        )
        parse_failures = 0
        attempt = 0
        async with self.semaphore:
            while True:
                try:
                    resp = await self.client.chat.completions.create(
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
                        return arr, True
                    # Parse failure — give the model a few tries then accept zeros.
                    parse_failures += 1
                    if parse_failures >= self.cfg.max_retries:
                        return np.zeros(5, dtype=np.float32), True
                except (openai.APIConnectionError, openai.APITimeoutError) as e:
                    # Network problem — wait, don't give up.
                    wait = min(60, 2 ** min(attempt, 6))
                    attempt += 1
                    await asyncio.sleep(wait)
                except openai.RateLimitError:
                    # Tier rate limit — wait a longer fixed time.
                    await asyncio.sleep(30)
                except Exception:
                    # Other API errors (auth, model not found etc): mark as transient
                    # so we never cache wrong zeros for those either.
                    wait = min(60, 2 ** min(attempt, 6))
                    attempt += 1
                    await asyncio.sleep(wait)

    async def score_pair(self, ticker: str, date_str: str, raw_texts: Sequence[str]) -> tuple[np.ndarray, bool]:
        """Return (scores, all_ok). all_ok=False ⇒ caller should NOT cache."""
        scores = np.zeros((self.l, 5), dtype=np.float32)
        stock_name = self.full_name.get(ticker, ticker)
        texts = list(raw_texts)[: self.l]
        if not texts:
            return scores, True
        tasks = [self._score_one(stock_name, t, date_str) for t in texts]
        results = await asyncio.gather(*tasks)
        all_ok = True
        for k, (arr, ok) in enumerate(results):
            scores[k] = arr
            all_ok = all_ok and ok
        return scores, all_ok

    async def score_pairs(
        self, pairs: list[tuple[str, str, Sequence[str]]]
    ) -> list[tuple[np.ndarray, bool]]:
        tasks = [self.score_pair(ticker, ds, raw) for ticker, ds, raw in pairs]
        return await asyncio.gather(*tasks)

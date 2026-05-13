"""Async parallel version of scripts/score_news.py.

Usage:
    .venv/bin/python scripts/score_news_async.py --gpt --concurrency 30 \
        --cache data/processed/dne_acl18.parquet --batch 200

Strategy:
    - Build the full list of (ticker, date_str, raw_texts) we still need to score
      (skip ones already in cache).
    - Process in waves of `--batch` pairs. Each wave runs `concurrency` API
      calls in flight at any time. After each wave, save the cache.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.data import DNECache, load_stocknet_tweets
from src.data.acl18 import build_acl18_splits
from src.data.stock_names import STOCK_FULL_NAMES
from src.data.dne_gpt_async import AsyncGPTDNEScorer


async def run(args):
    cache_path = ROOT / args.cache
    cache = DNECache(cache_path, news_per_day=args.news_per_day)
    print(f"cache initial size: {len(cache)} (news_per_day={args.news_per_day})")

    train, _, _ = build_acl18_splits(
        price_root=ROOT / "reference_data/stocknet-dataset-master/price/preprocessed",
        tweet_root=ROOT / "reference_data/stocknet-dataset-master/tweet/preprocessed",
        train_range=("2014-01-02", "2015-08-02"),
        valid_range=("2015-08-03", "2015-09-30"),
        test_range=("2015-10-01", "2016-01-01"),
        lag_L=5,
    )
    lo, hi = pd.Timestamp(args.date_start), pd.Timestamp(args.date_end)
    all_dates = sorted({d for d in train._common_dates if lo <= d <= hi})
    stocks = train.stocks
    print(f"date range: {all_dates[0].date()} .. {all_dates[-1].date()} ({len(all_dates)} days, {len(stocks)} stocks)")
    tweets = load_stocknet_tweets(
        ROOT / "reference_data/stocknet-dataset-master/tweet/preprocessed",
        list(stocks), all_dates,
    )

    # Build queue, skipping already-cached pairs.
    queue = []
    for s in stocks:
        for d in all_dates:
            ds = d.strftime("%Y-%m-%d")
            if cache.get(s, ds) is not None:
                continue
            raw = tweets.get((s, ds), [])
            queue.append((s, ds, raw))
    print(f"pairs to score: {len(queue)}")

    if args.limit:
        queue = queue[: args.limit]

    from src.data.dne_gpt import GPTConfig
    scorer = AsyncGPTDNEScorer(
        full_name_map=STOCK_FULL_NAMES,
        news_per_day=args.news_per_day,
        concurrency=args.concurrency,
        config=GPTConfig(model=args.model),
    )
    print(f"using model: {args.model}")

    pbar = tqdm(total=len(queue), desc="scoring (async)")
    t0 = time.time()
    api_calls_est = 0
    for batch_start in range(0, len(queue), args.batch):
        batch = queue[batch_start : batch_start + args.batch]
        # Skip empty pairs entirely (write zero immediately).
        nonempty = [(s, ds, raw) for (s, ds, raw) in batch if raw]
        empty = [(s, ds) for (s, ds, raw) in batch if not raw]
        for s, ds in empty:
            cache.put(s, ds, np.zeros((args.news_per_day, 5), dtype=np.float32), "")

        if nonempty:
            results = await scorer.score_pairs(nonempty)
            skipped = 0
            for (s, ds, raw), (arr, ok) in zip(nonempty, results):
                if ok:
                    cache.put(s, ds, arr, "")
                    api_calls_est += min(len(raw), args.news_per_day)
                else:
                    skipped += 1  # transient failure → don't cache, retry on next run
            if skipped:
                pbar.write(f"  ⚠ skipped {skipped} pair(s) due to transient errors — will retry on rerun")

        pbar.update(len(batch))
        cache.save()
        elapsed = time.time() - t0
        rate = pbar.n / elapsed if elapsed > 0 else 0
        eta = (len(queue) - pbar.n) / rate if rate > 0 else 0
        pbar.set_postfix(rate=f"{rate:.1f} pair/s", calls=api_calls_est, eta=f"{eta/60:.0f}m")
    pbar.close()
    cache.save()
    print(f"\ncache final size: {len(cache)} | API calls ~{api_calls_est} | wall-clock {(time.time()-t0)/60:.1f}m")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpt", action="store_true", required=True)
    p.add_argument("--cache", default="data/processed/dne_acl18.parquet")
    p.add_argument("--batch", type=int, default=200,
                   help="Pairs per wave; cache.save() after each wave")
    p.add_argument("--concurrency", type=int, default=20,
                   help="Concurrent in-flight GPT calls (semaphore size)")
    p.add_argument("--news-per-day", type=int, default=20)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--date-start", default="2013-12-15")
    p.add_argument("--date-end", default="2016-01-15")
    p.add_argument("--model", default="gpt-5.4-mini",
                   help="OpenAI model id. Paper used gpt-3.5-turbo-0125.")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

"""Precompute DNE scores for a dataset using the real GPT-3.5 API.

WARNING: Spends real money. Default uses MockDNEScorer; pass --gpt for real.

    .venv/bin/python scripts/score_news.py --dataset acl18 --gpt --batch 50

Resumable: every batch is appended to the parquet cache; if interrupted,
re-running skips already-cached (stock, date) pairs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass  # .env not loaded; rely on shell-exported OPENAI_API_KEY

import pandas as pd

from src.data import DNECache, MockDNEScorer, load_stocknet_tweets
from src.data.acl18 import build_acl18_splits
from src.data.stock_names import STOCK_FULL_NAMES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="acl18", choices=["acl18"])
    p.add_argument("--gpt", action="store_true", help="Use real GPT-3.5 (costs money)")
    p.add_argument("--batch", type=int, default=50, help="Save cache every N (stock, date) pairs")
    p.add_argument("--cache", default="data/processed/dne_acl18.parquet")
    p.add_argument("--limit", type=int, default=None, help="Stop after N (stock, date) pairs")
    p.add_argument("--date-start", default="2013-12-15",
                   help="Earliest date to score (default covers train_start - lag_L lookback)")
    p.add_argument("--date-end", default="2016-01-15",
                   help="Latest date to score (default covers test_end)")
    p.add_argument("--skip-empty", action="store_true",
                   help="Skip (stock, date) pairs that have no news (still write zeros to cache)")
    p.add_argument("--news-per-day", type=int, default=20,
                   help="How many news to score per day. Paper scored 20 (model uses 10).")
    args = p.parse_args()

    cache_path = ROOT / args.cache
    cache = DNECache(cache_path, news_per_day=args.news_per_day)
    print(f"cache initial size: {len(cache)} (news_per_day={args.news_per_day})")

    train, valid, test = build_acl18_splits(
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
    print(f"date range: {all_dates[0].date()} .. {all_dates[-1].date()} ({len(all_dates)} trading days, {len(stocks)} stocks)")
    tweets = load_stocknet_tweets(
        ROOT / "reference_data/stocknet-dataset-master/tweet/preprocessed",
        list(stocks), all_dates,
    )

    if args.gpt:
        from src.data.dne_gpt import GPTDNEScorer
        scorer = GPTDNEScorer(full_name_map=STOCK_FULL_NAMES, news_per_day=args.news_per_day)
    else:
        scorer = MockDNEScorer(news_per_day=args.news_per_day)

    pairs = [(s, d.strftime("%Y-%m-%d")) for s in stocks for d in all_dates]
    if args.limit:
        pairs = pairs[: args.limit]

    processed = 0
    api_calls = 0
    for ticker, ds in tqdm(pairs, desc="scoring"):
        if cache.get(ticker, ds) is not None:
            continue
        raw = tweets.get((ticker, ds), [])
        if args.skip_empty and not raw:
            import numpy as np
            cache.put(ticker, ds, np.zeros((args.news_per_day, 5), dtype=np.float32), "")
            processed += 1
            continue
        arr = scorer(ticker, ds, raw)
        cache.put(ticker, ds, arr, "")
        processed += 1
        api_calls += len(raw[: args.news_per_day])
        if processed % args.batch == 0:
            cache.save()
    cache.save()
    print(f"cache final size: {len(cache)} (added {processed}, GPT calls ~{api_calls})")


if __name__ == "__main__":
    main()

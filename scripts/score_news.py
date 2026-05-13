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

from src.data import DNECache, MockDNEScorer, load_stocknet_tweets
from src.data.acl18 import build_acl18_splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="acl18", choices=["acl18"])
    p.add_argument("--gpt", action="store_true", help="Use real GPT-3.5 (costs money)")
    p.add_argument("--batch", type=int, default=50, help="Save cache every N (stock, date) pairs")
    p.add_argument("--cache", default="data/processed/dne_acl18.parquet")
    p.add_argument("--limit", type=int, default=None, help="Stop after N pairs (testing)")
    args = p.parse_args()

    cache_path = ROOT / args.cache
    cache = DNECache(cache_path, news_per_day=10)
    print(f"cache initial size: {len(cache)}")

    train, valid, test = build_acl18_splits(
        price_root=ROOT / "reference_data/stocknet-dataset-master/price/preprocessed",
        tweet_root=ROOT / "reference_data/stocknet-dataset-master/tweet/preprocessed",
        train_range=("2014-01-02", "2015-08-02"),
        valid_range=("2015-08-03", "2015-09-30"),
        test_range=("2015-10-01", "2016-01-01"),
        lag_L=5,
    )
    all_dates = sorted(set(train._common_dates))
    stocks = train.stocks
    tweets = load_stocknet_tweets(
        ROOT / "reference_data/stocknet-dataset-master/tweet/preprocessed",
        list(stocks), all_dates,
    )

    if args.gpt:
        from src.data.dne_gpt import GPTDNEScorer
        full_name_map = {s: s for s in stocks}  # caller can replace with company names
        scorer = GPTDNEScorer(full_name_map=full_name_map, news_per_day=10)
    else:
        scorer = MockDNEScorer(news_per_day=10)

    pairs = [(s, d.strftime("%Y-%m-%d")) for s in stocks for d in all_dates]
    if args.limit:
        pairs = pairs[: args.limit]

    processed = 0
    for ticker, ds in tqdm(pairs, desc="scoring"):
        if cache.get(ticker, ds) is not None:
            continue
        arr = scorer(ticker, ds, tweets.get((ticker, ds), []))
        cache.put(ticker, ds, arr, "")
        processed += 1
        if processed % args.batch == 0:
            cache.save()
    cache.save()
    print(f"cache final size: {len(cache)} (added {processed})")


if __name__ == "__main__":
    main()

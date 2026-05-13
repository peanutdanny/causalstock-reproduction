"""StockNet-format I/O.

Price files (TSV, no header): date, movement%, high_norm, low_norm, close_norm,
close_raw, volume — i.e. 1 date column + 6 numeric features (paper-summary §3.1
line 53; the "7-dim" label in §7.1 counts the date column).

Tweet files: one JSON record per line under tweet/preprocessed/{TICKER}/{YYYY-MM-DD},
each with a "text" field that is a *list of tokens*. To match the paper's
GPT-3.5 prompt (CausalStock_code/GPT_scoremaker/GPT_news_score.py line 294-297,
which passes row['text'] — the python list — directly to `.format()`), we
preserve the list and stringify it with `str(list)` so the prompt receives
literal `"['$', 'aapl', '-', ...]"` text.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


PRICE_COLUMNS = ["date", "movement", "high", "low", "close_norm", "close", "volume"]
PRICE_FEATURE_COLUMNS = ["movement", "high", "low", "close_norm", "close", "volume"]
NUM_PRICE_FEATURES = len(PRICE_FEATURE_COLUMNS)


def load_stocknet_prices(price_root: str | Path) -> Dict[str, pd.DataFrame]:
    """Return {ticker: DataFrame[date(asc), 6 features]}."""
    root = Path(price_root)
    out: Dict[str, pd.DataFrame] = {}
    for f in sorted(root.glob("*.txt")):
        df = pd.read_csv(f, sep="\t", header=None, names=PRICE_COLUMNS)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        out[f.stem] = df
    return out


def _tweet_texts_for_file(path: Path) -> List[str]:
    """Return one string per tweet in the file.

    For paper-faithfulness, list-of-token entries are stringified as `str(list)`
    (e.g. `"['$', 'aapl', '-', 'wall', ...]"`), matching the paper authors'
    `prompt_template.format(news_content=row['text'])` where `row['text']` is
    the python list.
    """
    texts: List[str] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            toks = obj.get("text") or []
            if isinstance(toks, list):
                texts.append(str(toks))
            elif isinstance(toks, str):
                texts.append(toks)
    return texts


def load_stocknet_tweets(
    tweet_root: str | Path, stocks: List[str], dates: List[pd.Timestamp]
) -> Dict[Tuple[str, str], List[str]]:
    """Return {(ticker, 'YYYY-MM-DD'): [joined-token strings]} for requested keys.

    Missing files return empty list (caller decides padding behavior).
    """
    root = Path(tweet_root)
    out: Dict[Tuple[str, str], List[str]] = {}
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]
    for ticker in stocks:
        for ds in date_strs:
            p = root / ticker / ds
            out[(ticker, ds)] = _tweet_texts_for_file(p) if p.exists() else []
    return out


def trading_day_index(price_dfs: Dict[str, pd.DataFrame]) -> List[pd.Timestamp]:
    """Intersection of trading dates across all stocks."""
    sets = [set(df["date"].tolist()) for df in price_dfs.values()]
    return sorted(set.intersection(*sets)) if sets else []

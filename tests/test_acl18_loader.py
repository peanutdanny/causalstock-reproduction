"""Phase 1 verification."""
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.data import build_acl18_splits
from src.data.stocknet import NUM_PRICE_FEATURES, load_stocknet_prices

ROOT = Path(__file__).resolve().parents[1]
PRICE_ROOT = ROOT / "reference_data/stocknet-dataset-master/price/preprocessed"
TWEET_ROOT = ROOT / "reference_data/stocknet-dataset-master/tweet/preprocessed"


def test_price_loader_shape():
    dfs = load_stocknet_prices(PRICE_ROOT)
    assert len(dfs) == 88  # paper §7.1
    for df in dfs.values():
        assert len(df) > 100
        assert df["date"].is_monotonic_increasing
        assert NUM_PRICE_FEATURES == 6


@pytest.fixture(scope="module")
def splits():
    return build_acl18_splits(
        price_root=PRICE_ROOT,
        tweet_root=TWEET_ROOT,
        train_range=("2014-01-02", "2015-08-02"),
        valid_range=("2015-08-03", "2015-09-30"),
        test_range=("2015-10-01", "2016-01-01"),
        lag_L=5,
    )


def test_split_sizes(splits):
    train, valid, test = splits
    # 2014-01-02..2015-08-02 ≈ 400 trading days minus lag_L
    assert len(train) > 300
    assert len(valid) > 30
    assert len(test) > 50


def test_no_date_leakage(splits):
    train, valid, test = splits
    train_max = max(train.samples)
    valid_min = min(valid.samples)
    test_min = min(test.samples)
    assert train_max < valid_min
    assert valid_min <= max(valid.samples) < test_min


def test_sample_shapes(splits):
    train, _, _ = splits
    s = train[0]
    D = len(train.stocks)
    assert s.P.shape == (D, 5, 6)
    assert s.C.shape == (D, 5, 10, 5)  # zeros, scorer is None
    assert s.y.shape == (D,)
    assert s.y.dtype == torch.long
    assert ((s.y == 0) | (s.y == 1)).all()


def test_stocks_count_after_intersection(splits):
    train, _, _ = splits
    # Paper claims 88; after dropping IPO-late tickers (GMRE/BABA/AGFS) we get
    # 85. Tweet dir has 87. Final intersection ≥ 84.
    assert len(train.stocks) >= 84

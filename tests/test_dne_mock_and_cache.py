import numpy as np
from pathlib import Path

from src.data import CachedScorer, DNECache, MockDNEScorer


def test_mock_is_deterministic():
    s = MockDNEScorer(news_per_day=10)
    a = s("AAPL", "2014-01-02", ["apple beats earnings", "buyback announced"])
    b = s("AAPL", "2014-01-02", ["apple beats earnings", "buyback announced"])
    assert np.array_equal(a, b)


def test_mock_shape_and_ranges():
    s = MockDNEScorer(news_per_day=10)
    arr = s("AAPL", "2014-01-02", ["x"] * 3)  # only 3 news → rest zero
    assert arr.shape == (10, 5)
    assert (arr[3:] == 0).all()
    nonzero = arr[:3]
    assert (nonzero[:, 0] >= 0).all() and (nonzero[:, 0] <= 10).all()  # Correlation
    assert (nonzero[:, 1] >= -1).all() and (nonzero[:, 1] <= 1).all()  # Sentiment
    assert (nonzero[:, 2] >= 0).all() and (nonzero[:, 2] <= 10).all()  # Importance
    assert (nonzero[:, 3] >= 0).all() and (nonzero[:, 3] <= 10).all()  # Impact
    assert (nonzero[:, 4] >= 0).all() and (nonzero[:, 4] <= 10).all()  # Duration


def test_mock_excess_news_truncated():
    s = MockDNEScorer(news_per_day=10)
    arr = s("AAPL", "2014-01-02", [f"news_{i}" for i in range(25)])
    assert arr.shape == (10, 5)
    # 11th news shouldn't influence score
    arr2 = s("AAPL", "2014-01-02", [f"news_{i}" for i in range(10)])
    assert np.array_equal(arr, arr2)


def test_cache_roundtrip(tmp_path: Path):
    cache_path = tmp_path / "dne.parquet"
    cache = DNECache(cache_path, news_per_day=10)
    base = MockDNEScorer(news_per_day=10)
    scorer = CachedScorer(base, cache)
    arr1 = scorer("AAPL", "2014-01-02", ["a", "b"])
    assert len(cache) == 1
    cache.save()

    # Reload
    cache2 = DNECache(cache_path, news_per_day=10)
    assert len(cache2) == 1
    arr2 = cache2.get("AAPL", "2014-01-02")
    np.testing.assert_array_equal(arr1, arr2)


def test_cache_skip_base_on_hit(tmp_path: Path):
    cache = DNECache(tmp_path / "c.parquet", news_per_day=10)
    calls = {"n": 0}

    def base(ticker, date_str, texts):
        calls["n"] += 1
        return np.ones((10, 5), dtype=np.float32)

    scorer = CachedScorer(base, cache)
    scorer("X", "2014-01-02", ["a"])
    scorer("X", "2014-01-02", ["a"])  # should not call base again
    assert calls["n"] == 1

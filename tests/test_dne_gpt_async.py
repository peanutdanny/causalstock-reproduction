"""AsyncGPTDNEScorer tests with a fake async client."""
import asyncio

import numpy as np

from src.data.dne_gpt_async import AsyncGPTDNEScorer


class _AsyncFakeChoiceMsg:
    def __init__(self, content):
        self.content = content


class _AsyncFakeChoice:
    def __init__(self, content):
        self.message = _AsyncFakeChoiceMsg(content)


class _AsyncFakeResp:
    def __init__(self, content):
        self.choices = [_AsyncFakeChoice(content)]


class _AsyncFakeChat:
    def __init__(self, response_text):
        self._text = response_text
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        await asyncio.sleep(0)
        return _AsyncFakeResp(self._text)


class _AsyncFakeCompletions:
    def __init__(self, response_text):
        self.completions = _AsyncFakeChat(response_text)


class _AsyncFakeClient:
    def __init__(self, response_text):
        self.chat = _AsyncFakeCompletions(response_text)


WELL_FORMED = "Correlation: 7\nSentiment: 0.5\nImportance: 6\nImpact: 5\nDuration: 4\n"


def test_async_score_pair_parallel_calls():
    client = _AsyncFakeClient(WELL_FORMED)
    sc = AsyncGPTDNEScorer({"AAPL": "Apple"}, news_per_day=10, concurrency=4, client=client)
    arr, ok = asyncio.run(sc.score_pair("AAPL", "2014-01-02", ["t1", "t2", "t3"]))
    assert ok is True
    assert arr.shape == (10, 5)
    assert client.chat.completions.calls == 3
    np.testing.assert_array_almost_equal(arr[0], [7.0, 0.5, 6.0, 5.0, 4.0])
    assert (arr[3:] == 0).all()


def test_async_score_pairs_multiple_pairs():
    client = _AsyncFakeClient(WELL_FORMED)
    sc = AsyncGPTDNEScorer({"AAPL": "Apple", "MSFT": "Microsoft"}, news_per_day=10, client=client)
    pairs = [
        ("AAPL", "2014-01-02", ["t1", "t2"]),
        ("MSFT", "2014-01-02", ["t1", "t2", "t3", "t4"]),
        ("AAPL", "2014-01-03", []),  # empty
    ]
    results = asyncio.run(sc.score_pairs(pairs))
    assert len(results) == 3
    arr0, ok0 = results[0]
    arr1, ok1 = results[1]
    arr2, ok2 = results[2]
    assert ok0 and ok1 and ok2
    assert arr0.shape == (10, 5)
    assert (arr0[:2] != 0).any()
    assert (arr1[:4] != 0).any()
    assert (arr2 == 0).all()
    assert client.chat.completions.calls == 6


def test_async_falls_back_to_zero_on_parse_failure():
    """Parse failure (model returns gibberish) → ok=True with zeros, after retries."""
    client = _AsyncFakeClient("gibberish")
    sc = AsyncGPTDNEScorer({"AAPL": "Apple"}, news_per_day=10, client=client)
    arr, ok = asyncio.run(sc.score_pair("AAPL", "2014-01-02", ["t1"]))
    assert ok is True  # model's fault — accept zero so we don't loop forever
    assert (arr[0] == 0).all()


def test_async_caps_at_news_per_day():
    client = _AsyncFakeClient(WELL_FORMED)
    sc = AsyncGPTDNEScorer({"AAPL": "Apple"}, news_per_day=3, client=client)
    arr, ok = asyncio.run(sc.score_pair("AAPL", "2014-01-02", ["t1", "t2", "t3", "t4", "t5"]))
    assert ok and arr.shape == (3, 5)
    assert client.chat.completions.calls == 3

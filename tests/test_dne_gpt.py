"""Phase 3b tests using a fake OpenAI client; real API is not exercised."""
import numpy as np
import pytest

from src.data.dne_gpt import GPTDNEScorer, parse_scores


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeChat:
    def __init__(self, response_text):
        self._text = response_text
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _FakeResp(self._text)


class _FakeCompletions:
    def __init__(self, response_text):
        self.completions = _FakeChat(response_text)


class _FakeClient:
    def __init__(self, response_text):
        self.chat = _FakeCompletions(response_text)


def test_parse_scores_well_formed():
    txt = (
        "Correlation: 8\nSentiment: -0.5\nImportance: 7.0\nImpact: 6\nDuration: 5\n"
    )
    arr = parse_scores(txt)
    assert arr is not None
    np.testing.assert_array_almost_equal(arr, [8.0, -0.5, 7.0, 6.0, 5.0])


def test_parse_scores_missing_field_returns_none():
    txt = "Correlation: 8\nSentiment: -0.5\nImpact: 6\nDuration: 5\n"
    assert parse_scores(txt) is None


def test_scorer_calls_api_per_news_and_returns_padded():
    client = _FakeClient("Correlation: 4\nSentiment: 0\nImportance: 3\nImpact: 2\nDuration: 1\n")
    sc = GPTDNEScorer(full_name_map={"AAPL": "Apple"}, news_per_day=10, client=client)
    arr = sc("AAPL", "2014-01-02", ["news a", "news b"])
    assert arr.shape == (10, 5)
    assert client.chat.completions.calls == 2
    # First two non-zero, rest zero.
    np.testing.assert_array_almost_equal(arr[0], [4.0, 0.0, 3.0, 2.0, 1.0])
    assert (arr[2:] == 0).all()


def test_scorer_handles_parse_failure_with_zero_fallback():
    client = _FakeClient("the model returned chatter instead of scores")
    sc = GPTDNEScorer(full_name_map={"AAPL": "Apple"}, news_per_day=10, client=client)
    # max_retries default 3 → still falls back to zero
    arr = sc("AAPL", "2014-01-02", ["news a"])
    assert (arr[0] == 0).all()

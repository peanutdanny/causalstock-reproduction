import math

import numpy as np
import torch

from src.evaluation import (
    accumulated_portfolio_value,
    accuracy,
    confusion,
    matthews_corrcoef,
    sharpe_ratio,
    top_k_portfolio_returns,
)


def test_acc_perfect():
    p = np.array([1, 0, 1, 0])
    t = np.array([1, 0, 1, 0])
    assert accuracy(p, t) == 1.0


def test_acc_half():
    p = np.array([1, 1, 0, 0])
    t = np.array([1, 0, 1, 0])
    assert accuracy(p, t) == 0.5


def test_mcc_perfect_and_worst():
    perfect = matthews_corrcoef(np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0]))
    worst = matthews_corrcoef(np.array([0, 1, 0, 1]), np.array([1, 0, 1, 0]))
    assert math.isclose(perfect, 1.0, abs_tol=1e-9)
    assert math.isclose(worst, -1.0, abs_tol=1e-9)


def test_mcc_matches_sklearn():
    from sklearn.metrics import matthews_corrcoef as sk_mcc
    rng = np.random.default_rng(0)
    p = rng.integers(0, 2, 200)
    t = rng.integers(0, 2, 200)
    assert math.isclose(matthews_corrcoef(p, t), sk_mcc(t, p), abs_tol=1e-9)


def test_confusion_counts():
    p = np.array([1, 1, 0, 0])
    t = np.array([1, 0, 1, 0])
    c = confusion(p, t)
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 1)


def test_apv_monotonic_on_positive_returns():
    apv = accumulated_portfolio_value([0.01, 0.02, 0.005])
    assert (np.diff(apv) > 0).all()
    assert math.isclose(apv[-1], (1.01 * 1.02 * 1.005), abs_tol=1e-9)


def test_sharpe_basic():
    r = np.array([0.01, 0.02, -0.01, 0.005])
    sr = sharpe_ratio(r)
    expected = r.mean() / r.std(ddof=1)
    assert math.isclose(sr, expected, abs_tol=1e-9)


def test_top_k_portfolio():
    # 3 stocks, 2 days. Day 0: pick stock 0 (highest prob), returns 0.05.
    probs = np.array([[0.9, 0.1, 0.0], [0.0, 0.0, 1.0]])
    rets = np.array([[0.05, -0.10, 0.03], [0.1, 0.2, -0.05]])
    pr = top_k_portfolio_returns(probs, rets, k=1)
    assert math.isclose(pr[0], 0.05)
    assert math.isclose(pr[1], -0.05)


def test_torch_input_accepted():
    p = torch.tensor([1, 0, 1, 0])
    t = torch.tensor([1, 0, 1, 0])
    assert accuracy(p, t) == 1.0

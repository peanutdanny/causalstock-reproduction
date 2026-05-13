"""Smoke test: end-to-end forward + backward on a tiny subset, 1 epoch."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import (
    CachedScorer,
    DNECache,
    MockDNEScorer,
    build_acl18_splits,
    collate_samples,
)
from src.data.stocknet import NUM_PRICE_FEATURES
from src.models import CausalStockModel
from src.training import CausalStockLoss, Trainer
from src.utils import set_global_seed

ROOT = Path(__file__).resolve().parents[1]


def test_trainer_one_epoch(tmp_path: Path):
    set_global_seed(0)
    tiny = ["AAPL", "AMZN", "BA", "BAC", "C", "CMCSA", "CVX", "DIS"]
    cache = DNECache(tmp_path / "dne.parquet", news_per_day=10)
    scorer = CachedScorer(MockDNEScorer(news_per_day=10), cache)
    train, valid, test = build_acl18_splits(
        price_root=ROOT / "reference_data/stocknet-dataset-master/price/preprocessed",
        tweet_root=ROOT / "reference_data/stocknet-dataset-master/tweet/preprocessed",
        train_range=("2014-01-02", "2014-04-01"),
        valid_range=("2014-04-02", "2014-05-01"),
        test_range=("2014-05-02", "2014-05-15"),
        lag_L=5,
        news_scorer=scorer,
        only_tickers=tiny,
    )
    D = len(train.stocks)
    assert D == 8

    loader = lambda ds: DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_samples)
    model = CausalStockModel(
        D=D, L=5, price_in_dim=NUM_PRICE_FEATURES, d_p=4, d_m=64, hidden=32, h_uv_layers=1
    )
    loss = CausalStockLoss(bce_weight=0.01)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model, loss, optim,
        log_dir=tmp_path / "logs",
        checkpoint_dir=tmp_path / "ckpt",
    )

    initial_acc, _, _, _ = trainer.evaluate(loader(valid))
    summary = trainer.fit(loader(train), loader(valid), max_epochs=2, patience=10)
    assert summary["best_epoch"] >= 1
    test_acc, _, _, _ = trainer.evaluate(loader(test))
    # Smoke: predictions are in {0,1}, no NaN, file artifacts exist.
    assert 0.0 <= test_acc <= 1.0
    assert (tmp_path / "logs/history.json").exists()
    assert (tmp_path / "ckpt/best.pt").exists()

"""Phase 9.5 sanity sweep — does the architecture learn at all on mock DNE?

We try (lr, likelihood_form) combinations on the price-only ablation (so mock
news noise is removed) for 30 epochs each. If even the strongest setting
plateaus at chance, the architecture has a bug that needs fixing before Phase 10.

Reference upper bound: paper's `w/o news` ablation reaches ~58% on real ACL18
(docs/expected-results.md Table 2). Mock data should still permit learning from
the price signal — we'd expect ≥52% if the architecture is wired correctly.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


SWEEP = [
    # (label, lr, likelihood_form, use_news)
    ("price_only_lr1e-3_bern", 1e-3, "bernoulli", False),
    ("price_only_lr1e-4_bern", 1e-4, "bernoulli", False),
    ("price_only_lr1e-5_gauss", 1e-5, "gaussian", False),  # paper-exact
    ("full_lr1e-3_bern",       1e-3, "bernoulli", True),
]


def _make_loaders():
    cache = DNECache(ROOT / "data/processed/dne_mock_acl18.parquet", news_per_day=10)
    scorer = CachedScorer(MockDNEScorer(news_per_day=10), cache)
    train, valid, test = build_acl18_splits(
        price_root=ROOT / "reference_data/stocknet-dataset-master/price/preprocessed",
        tweet_root=ROOT / "reference_data/stocknet-dataset-master/tweet/preprocessed",
        train_range=("2014-01-02", "2015-08-02"),
        valid_range=("2015-08-03", "2015-09-30"),
        test_range=("2015-10-01", "2016-01-01"),
        lag_L=5,
        news_scorer=scorer,
    )
    cache.save()
    L = lambda ds, sh: DataLoader(ds, batch_size=32, shuffle=sh, collate_fn=collate_samples)
    return train, L(train, True), L(valid, False), L(test, False)


def run_one(label, lr, form, use_news, train_ds, train_loader, val_loader, test_loader, log_dir):
    set_global_seed(0)
    D = len(train_ds.stocks)
    model = CausalStockModel(
        D=D, L=5, price_in_dim=NUM_PRICE_FEATURES,
        d_p=4, d_m=64, hidden=332,
        use_tcd=True, use_news=use_news, lag_dep=True, h_uv_layers=1,
    )
    loss = CausalStockLoss(bce_weight=0.01, lambda_s=1.0, likelihood_form=form)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(
        model, loss, optim,
        log_dir=log_dir / label,
        checkpoint_dir=log_dir / label / "ckpt",
    )
    t0 = time.time()
    summary = trainer.fit(train_loader, val_loader, max_epochs=30, patience=15)
    trainer.load("best.pt")
    test_acc, test_mcc, _, _ = trainer.evaluate(test_loader)
    return {
        "label": label,
        "lr": lr,
        "form": form,
        "use_news": use_news,
        "best_epoch": summary["best_epoch"],
        "best_val_acc": summary["best_val_acc"],
        "test_acc": test_acc,
        "test_mcc": test_mcc,
        "wall_clock_sec": time.time() - t0,
    }


def main():
    log_dir = ROOT / "experiments/logs/sanity_sweep"
    log_dir.mkdir(parents=True, exist_ok=True)
    train_ds, train_loader, val_loader, test_loader = _make_loaders()
    results = []
    for label, lr, form, use_news in SWEEP:
        print(f"\n>>> {label}", flush=True)
        r = run_one(label, lr, form, use_news, train_ds, train_loader, val_loader, test_loader, log_dir)
        print(json.dumps(r, indent=2))
        results.append(r)
    out_path = ROOT / "experiments/results/sanity_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nresults → {out_path}")


if __name__ == "__main__":
    main()

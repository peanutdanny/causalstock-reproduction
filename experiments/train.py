"""Run training from a YAML config.

Usage:
    .venv/bin/python -m experiments.train --config experiments/configs/acl18.yaml
    .venv/bin/python -m experiments.train --config experiments/configs/acl18.yaml \
        --max-epochs 2 --tiny  # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
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
from src.utils import get_logger, load_yaml_config, set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--tiny", action="store_true", help="Subset to first 8 stocks, 1 epoch")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.train.seed)
    set_global_seed(seed)
    logger = get_logger("experiments.train")

    # ---- Data --------------------------------------------------------------
    only = None
    if args.tiny:
        only = ["AAPL", "AMZN", "BA", "BAC", "C", "CMCSA", "CVX", "DIS"]

    cache_path = ROOT / "data/processed/dne_mock_acl18.parquet"
    cache = DNECache(cache_path, news_per_day=int(cfg.data.news_per_day))
    base_scorer = MockDNEScorer(news_per_day=int(cfg.data.news_per_day))
    scorer = CachedScorer(base_scorer, cache) if cfg.data.use_news else None

    train_ds, valid_ds, test_ds = build_acl18_splits(
        price_root=ROOT / cfg.data.price_root,
        tweet_root=ROOT / cfg.data.tweet_root,
        train_range=(cfg.data.train_start, cfg.data.train_end),
        valid_range=(cfg.data.valid_start, cfg.data.valid_end),
        test_range=(cfg.data.test_start, cfg.data.test_end),
        lag_L=int(cfg.data.lag_L),
        news_per_day=int(cfg.data.news_per_day),
        news_scorer=scorer,
        only_tickers=only,
    )
    cache.save()

    logger.info(f"train={len(train_ds)} valid={len(valid_ds)} test={len(test_ds)} D={len(train_ds.stocks)}")

    train_loader = DataLoader(
        train_ds, batch_size=int(cfg.train.batch_size), shuffle=True,
        collate_fn=collate_samples, num_workers=int(cfg.runtime.num_workers),
    )
    val_loader = DataLoader(
        valid_ds, batch_size=int(cfg.train.batch_size), shuffle=False,
        collate_fn=collate_samples, num_workers=int(cfg.runtime.num_workers),
    )
    test_loader = DataLoader(
        test_ds, batch_size=int(cfg.train.batch_size), shuffle=False,
        collate_fn=collate_samples, num_workers=int(cfg.runtime.num_workers),
    )

    # ---- Model -------------------------------------------------------------
    model = CausalStockModel(
        D=len(train_ds.stocks),
        L=int(cfg.data.lag_L),
        price_in_dim=NUM_PRICE_FEATURES,
        d_p=int(cfg.model.d_p),
        d_m=int(cfg.model.d_m),
        hidden=int(cfg.model.hidden_size),
        use_tcd=bool(cfg.model.use_tcd),
        use_news=bool(cfg.data.use_news),
        lag_dep=bool(cfg.model.lag_dep),
        h_uv_layers=int(cfg.model.h_uv_layers),
        gumbel_tau=float(cfg.model.gumbel_tau),
    )
    loss = CausalStockLoss(
        bce_weight=float(cfg.loss.bce_weight),
        lambda_s=float(cfg.model.graph_prior_lambda_s),
        lambda_d=float(cfg.model.graph_prior_lambda_d),
    )
    optim = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))

    trainer = Trainer(
        model=model,
        loss=loss,
        optimizer=optim,
        device=str(cfg.runtime.device),
        log_dir=ROOT / cfg.runtime.log_dir,
        checkpoint_dir=ROOT / cfg.runtime.checkpoint_dir,
    )

    max_epochs = args.max_epochs or int(cfg.train.max_epochs)
    if args.tiny:
        max_epochs = min(max_epochs, 1)

    summary = trainer.fit(
        train_loader, val_loader,
        max_epochs=max_epochs,
        patience=int(cfg.train.early_stop_patience),
    )

    # ---- Test --------------------------------------------------------------
    if (trainer.ckpt_dir / "best.pt").exists():
        trainer.load("best.pt")
    test_acc, test_mcc, _, _ = trainer.evaluate(test_loader)
    summary["test_acc"] = test_acc
    summary["test_mcc"] = test_mcc
    logger.info(f"test_acc={test_acc:.4f} test_mcc={test_mcc:.4f}")

    results_dir = ROOT / "experiments/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{cfg.experiment_name}_seed{seed}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info(f"results saved to {out_path}")


if __name__ == "__main__":
    main()

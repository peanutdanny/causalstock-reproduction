"""Test-set backtest: checkpoint → (probs, realized_returns) per (day, stock).

Produces an NPZ file consumed by `scripts/make_plots.py` to draw Figure 4
(APV curve + Sharpe bar). Independent of training so we don't have to
re-train just to plot.

Usage:
    .venv/bin/python scripts/run_backtest.py \
        --config experiments/configs/acl18.yaml \
        --checkpoint experiments/checkpoints/acl18/best.pt \
        --out experiments/results/backtest_acl18.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
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
from src.utils import load_yaml_config, set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None,
                   help="Override cfg.runtime.device (cpu/cuda/mps/auto).")
    return p.parse_args()


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_test_loader(cfg):
    """Mirror experiments/train.py exactly so checkpoint shapes match."""
    score_cache_path = getattr(cfg.data, "dne_cache_path", None)
    score_news_per_day = int(getattr(cfg.data, "score_news_per_day", 20))
    if score_cache_path and (ROOT / score_cache_path).exists():
        cache = DNECache(ROOT / score_cache_path, news_per_day=score_news_per_day)
        scorer = CachedScorer(MockDNEScorer(news_per_day=score_news_per_day), cache) if cfg.data.use_news else None
    else:
        cache_path = ROOT / "data/processed/dne_mock_acl18.parquet"
        cache = DNECache(cache_path, news_per_day=int(cfg.data.news_per_day))
        scorer = CachedScorer(MockDNEScorer(news_per_day=int(cfg.data.news_per_day)), cache) if cfg.data.use_news else None

    _train, _valid, test_ds = build_acl18_splits(
        price_root=ROOT / cfg.data.price_root,
        tweet_root=ROOT / cfg.data.tweet_root,
        train_range=(cfg.data.train_start, cfg.data.train_end),
        valid_range=(cfg.data.valid_start, cfg.data.valid_end),
        test_range=(cfg.data.test_start, cfg.data.test_end),
        lag_L=int(cfg.data.lag_L),
        news_per_day=int(cfg.data.news_per_day),
        news_scorer=scorer,
    )
    loader = DataLoader(
        test_ds, batch_size=int(cfg.train.batch_size), shuffle=False,
        collate_fn=collate_samples, num_workers=int(cfg.runtime.num_workers),
    )
    return test_ds, loader


def _build_model(cfg, D: int) -> CausalStockModel:
    return CausalStockModel(
        D=D,
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


def _realized_returns_for_target(test_ds, target_date) -> np.ndarray:
    """r_i for day T = StockNet `movement` (daily % change of raw close).

    The `close` column in stocknet preprocessed TSVs is z-score-normalized
    (mean≈0, std~1.5) and not suitable as a price for r = close_T/close_{T-1}-1.
    Use `movement` directly: this is the per-row return (close_raw_T -
    close_raw_{T-1}) / close_raw_{T-1}.
    """
    rets = np.zeros(test_ds.D, dtype=np.float64)
    if target_date not in test_ds._date_to_idx:
        return rets
    for i, s in enumerate(test_ds.stocks):
        df = test_ds.price_dfs[s]
        rows = test_ds._stock_date_to_row[s]
        r_T = rows.get(target_date)
        if r_T is None:
            continue
        rets[i] = float(df["movement"].iat[r_T])
    return rets


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.train.seed)
    set_global_seed(seed)

    device = _resolve_device(args.device if args.device else str(cfg.runtime.device))

    test_ds, loader = _build_test_loader(cfg)
    D = len(test_ds.stocks)
    if D == 0 or len(test_ds) == 0:
        raise RuntimeError(f"Empty test set: D={D}, len={len(test_ds)}")

    model = _build_model(cfg, D=D).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    probs_rows, ret_rows, y_rows, dates = [], [], [], []
    for batch in loader:
        P = batch["P"].to(device)
        C = batch["C"].to(device)
        out = model(P, C)
        f_i = out.f_i.detach().cpu().numpy()       # (B, D)
        y = batch["y"].numpy()                     # (B, D)
        for b, td in enumerate(batch["target_dates"]):
            probs_rows.append(f_i[b])
            y_rows.append(y[b])
            ret_rows.append(_realized_returns_for_target(test_ds, td))
            dates.append(str(td.date()))

    probs = np.stack(probs_rows, axis=0).astype(np.float32)
    returns = np.stack(ret_rows, axis=0).astype(np.float32)
    y_all = np.stack(y_rows, axis=0).astype(np.int8)
    dates_arr = np.array(dates)
    stocks_arr = np.array(list(test_ds.stocks))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        probs=probs,
        returns=returns,
        y=y_all,
        dates=dates_arr,
        stocks=stocks_arr,
        config=str(args.config),
        checkpoint=str(args.checkpoint),
    )
    print(f"saved backtest → {out_path}")
    print(f"  shape: probs={probs.shape}  returns={returns.shape}")
    print(f"  date range: {dates_arr[0]} .. {dates_arr[-1]}  ({D} stocks)")
    print(f"  mean realized |r|={np.abs(returns).mean():.4f}  "
          f"empirical up rate={y_all.mean():.4f}")


if __name__ == "__main__":
    main()

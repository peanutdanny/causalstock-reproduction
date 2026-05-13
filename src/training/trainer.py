"""Training loop with early stopping on validation accuracy.

Hyperparameters from docs/paper-summary.md §10 (Appendix C.4):
    Adam, lr=1e-5, batch=32, max 100 epochs, patience 10.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation import accuracy, matthews_corrcoef
from src.models import CausalStockModel
from src.training.loss import CausalStockLoss
from src.utils import get_logger


@dataclass
class TrainStats:
    epoch: int
    train_loss: float
    val_acc: float
    val_mcc: float
    elapsed_sec: float


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available backend: cuda → mps → cpu."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Trainer:
    def __init__(
        self,
        model: CausalStockModel,
        loss: CausalStockLoss,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        log_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        device = _resolve_device(device)
        self.model = model.to(device)
        self.loss = loss.to(device)
        self.optim = optimizer
        self.device = device
        self.log_dir = Path(log_dir) if log_dir else None
        self.ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.logger = get_logger(
            "trainer",
            log_file=(self.log_dir / "trainer.log") if self.log_dir else None,
        )
        if self.ckpt_dir:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[TrainStats] = []

    def _batch_to_device(self, batch):
        return (
            batch["P"].to(self.device),
            batch["C"].to(self.device),
            batch["y"].to(self.device),
        )

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total, n = 0.0, 0
        for batch in loader:
            P, C, y = self._batch_to_device(batch)
            out = self.model(P, C)
            lo = self.loss(
                f_i=out.f_i,
                y=y,
                sigma_q=out.sigma,
                G_sample=out.G_sample,
                sigma_noise=self.model.fcm.sigma_noise,
            )
            self.optim.zero_grad()
            lo.total.backward()
            self.optim.step()
            total += float(lo.total.item()) * P.shape[0]
            n += P.shape[0]
        return total / max(n, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float, np.ndarray, np.ndarray]:
        self.model.eval()
        preds, targets = [], []
        for batch in loader:
            P, C, y = self._batch_to_device(batch)
            out = self.model(P, C)
            preds.append((out.f_i > 0.5).long().cpu().numpy())
            targets.append(y.cpu().numpy())
        P_all = np.concatenate(preds, axis=0)
        T_all = np.concatenate(targets, axis=0)
        return accuracy(P_all, T_all), matthews_corrcoef(P_all, T_all), P_all, T_all

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100,
        patience: int = 10,
    ) -> dict:
        best_acc = -1.0
        best_epoch = -1
        bad_epochs = 0
        t0 = time.time()
        for epoch in range(1, max_epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_acc, val_mcc, _, _ = self.evaluate(val_loader)
            stats = TrainStats(
                epoch=epoch,
                train_loss=train_loss,
                val_acc=val_acc,
                val_mcc=val_mcc,
                elapsed_sec=time.time() - t0,
            )
            self.history.append(stats)
            self.logger.info(
                f"epoch={epoch:03d} train_loss={train_loss:.4f} "
                f"val_acc={val_acc:.4f} val_mcc={val_mcc:.4f} "
                f"elapsed={stats.elapsed_sec:.0f}s"
            )
            if val_acc > best_acc:
                best_acc, best_epoch, bad_epochs = val_acc, epoch, 0
                if self.ckpt_dir:
                    self.save("best.pt")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    self.logger.info(f"early stop at epoch {epoch} (best={best_epoch})")
                    break
        summary = {
            "best_epoch": best_epoch,
            "best_val_acc": best_acc,
            "history": [asdict(s) for s in self.history],
        }
        if self.log_dir:
            (self.log_dir / "history.json").write_text(json.dumps(summary, indent=2))
        return summary

    def save(self, name: str) -> Path:
        assert self.ckpt_dir is not None
        path = self.ckpt_dir / name
        torch.save({"model": self.model.state_dict(), "optim": self.optim.state_dict()}, path)
        return path

    def load(self, name: str) -> None:
        assert self.ckpt_dir is not None
        ckpt = torch.load(self.ckpt_dir / name, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optim.load_state_dict(ckpt["optim"])

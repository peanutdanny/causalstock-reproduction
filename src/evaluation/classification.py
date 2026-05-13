"""Classification metrics (Section 8.1 of docs/paper-summary.md).

Eq. 16 (corrected — paper's "fp + gn" is a typo per docs/paper-summary line 274):

    MCC = (tp·tn - fp·fn) / sqrt( (tp+fp)·(fn+tp)·(fn+tn)·(fp+tn) )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int


def _as_numpy(t) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def confusion(pred: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> Confusion:
    p = _as_numpy(pred).astype(np.int64).reshape(-1)
    t = _as_numpy(target).astype(np.int64).reshape(-1)
    tp = int(((p == 1) & (t == 1)).sum())
    fp = int(((p == 1) & (t == 0)).sum())
    fn = int(((p == 0) & (t == 1)).sum())
    tn = int(((p == 0) & (t == 0)).sum())
    return Confusion(tp, fp, fn, tn)


def accuracy(pred, target) -> float:
    c = confusion(pred, target)
    total = c.tp + c.fp + c.fn + c.tn
    return (c.tp + c.tn) / total if total else 0.0


def matthews_corrcoef(pred, target) -> float:
    c = confusion(pred, target)
    num = c.tp * c.tn - c.fp * c.fn
    denom_sq = (
        (c.tp + c.fp)
        * (c.fn + c.tp)
        * (c.fn + c.tn)
        * (c.fp + c.tn)
    )
    if denom_sq <= 0:
        return 0.0
    return num / float(np.sqrt(denom_sq))

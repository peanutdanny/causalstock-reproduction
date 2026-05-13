"""Global RNG seeding per CLAUDE.md reproducibility rule."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, *, deterministic_cudnn: bool = True) -> None:
    """Seed all RNGs used in this project.

    Args:
        seed: Integer seed.
        deterministic_cudnn: If True and CUDA is available, force cuDNN to
            deterministic mode. Trades throughput for reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

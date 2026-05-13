"""Logging helper: INFO to stdout, DEBUG to file (per CLAUDE.md convention)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_logger(name: str, *, log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(logging.INFO)
    stream.setFormatter(logging.Formatter(_FMT))
    logger.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(fh)
    logger.propagate = False
    return logger

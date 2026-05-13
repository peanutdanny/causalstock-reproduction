"""Run all 4 ablation configs sequentially (one seed each).

This validates that each ablation flag wires through to the model/loss.
Full reproduction (10 seeds, 100 epochs) is Phase 10 territory.

Usage:
    .venv/bin/python -m experiments.run_ablations --max-epochs 5
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ABLATIONS = [
    "experiments/configs/ablations/no_tcd.yaml",
    "experiments/configs/ablations/no_news.yaml",
    "experiments/configs/ablations/no_lag_dep.yaml",
    "experiments/configs/ablations/lambda_0.yaml",
    "experiments/configs/acl18.yaml",  # full reference
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-epochs", type=int, default=5)
    p.add_argument("--tiny", action="store_true")
    args = p.parse_args()

    for cfg in ABLATIONS:
        cmd = [
            sys.executable, "-m", "experiments.train",
            "--config", cfg,
            "--max-epochs", str(args.max_epochs),
        ]
        if args.tiny:
            cmd.append("--tiny")
        print(f"\n>>> Running {cfg}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

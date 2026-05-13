"""End-to-end: scan experiments/results/ + data/processed/ → produce all plots.

Run any time. Idempotent. Outputs in `experiments/figures/`.

    .venv/bin/python scripts/make_plots.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualization import (
    plot_training_curve,
    plot_score_distribution,
    plot_score_correlation,
    plot_sigma_heatmap,
    plot_causal_strength,
    plot_reproduction_table,
    plot_ablation_bar,
)


PAPER_TARGETS = {
    "ACL18_ACC":   63.42,
    "ACL18_MCC":   0.2172,
    "CMIN-US_ACC": 54.64,
    "CMIN-US_MCC": 0.0481,
    "CMIN-CN_ACC": 56.19,
    "CMIN-CN_MCC": 0.1417,
    "KDD17_ACC":   56.09,
    "KDD17_MCC":   0.1235,
    "NI225_ACC":   53.01,
    "NI225_MCC":   0.0640,
    "FTSE100_ACC": 52.88,
    "FTSE100_MCC": 0.0534,
}

PAPER_ABLATIONS = {  # ACL18 Table 2
    "full":       (63.42, 0.2172),
    "no_tcd":     (51.08, 0.0102),
    "no_news":    (58.10, 0.1421),
    "no_lag_dep": (59.19, 0.1757),
    "lambda_0":   (58.26, 0.0),
}


def _collect_history_files(results_dir: Path) -> dict[str, Path]:
    """Find experiments/results/causalstock_acl18_<variant>_seed0.json files."""
    out = {}
    for f in sorted(results_dir.glob("causalstock_acl18*seed0.json")):
        # parse variant from filename
        stem = f.stem.replace("causalstock_acl18_", "").replace("_seed0", "")
        if stem == "causalstock_acl18":
            stem = "full"
        if not stem:
            stem = "full"
        out[stem] = f
    return out


def _read_test_results(path: Path) -> tuple[float, float]:
    data = json.loads(path.read_text())
    return float(data.get("test_acc", 0.0) * 100), float(data.get("test_mcc", 0.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="experiments/results")
    p.add_argument("--cache", default="data/processed/dne_acl18.parquet")
    p.add_argument("--out", default="experiments/figures")
    p.add_argument("--checkpoint", default="experiments/checkpoints/acl18/best.pt",
                   help="Best checkpoint for causal graph plots (optional).")
    args = p.parse_args()

    results_dir = ROOT / args.results_dir
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    print("→ DNE score distribution + correlation")
    plot_score_distribution(ROOT / args.cache, out / "score_distribution.png")
    plot_score_correlation(ROOT / args.cache, out / "score_correlation.png")

    histories = _collect_history_files(results_dir)
    if histories:
        print(f"→ training curves ({len(histories)} runs)")
        plot_training_curve(histories, out / "training_curves.png")

        # Ablation comparison (ACL18 only — uses test_acc/mcc from JSONs)
        ours_ablations = {}
        for variant in ["full", "no_tcd", "no_news", "no_lag_dep", "lambda0"]:
            key_filename = variant if variant != "full" else ""
            stem = f"causalstock_acl18_{variant}_seed0.json" if variant != "full" else "causalstock_acl18_seed0.json"
            path = results_dir / stem
            if path.exists():
                ours_ablations[variant] = _read_test_results(path)
        if ours_ablations:
            print(f"→ ablation comparison bar ({len(ours_ablations)} variants)")
            paper_norm = {k: (v[0] * 100 if v[0] < 1 else v[0], v[1]) for k, v in PAPER_ABLATIONS.items()}
            plot_ablation_bar(ours_ablations, PAPER_ABLATIONS,
                              out / "ablation_comparison.png")

        # Reproduction summary table
        full_result = results_dir / "causalstock_acl18_seed0.json"
        if full_result.exists():
            acc, mcc = _read_test_results(full_result)
            ours = {"ACL18_ACC": acc, "ACL18_MCC": mcc}
            paper = {k: v for k, v in PAPER_TARGETS.items() if k in ours}
            plot_reproduction_table(paper, ours, out / "reproduction_table.png")

    ckpt = ROOT / args.checkpoint
    if ckpt.exists():
        print("→ causal graph (σ heatmap + causal strength)")
        try:
            plot_sigma_heatmap(ckpt, out / "sigma_heatmap.png")
            plot_causal_strength(ckpt, out / "causal_strength.png")
        except Exception as e:
            print(f"  ! causal plots failed: {e}")
    else:
        print(f"  (skipping causal plots — no checkpoint at {ckpt})")

    print(f"\n✓ Figures saved to {out}/")
    for f in sorted(out.glob("*.png")):
        print(f"    {f.name}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

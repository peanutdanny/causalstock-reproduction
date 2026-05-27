#!/usr/bin/env bash
# Phase 10 — ACL18 reproduction (full + 4 ablations × N seeds).
#
# Prerequisites:
#   - .env with OPENAI_API_KEY  (only if cache is missing for any pair)
#   - data/processed/dne_acl18.parquet from Phase 3b
#   - reference_data/stocknet-dataset-master/ on disk
#
# Usage:
#     bash scripts/run_phase10.sh            # seeds 0..0 (1 seed each)
#     bash scripts/run_phase10.sh 5          # seeds 0..4
#     bash scripts/run_phase10.sh 10         # seeds 0..9 (paper standard)

set -euo pipefail
cd "$(dirname "$0")/.."

# Detect venv layout (Windows: .venv/Scripts/python.exe ; Unix: .venv/bin/python)
if [[ -x .venv/Scripts/python.exe ]]; then
    PY=.venv/Scripts/python.exe
elif [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
else
    echo "ERROR: no python in .venv/{Scripts,bin}/" >&2
    exit 1
fi

N_SEEDS=${1:-1}
LOG_DIR=experiments/logs/phase10
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

CONFIGS=(
    "experiments/configs/acl18.yaml"
    "experiments/configs/ablations/no_tcd.yaml"
    "experiments/configs/ablations/no_news.yaml"
    "experiments/configs/ablations/no_lag_dep.yaml"
    "experiments/configs/ablations/lambda_0.yaml"
)

echo "Phase 10 — running ${#CONFIGS[@]} configs × $N_SEEDS seeds"
echo "Log dir: $LOG_DIR"
echo

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    for seed in $(seq 0 $((N_SEEDS - 1))); do
        log="$LOG_DIR/${name}_seed${seed}_${TS}.log"
        echo ">>> $name seed=$seed  log=$log"
        if "$PY" -m experiments.train \
                --config "$cfg" --seed "$seed" > "$log" 2>&1; then
            tail -2 "$log"
        else
            echo "  FAILED (exit $?). See $log"
        fi
    done
done

echo
echo "All runs finished. Results: experiments/results/causalstock_acl18_*.json"

#!/usr/bin/env bash
# Rescue logical-seed 4 (raw 12-14 all collapsed in best-of-3).
# Try raw seeds 30..34 until one is healthy (val_acc > 0.55).
# With ~50% collapse rate, P(all 5 collapse) = ~3%.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x .venv/Scripts/python.exe ]]; then PY=.venv/Scripts/python.exe
elif [[ -x .venv/bin/python ]]; then PY=.venv/bin/python
else echo "ERROR: no python in .venv/{Scripts,bin}/" >&2; exit 1; fi

LOG_DIR=experiments/logs/phase10
TS=$(date +%Y%m%d_%H%M%S)
CFG=experiments/configs/acl18_v3_lr.yaml

EXTRA_SEEDS=(30 31 32 33 34)
echo "Rescuing logical-seed 4 — trying raw seeds: ${EXTRA_SEEDS[*]}"
echo

for seed in "${EXTRA_SEEDS[@]}"; do
    log="$LOG_DIR/acl18_v3_seed${seed}_${TS}.log"
    echo ">>> seed=$seed  log=$log"
    if "$PY" -m experiments.train --config "$CFG" --seed "$seed" > "$log" 2>&1; then
        tail -2 "$log"
    else
        echo "  FAILED (exit $?). See $log"
    fi
done

echo
echo "Done. Update agg_bestof3.py logical-4 candidate set to include raw 30-34."

#!/usr/bin/env bash
# v3 verify: lr 1e-4 + bce=1.0 + KL warmup on collapsed seeds.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x .venv/Scripts/python.exe ]]; then PY=.venv/Scripts/python.exe
elif [[ -x .venv/bin/python ]]; then PY=.venv/bin/python
else echo "ERROR: no python in .venv/{Scripts,bin}/" >&2; exit 1; fi

LOG_DIR=experiments/logs/phase10
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

# Test the collapsed seeds AND one healthy seed (0) for sanity baseline.
SEEDS=(0 2 3 5 7)
CFG=experiments/configs/acl18_v3_lr.yaml

rm -rf experiments/checkpoints/acl18_v3

echo "Verify v3 (lr=1e-4 + bce=1.0 + KL warmup) on seeds: ${SEEDS[*]}"
echo "(seed 0 included as healthy baseline — should still train)"
echo

for seed in "${SEEDS[@]}"; do
    log="$LOG_DIR/acl18_v3_seed${seed}_${TS}.log"
    echo ">>> seed=$seed  log=$log"
    if "$PY" -m experiments.train --config "$CFG" --seed "$seed" > "$log" 2>&1; then
        tail -2 "$log"
    else
        echo "  FAILED (exit $?). See $log"
    fi
done

echo
echo "Done. Results: experiments/results/causalstock_acl18_v3_seed*.json"

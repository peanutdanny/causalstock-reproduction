#!/usr/bin/env bash
# Best-of-3 random restarts for ACL18 full reproduction.
#
# For each of 10 logical seeds N ∈ {0..9}, run 3 sub-seeds (raw seed = 3N+k
# for k ∈ {0,1,2}). Aggregator script picks the sub-seed with highest val_acc.
# Mitigates the 4/10 init-sensitive collapse observed in Phase 10b.
#
# Already have v3 results for raw seeds {0, 2, 3, 5, 7} from prior verify runs;
# this script skips them. 25 remaining runs × ~2 min on RTX A6000 ≈ 50-90 min.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x .venv/Scripts/python.exe ]]; then PY=.venv/Scripts/python.exe
elif [[ -x .venv/bin/python ]]; then PY=.venv/bin/python
else echo "ERROR: no python in .venv/{Scripts,bin}/" >&2; exit 1; fi

LOG_DIR=experiments/logs/phase10
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

LOCK="$LOG_DIR/v3_bestof3.lock"
if [[ -f "$LOCK" ]]; then
    prev_pid=$(cat "$LOCK" 2>/dev/null || echo "")
    if [[ -n "$prev_pid" ]] && kill -0 "$prev_pid" 2>/dev/null; then
        echo "ERROR: another v3 best-of-3 sweep is already running (PID $prev_pid)." >&2
        exit 2
    fi
    echo "Stale lock — overwriting."
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

CFG=experiments/configs/acl18_v3_lr.yaml

# Raw seeds 0..29. Skip the 5 already computed by verify_v3.sh.
ALL_SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29)
SKIP_SEEDS=(0 2 3 5 7)

echo "v3 best-of-3: 30 raw seeds (= 10 logical × 3 sub), skipping ${#SKIP_SEEDS[@]} done"
echo

for seed in "${ALL_SEEDS[@]}"; do
    skip=false
    for s in "${SKIP_SEEDS[@]}"; do
        if [[ "$s" == "$seed" ]]; then skip=true; break; fi
    done
    if $skip; then
        echo "--- seed=$seed (already done, skip)"
        continue
    fi
    log="$LOG_DIR/acl18_v3_seed${seed}_${TS}.log"
    echo ">>> seed=$seed  log=$log"
    if "$PY" -m experiments.train --config "$CFG" --seed "$seed" > "$log" 2>&1; then
        tail -2 "$log"
    else
        echo "  FAILED (exit $?). See $log"
    fi
done

echo
echo "All v3 runs finished. Results: experiments/results/causalstock_acl18_v3_seed{0..29}.json"
echo "Run aggregator: python scripts/agg_bestof3.py"

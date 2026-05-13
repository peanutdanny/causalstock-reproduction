#!/usr/bin/env bash
# Phase C — overnight-safe ACL18 DNE scoring.
#
# Robustness layers:
#   1. nohup    : survives terminal close
#   2. caffeinate -i : prevents Mac sleep (display can sleep, system stays awake)
#   3. Cache    : saved after every batch; restart safely picks up where it left off
#   4. Code     : network errors do NOT cache zero, they wait & retry until the
#                 connection is back (see src/data/dne_gpt_async.py)
#
# Usage:
#     bash scripts/run_scoring_overnight.sh
#
# Re-runnable: if it dies for any reason, simply re-execute. Cached pairs are
# skipped automatically.

set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=experiments/logs/scoring
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/scoring_${TS}.log"

echo "Starting scoring run. Log: $LOG"
echo "Use: tail -f $LOG  to watch progress"
echo "Stop: pkill -f score_news_async.py"
echo

nohup caffeinate -i .venv/bin/python scripts/score_news_async.py \
    --gpt \
    --cache data/processed/dne_acl18.parquet \
    --batch 200 \
    --concurrency 30 \
    --model gpt-5.4-mini \
    --news-per-day 20 \
    > "$LOG" 2>&1 &

PID=$!
echo "Background PID: $PID"
echo "$PID" > "$LOG_DIR/latest.pid"
echo
echo "Sleep prevented while this process is alive."
echo "It is safe to: close the terminal, disconnect, sleep display."
echo "Internet outage: the process will keep waiting and retry indefinitely."

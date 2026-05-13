---
name: experiment-runner
description: |
  Use when running training experiments, evaluating models, or analyzing experiment
  results. Triggers on "run experiment", "train", "evaluate", "test model", or
  when checking experiments/results/.
---

# Experiment Running Conventions

## Before Running
- Always check the config file (experiments/configs/*.yaml) for current hyperparameters
- Confirm random seed is set in config
- Estimate runtime; if >30 min, suggest screen/tmux or background run
- Check GPU availability: `nvidia-smi` (or skip for CPU)

## During Running
- Log to both stdout and experiments/logs/{exp_name}_{timestamp}.log
- Save checkpoints every N epochs (configurable, default N=5)
- Save best model by validation MCC

## After Running
- Always save final metrics to experiments/results/{exp_name}.json with this schema:
```json
  {
    "exp_name": "...",
    "config": {...},
    "git_commit": "...",
    "timestamp": "...",
    "final_metrics": {"acc": ..., "mcc": ..., "apv": ..., "sharpe": ...},
    "paper_targets": {"acc": ..., "mcc": ...},
    "match_within_1pct": true/false
  }
```
- Update docs/reproduction-progress.md with the result
- If results don't match paper, add a hypothesis to docs/reproduction-questions.md

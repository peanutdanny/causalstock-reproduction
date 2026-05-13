# CausalStock Reproduction Project

## Project Goal
Reproduce CausalStock (Li et al., NeurIPS 2024, arXiv:2411.06391) end-to-end as a benchmark
for my PhD thesis Paper α (Korean-Market CausalStock with Chaebol-aware Causal Priors).

## Background
This is my PhD reproduction work. I am a math PhD student at HUFS, specialization
financial mathematics. The end goal is to extend CausalStock with:
1. Korean market data (KRX + DART + Naver Finance)
2. Chaebol-aware causal priors (NTS-NOTEARS style)
3. GRU-VAR functional causal model (replacing the original FCM)

Phase 1 (current): Faithfully reproduce original results on ACL18 / CMIN-US / CMIN-CN /
KDD17 / NI225 / FTSE100. Match published ACC and MCC numbers within ±0.5%.

## Architecture (CausalStock)
- Market Information Encoder: OHLCV embedding + Denoised News Encoder (DNE)
- DNE: GPT-3.5 scores news on 5 aspects (Correlation, Sentiment, Importance, Impact, Duration)
- Lag-dependent Temporal Causal Discovery: p(G_l | G_{l-1}, X_{T-l}) with Gumbel-Softmax
- Functional Causal Model: additive noise SCM with σ(ζ_i(...))
- Loss: ELBO + BCE

## Tech Stack
- Python 3.10
- PyTorch 2.x (CPU first, GPU later)
- transformers (FinBERT, Llama)
- OpenAI API (for GPT-3.5 DNE)
- pandas, numpy, networkx, scikit-learn

## Code Conventions
- All experiments must be reproducible. Set random seeds (torch, numpy, random) globally.
- Use Hydra or simple YAML configs in experiments/configs/ for hyperparameters.
- Every model checkpoint saved to experiments/checkpoints/{exp_name}/{timestamp}/.
- Logging: Python logging module, INFO level to stdout, DEBUG to file.
- Type hints required on all public functions.
- Docstrings: Google style.
- Run tests before commit: `pytest tests/ -v`.

## File Conventions
- src/models/: all nn.Module classes
- src/data/: data loaders, preprocessing
- src/training/: training loops, optimizer setup
- src/evaluation/: ACC, MCC, APV, Sharpe ratio metrics
- src/utils/: seeds, logging, config loading
- experiments/: experiment scripts that combine the above

## Working Style
- I prefer small commits with clear messages (Conventional Commits).
- Before writing code, explain the plan briefly. After writing, run a quick test.
- When uncertain, ASK rather than guess — especially about paper details.
- If something is unclear in the paper, mark it as `# TODO(paper-ambiguity): ...`
  and add it to docs/reproduction-questions.md.

## Reproduction Targets (from paper Table 1)
| Dataset | ACC | MCC |
|---|---|---|
| ACL18 | 63.42 | 0.27 |
| CMIN-US | ... | ... |
(fill in from paper)

## Commands I Use Often
- `python -m experiments.train --config experiments/configs/acl18.yaml`
- `pytest tests/ -v --tb=short`
- `tensorboard --logdir experiments/logs/`

## Reference Files
- docs/paper-summary.md: Full paper summary
- docs/expected-results.md: Target numbers for each dataset
- docs/reproduction-questions.md: Open questions about paper details
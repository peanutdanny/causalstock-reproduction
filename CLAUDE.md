# CausalStock Reproduction → Cross-Asset Extension Project

## Project Goal

**Phase 1** (current, near complete): Reproduce CausalStock (Li et al., NeurIPS 2024, arXiv:2411.06391) end-to-end on the six paper datasets (ACL18 / CMIN-US / CMIN-CN / KDD17 / NI225 / FTSE100). Match published ACC/MCC within ±0.5%. Current state: ACL18 1-seed test ACC 0.623 (paper 0.6342). See `docs/project-status.md`.

**Phase 2 (Paper β)**: Extend CausalStock to **cross-asset causal discovery** — D≈25-30 universe combining FX majors, sovereign rates, commodities, and equity indices, with LLM-denoised macro-news from FOMC/ECB/BOJ/BOE/BOK. Three contributions:

1. First end-to-end causal-discovery + LLM-news system for cross-asset prediction (empty literature quadrant per May 2026 review).
2. **GRU-VAR Functional Causal Model** with Student-t likelihood for fat-tailed returns (math contribution, preserves advisor's line).
3. **Lead-lag causal prior G^p** from observed cross-asset propagation patterns (not triangular arbitrage — that's algebraic identity, not causal).

Primary metric: **portfolio Sharpe** (FX/macro directional ACC ceiling is ~53%, so ACC is secondary). Target submission: NeurIPS 2027 or ICML 2027.

Korean equities + chaebol prior: kept as a **comparison-appendix chapter** — same architecture run on KOSPI 200 to validate the domain-prior methodology, not the thesis spine.

## Background

PhD student at HUFS, financial mathematics. Pivoted on 2026-05-21 from "Korean equities + chaebol prior + GRU-VAR FCM (primary)" to "cross-asset primary, Korean appendix" after literature review showed FX/cross-asset causal discovery is essentially empty post-CausalStock while equity multi-relational GNN and dynamic-graph work is crowded (OmniGNN, MDGNN, FinMamba, …).

Full thesis plan: `~/.claude/plans/glimmering-percolating-mist.md` (approved 2026-05-21).

## Architecture

**Base CausalStock (Phase 1, faithful reproduction)**:
- Market Information Encoder (MIE): OHLCV embedding + Denoised News Encoder (DNE)
- DNE: GPT-3.5 scores news on 5 aspects (Correlation, Sentiment, Importance, Impact, Duration)
- Lag-dependent Temporal Causal Discovery (TCD): p(G_l | G_{l-1}, X_{T-l}) with Gumbel-Softmax
- Functional Causal Model (FCM): additive noise SCM with σ(ζ_i(...)) + Bernoulli p(rise)
- Loss: ELBO + BCE

**Phase 2 structural rewrites** (not config changes — see plan file for details):
- **FCM** (`src/models/fcm.py`): sigmoid-Bernoulli → **Student-t likelihood + GRU-VAR transition** for fat-tailed cross-asset log-returns
- **TCD** (`src/models/tcd.py`): input-independent U,V parameters → **amortized posterior** conditioned on X_{<T} (paper's own Appendix E.1 main limitation)
- **MIE/Dataset** (`src/data/dataset.py`): per-asset news → **shared-macro + per-asset local-news two-channel** input

Modules that port over with config-only changes: `src/training/loss.py` (just swap likelihood term), `src/training/trainer.py`, `src/evaluation/` (APV/Sharpe becomes primary), `src/data/dne_cache.py`, `src/utils/`.

## Tech Stack
- Python 3.10
- PyTorch 2.x (CPU first, GPU later)
- transformers (FinBERT, Llama — for Phase 2 LLM ablations)
- OpenAI API (GPT-3.5/4 DNE)
- pandas, numpy, networkx, scikit-learn, pyarrow
- **Phase 2 additions**: pandas-datareader / fredapi (FRED), yfinance, BeautifulSoup (central-bank statement scrapers)

## Code Conventions
- All experiments must be reproducible. Set random seeds (torch, numpy, random) globally.
- Use simple YAML configs in `experiments/configs/` for hyperparameters.
- Every model checkpoint saved to `experiments/checkpoints/{exp_name}/{timestamp}/`.
- Logging: Python logging module, INFO level to stdout, DEBUG to file.
- Type hints required on all public functions.
- Docstrings: Google style.
- Run tests before commit: `pytest tests/ -v`.

## Verification Cadence
- Per component: unit test + commit. Skip paper-checker for glue code.
- Per milestone (G1–G5): paper-checker for math/arch parts, experiment-analyst for G5 numbers.
- When paper is ambiguous: update `docs/reproduction-questions.md` BEFORE guessing.

## File Conventions
- `src/models/`: all nn.Module classes
- `src/data/`: data loaders, preprocessing (Phase 1: `acl18.py`, `stocknet.py`; Phase 2 add: `cross_asset.py`, `macro_news.py`)
- `src/training/`: training loops, optimizer setup
- `src/evaluation/`: ACC, MCC, APV, Sharpe ratio metrics
- `src/utils/`: seeds, logging, config loading
- `experiments/`: experiment scripts that combine the above

## Working Style
- I prefer small commits with clear messages (Conventional Commits).
- Before writing code, explain the plan briefly. After writing, run a quick test.
- When uncertain, ASK rather than guess — especially about paper details.
- If something is unclear in the paper, mark it as `# TODO(paper-ambiguity): ...`
  and add it to `docs/reproduction-questions.md`.

## Phase 1 Targets (CausalStock reproduction, from paper Table 1)
| Dataset | ACC | MCC |
|---|---|---|
| ACL18 | 63.42 | 0.27 |
| CMIN-US | ... | ... |
(fill in from paper)

## Phase 2 Targets (cross-asset, finalized after Phase 1 wrap-up)
- Headline: **Sharpe ≥ +0.2** over (i) LSTM-only, (ii) CausalStock-direct-port (Bernoulli FCM), (iii) naïve carry trade, (iv) S&P 500 buy-and-hold
- D ≥ 25 assets, 2010-01-01 → 2024-12-31 daily, NY 17:00 ET close
- 5 seeds for headline, 3 seeds for ablations
- Statistical rigor: bootstrap Sharpe CI (1000 resamples), Diebold-Mariano vs LSTM
- Detailed cross-asset spec to live in `docs/cross-asset-spec.md` (to be created in Phase 2A)

## Commands I Use Often
- `python -m experiments.train --config experiments/configs/acl18.yaml`
- `pytest tests/ -v --tb=short`
- `tensorboard --logdir experiments/logs/`

## Reference Files
- `docs/paper-summary.md`: Full CausalStock paper summary
- `docs/expected-results.md`: Phase 1 target numbers (Phase 2 section to be added)
- `docs/reproduction-questions.md`: Open questions about paper details
- `docs/project-status.md`: Current implementation/verification status (Phase 1)
- `~/.claude/plans/glimmering-percolating-mist.md`: Phase 2 thesis plan (approved 2026-05-21)

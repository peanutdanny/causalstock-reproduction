# Phase 9.5 — Low-fidelity Full-data Sanity Report

**Date**: 2026-05-13
**Goal**: Verify the full architecture (Phases 0–9) runs end-to-end on the
full 85-stock ACL18 universe and actually learns from the data, before
committing to Phase 10 (10 seeds × 100 epochs).

## Critical bug found and fixed

The initial 10-epoch run on the full universe showed **all configs at chance
level (~0.50 test ACC)**. Investigation revealed that two of the six raw price
features were on extreme scales:

| Feature | min | max | mean | std |
|---|---|---|---|---|
| movement | -0.49 | 0.22 | 0.0008 | 0.018 |
| feat1 | -0.15 | 0.33 | 0.056 | 0.054 |
| feat2 | -0.09 | 0.33 | 0.067 | 0.054 |
| feat3 | -0.49 | 0.23 | 0.046 | 0.056 |
| feat4 | **-48.4** | **57.6** | 0.11 | **3.15** |
| volume | **0** | **463M** | **21.9M** | **40.3M** |

Xavier-initialized linear layers receiving `volume ≈ 10⁷` produce saturated
activations on the first forward pass. The model never learned.

**Fix**: z-score normalize all 6 features per-feature, using train-split
statistics (docs/reproduction-questions.md A.1 default). Implementation in
`CausalStockDataset.compute_feature_stats` + `build_acl18_splits`.

## After-fix sanity sweep (30 epochs, 1 seed, full 85 stocks)

| Label | lr | likelihood | use_news | best val ACC | test ACC | test MCC | wall-clock |
|---|---|---|---|---|---|---|---|
| price_only_lr1e-3_bern | 1e-3 | bernoulli | ❌ | 0.715 | **0.719** | 0.438 | 99 s |
| price_only_lr1e-4_bern | 1e-4 | bernoulli | ❌ | 0.638 | 0.636 | 0.270 | 103 s |
| **price_only_lr1e-5_gauss** (paper-exact) | 1e-5 | gaussian | ❌ | 0.562 | **0.560** | 0.123 | 102 s |
| full_lr1e-3_bern (mock DNE) | 1e-3 | bernoulli | ✅ | 0.708 | 0.717 | 0.432 | 135 s |

## Interpretation

1. **Architecture is sound**. The paper-exact setting reaches 0.560 test ACC
   on price-only — within striking distance of the paper's "w/o news"
   ablation target of **0.581 ± 0.01** (docs/expected-results.md Table 2).

2. **Higher lr + bernoulli likelihood reaches 0.72**. This is well above the
   paper's full 0.6342 but uses non-paper hyperparameters. Use only as a
   sanity ceiling; Phase 10 must stick to paper-exact (lr=1e-5, gaussian).

3. **Mock DNE adds no signal**. price-only ≈ full when news is mocked. This
   confirms the mock is correctly behaving as noise. The paper's 0.6342 vs
   0.5810 (~+5%p) requires real GPT-3.5 scoring.

## Phase 10 budget extrapolation

- Per epoch on full 85 stocks: **~3.3 s** (sweep result: 99 s / 30 epochs)
- Per seed × 100 epochs: **~5–6 min**
- 10 seeds: **~1 hour per config**
- 5 configs (full + 4 ablations) × 10 seeds × 100 epochs: **~5 hours CPU**

Much more tractable than the 50-hour initial estimate. Phase 10 is feasible
on a single workstation overnight.

## Recommendation

Architecture passes sanity. To reach the paper's 0.6342, Phase 10 needs:

1. **Real GPT-3.5 DNE scores** (Phase 3b actual run, ~$30–50, several hours
   of API calls). Without this, ceiling is ~0.58 (mock = noise).
2. **Paper-exact settings** in the existing `experiments/configs/acl18.yaml`
   (lr=1e-5, gaussian, 100 epochs, batch=32, λ=0.01).
3. Optionally: a parallel `paper-exact-bernoulli` config to test the
   likelihood-form ambiguity (docs/reproduction-questions.md I.3).

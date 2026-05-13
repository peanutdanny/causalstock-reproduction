---
name: math-rigor
description: |
  Use when implementing mathematical components (loss functions, KL divergences,
  Gumbel-Softmax, ELBO, DAG learning constraints). Triggers on "loss", "ELBO",
  "Gumbel", "KL", "gradient", "math", "equation", "theorem".
---

# Mathematical Implementation Rigor

## Required
- Every math operation must have a comment citing the paper equation or source
- Numerical stability:
  - Use log_softmax instead of log(softmax)
  - Use F.binary_cross_entropy_with_logits not BCE(sigmoid(x))
  - Add epsilon (1e-8) to denominators
  - Clamp log inputs: `torch.log(x.clamp(min=1e-8))`
- Gradient check on small examples for custom backward
- Loss components logged separately (ELBO = reconstruction + KL_G + KL_z)

## Common pitfalls in this project
- Gumbel-Softmax temperature schedule: paper anneals from 1.0 to 0.1 over training
- KL divergence between two Categoricals needs careful broadcasting
- DAG acyclicity constraint: in CausalStock, time-irreversibility removes the need
  for h(W) constraint (unlike NOTEARS) — DO NOT add NOTEARS constraint by mistake

## Reproducibility
- All randomness must respect the global seed set by src/utils/seeds.py
- For Gumbel sampling, use torch.rand with same seed

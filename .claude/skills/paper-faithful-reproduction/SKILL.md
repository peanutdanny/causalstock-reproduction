---
name: paper-faithful-reproduction
description: |
  Use this skill any time we are implementing or modifying code that needs to match
  the CausalStock paper exactly. Triggers when the user mentions "reproduce", "match
  the paper", "from the paper", or refers to specific equations, tables, or sections
  of CausalStock (Li et al., NeurIPS 2024, arXiv:2411.06391).
---

# Paper-Faithful Reproduction Rules

When implementing CausalStock code:

## Always
- Cite the exact equation/section from the paper in code comments. Example:
```python
  # Eq. (3): p(G_l | G_{l-1}, X_{T-l}) with Gumbel-Softmax relaxation
```
- If a hyperparameter is not explicitly stated in the paper, mark it with
  `# TODO(paper-ambiguity)` and add to docs/reproduction-questions.md.
- Match variable names to the paper where possible (G, X, y, ζ, etc.).
- Use the EXACT loss formulation, EXACT optimizer, EXACT lr schduler from the paper.
- Default to the paper's reported hyperparameters in configs.

## Never
- Don't "improve" the architecture before reproducing baseline results.
- Don't substitute a "more modern" component (e.g. swapping GRU for Transformer)
  during reproduction phase — that's for the extension phase (Paper α).
- Don't skip ablation components even if they look redundant.

## Verification Checklist
After implementing each component, verify:
1. Output shapes match paper's described tensor dimensions
2. Loss decreases on small toy dataset
3. Forward pass numerical magnitude is reasonable (not exploding)
4. Compare ACC/MCC to paper Table 1 within ±1% on at least ACL18

## Reference
- Paper: docs/paper-summary.md
- Targets: docs/expected-results.md
- Open questions: docs/reproduction-questions.md

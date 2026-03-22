# Changelog — Logos v0.2-alpha

## v0.2-alpha — Training stability + improved generation

**Released:** March 2026

Built on v0.1-alpha. The architecture is unchanged — this version focuses on
training robustness and generation quality.

### What changed from v0.1-alpha

**Training:**
- Added cosine LR scheduler with 100-step linear warmup (final LR decays to 5% of base)
- Added gradient clipping at max norm 1.0
- Added best-checkpoint saving — model weights saved whenever val loss improves

**Generation:**
- `generate()` now supports `temperature` and `top_k` sampling
- Default: `temperature=0.9`, `top_k=40`
- Replaced raw multinomial (greedy) with controlled sampling

**New:**
- CELL 19: generation quality check reports generated length, unique chars used, and val perplexity

### What was observed
- Loss curve shape is similar to v0.1 — sharp early drop then steady improvement
- Final metrics are marginally different from v0.1 (val loss 1.5055 vs 1.4996) — expected,
  the cosine LR schedule with warmup changes learning dynamics and the runs are not
  directly comparable
- Best checkpoint saved at step 4999 (val loss 1.5042)
- Generated text is noticeably more grammatically plausible with temperature + top-k
  sampling vs the greedy multinomial of v0.1

### Results
| Metric | v0.1-alpha | v0.2-alpha |
|---|---|---|
| Train loss | 1.2218 | 1.2607 |
| Val loss | 1.4996 | 1.5055 |
| Train perplexity | 3.39 | 3.53 |
| Val perplexity | 4.48 | 4.51 |
| Best val checkpoint | — | 1.5042 @ step 4999 |
| Sampling | greedy | temperature=0.9, top_k=40 |
# Changelog

---

## v0.2-alpha — Training stability + improved generation

**Released:** March 2026

Built on v0.1-alpha. Architecture unchanged — this version focuses on training
robustness and generation quality.

### What changed
- Added cosine LR scheduler with 100-step linear warmup (decays to 5% of base lr)
- Added gradient clipping at max norm 1.0
- Added best-checkpoint saving: model saved whenever val loss improves
- `generate()` now supports `temperature` and `top_k` sampling (default: 0.9 / 40)
- New generation quality check cell: reports length, unique chars, val perplexity

### What was observed
- Loss curve shape similar to v0.1 — expected
- Marginal metric difference vs v0.1 is expected: cosine LR schedule changes dynamics
  and runs are not directly comparable
- Best checkpoint saved at step 4999 (val loss 1.5042)
- Generated text noticeably more plausible with temperature + top-k vs greedy

### Results
| Metric | Value |
|---|---|
| Train loss | 1.2607 |
| Val loss | 1.5055 |
| Train perplexity | 3.53 |
| Val perplexity | 4.51 |
| Best val checkpoint | 1.5042 @ step 4999 |

---

## v0.1-alpha — Initial prototype

**Released:** March 2026

First working version of Logos. The goal was to build a decoder-only Transformer
from scratch in PyTorch and verify it could learn to generate Shakespeare-like text.

### What was built
- Character-level tokenizer (vocab size 65)
- Decoder-only Transformer: 6 layers, 6 heads, embedding dim 192, context length 128
- Causal masked self-attention with pre-LayerNorm and residuals
- Feed-forward network with 4x expansion and ReLU
- Cross-entropy loss, AdamW optimizer at fixed learning rate 3e-4
- Greedy multinomial sampling for generation
- 5,000 training steps on CPU (Kaggle), ~295 minutes

### What was observed
- Loss dropped sharply in the first 500 steps then slowed — expected for this model size
- No significant overfitting observed — train/val gap was stable throughout
- Generated text learned Shakespeare's structure (names, dialogue format, line breaks)
  but word-level coherence is poor, words are often invented or malformed
- No LR scheduling, gradient clipping, or checkpointing — known limitations to address

### Results
| Metric | Value |
|---|---|
| Train loss | 1.2218 |
| Val loss | 1.4996 |
| Train perplexity | 3.39 |
| Val perplexity | 4.48 |
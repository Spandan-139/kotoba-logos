# Changelog

---

## v0.1-alpha — Initial prototype

**Released:** March 2026

First working version of Logos. The goal was to build a decoder-only Transformer
from scratch in PyTorch and verify it could learn to generate Shakespeare-like text.

### What was built
- Character-level tokenizer (vocab size 65)
- Decoder-only Transformer: 6 layers, 6 heads, embedding dim 192, context length 128
- Causal masked self-attention with pre-LayerNorm and residuals
- Feed-forward network with 4× expansion and ReLU
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
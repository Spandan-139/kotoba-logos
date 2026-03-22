# Changelog — Logos v0.3-alpha

## v0.3-alpha — BPE tokenization + OpenWebText + GPU training

**Released:** March 2026

Built on v0.2-alpha. This version is the first major architectural shift —
moving from character-level tokenization and Shakespeare to BPE tokenization
and real-world web text, trained on GPU for the first time.

### What changed from v0.2-alpha

**Tokenizer:**
- Replaced character-level tokenizer (vocab size 65) with GPT-2 BPE via tiktoken
- Vocab size: 65 → 50,257
- Tokenization is now subword-level — the model works with word fragments
  rather than individual characters, enabling far more coherent output

**Dataset:**
- Replaced Tiny Shakespeare (1.1M chars) with OpenWebText (8,000 samples, 38.8M chars)
- Encoded token count: 8,876,419 tokens
- Train / Val split: 7,988,777 / 887,642 tokens
- Text is now diverse real-world web content rather than literary prose

**Model:**
- Embedding dim: 192 → 256
- Attention heads: 6 → 8
- Layers: 6 → 8
- Context length: 128 → 256
- Total parameters: 2,715,713 → ~32M (driven primarily by larger embedding + lm_head)

**Generation:**
- Added top-p (nucleus) sampling alongside existing top-k and temperature
- Default: `temperature=0.9`, `top_k=40`, `top_p=0.9`

**Training:**
- First GPU training run (Kaggle P100)
- Mixed precision training via torch.amp.autocast + GradScaler
- Batch size reduced to 32 (from 64) due to larger vocab embedding memory cost
- All other training settings carried over from v0.2:
  - Cosine LR with warmup
  - Gradient clipping
  - Best checkpoint saving

### What was observed
- Initial loss ~10.87 — correct for random init over vocab size 50,257 (ln(50257) ≈ 10.82)
- Loss curve showed steady improvement throughout 5,000 steps
- Best checkpoint saved at step 4999 — model still improving at end of run,
  suggesting more steps or data would yield further gains
- Generated text is qualitatively far more coherent than v0.2 — real words,
  real grammar, real sentence structure throughout
- Perplexity is not directly comparable to v0.1/v0.2 due to fundamentally
  different tokenizer and dataset

### Results
| Metric | v0.2-alpha | v0.3-alpha |
|---|---|---|
| Train loss | 1.2607 | 4.8780 |
| Val loss | 1.5055 | 5.2579 |
| Train perplexity | 3.53 | 131.36 |
| Val perplexity | 4.51 | 192.08 |
| Best val checkpoint | 1.5042 @ step 4999 | 5.2422 @ step 4999 |
| Tokenizer | Character-level (65) | GPT-2 BPE (50,257) |
| Dataset | Tiny Shakespeare | OpenWebText (8k samples) |
| Sampling | temp=0.9, top_k=40 | temp=0.9, top_k=40, top_p=0.9 |
| Hardware | CPU (Kaggle) | GPU P100 (Kaggle) |

> Note: Metrics are not comparable across versions due to the tokenizer and
> dataset change. Perplexity is computed over a 50,257-token vocabulary on
> real web text — a fundamentally harder task than character-level Shakespeare.
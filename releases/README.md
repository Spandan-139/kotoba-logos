# Releases

All Logos versions, newest first.

| Version | Stage | Train loss | Val loss | Val PPL | Key changes |
|---|---|---|---|---|---|
| v0.3 | alpha | 4.8780 | 5.2579 | 192.08 | BPE tokenizer, OpenWebText, GPU training, top-p sampling, ~32M params |
| v0.2 | alpha | 1.2607 | 1.5055 | 4.51 | LR scheduler, grad clip, best checkpoint, top-k/temp sampling |
| v0.1 | alpha | 1.2218 | 1.4996 | 4.48 | Initial prototype |

> Note: v0.3 metrics are not comparable to v0.1/v0.2 due to tokenizer and dataset change.

---

## v0.3-alpha
First major architectural shift:
- GPT-2 BPE tokenizer
- Real-world web text (OpenWebText)
- Mixed precision GPU training
- Top-p sampling

→ [`v0.3/alpha/`](v0.3/alpha/)

---

## v0.2-alpha
Training stability + improved generation:
- Cosine LR scheduler
- Gradient clipping
- Best-checkpoint saving
- Temperature + top-k sampling

→ [`v0.2/alpha/`](v0.2/alpha/)

---

## v0.1-alpha
First working prototype:
- Character-level Transformer trained on Tiny Shakespeare
- Fixed learning rate
- Greedy sampling
- No checkpointing

→ [`v0.1/alpha/`](v0.1/alpha/)
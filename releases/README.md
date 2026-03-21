# Releases

All Logos versions, newest first.

| Version | Stage | Train loss | Val loss | Val PPL | Key changes |
|---|---|---|---|---|---|
| v0.2 | alpha | 1.2607 | 1.5055 | 4.51 | LR scheduler, grad clip, best checkpoint, top-k/temp sampling |
| v0.1 | alpha | 1.2218 | 1.4996 | 4.48 | Initial prototype |

---

## v0.2-alpha
Training stability + improved generation. Cosine LR scheduler, gradient clipping, best-checkpoint saving, temperature + top-k sampling.
→ [`v0.2/alpha/`](v0.2/alpha/)

---

## v0.1-alpha
First working prototype. Character-level Transformer trained on Tiny Shakespeare.
Fixed learning rate, greedy sampling, no checkpointing.
→ [`v0.1/alpha/`](v0.1/alpha/)
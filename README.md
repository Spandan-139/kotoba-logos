# Logos

**Logos** is a research preview of a mini decoder-only Transformer language model built under **Kotoba**.

Built from scratch with PyTorch to understand and implement a small autoregressive language model. This version is a character-level Transformer trained on the Tiny Shakespeare dataset for next-token prediction and text generation.

## Status
**Alpha / Research Preview** — `v0.1.0-alpha`

---

## Architecture

| Component | Detail |
|---|---|
| Type | Decoder-only Transformer |
| Tokenizer | Character-level (vocab size: 65) |
| Embedding dim (`N_EMBD`) | 192 |
| Attention heads (`N_HEAD`) | 6 |
| Layers (`N_LAYER`) | 6 |
| Context length (`BLOCK_SIZE`) | 128 |
| Batch size | 64 |
| Dropout | 0.2 |
| Total parameters | **2,715,713** |

### Components
- Token + positional embeddings
- Causal masked self-attention (`Head`)
- Multi-head self-attention (`MultiHeadAttention`)
- Feed-forward network with ReLU (4× expansion)
- Residual connections + Pre-LayerNorm (`Block`)
- Cross-entropy loss, AdamW optimizer (lr = 3e-4)

---

## Training

| Setting | Value |
|---|---|
| Dataset | Tiny Shakespeare (1,115,394 chars) |
| Train / Val split | 90% / 10% |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Iterations | 5,000 |
| Hardware | CPU (Kaggle) |
| Training time | ~295 minutes |

### Loss curve

| Step | Train loss | Val loss |
|---|---|---|
| 0 | 4.1834 | 4.1805 |
| 500 | 1.9262 | 2.0115 |
| 1000 | 1.5777 | 1.7492 |
| 1500 | 1.4526 | 1.6543 |
| 2000 | 1.3840 | 1.6010 |
| 2500 | 1.3366 | 1.5612 |
| 3000 | 1.3053 | 1.5357 |
| 3500 | 1.2765 | 1.5270 |
| 4000 | 1.2637 | 1.5217 |
| 4500 | 1.2422 | 1.5020 |
| 4999 | 1.2226 | 1.5001 |

### Final results

| Metric | Value |
|---|---|
| Train loss | **1.2218** |
| Val loss | **1.4996** |
| Train perplexity | **3.39** |
| Val perplexity | **4.48** |

---

## Sample Output

```
CLARENCE:
Ah, see how it is no fearful an reverage!
Touch's unto amen so happiness, sir, if you are did.

TYBALT:
No, love, still hand's far a smalliabless;
I are so muter with his thins heir;
For sucry thus goes himself, velute! O God!

ELBOW:
She trespast love, by a good taggerous sonarer
Is nothing; that 'twasting for cit?

QUEEN MARGARET:
My brathe king and by the king all the strong.
And my name under to my my hthought.
```

---

## Repository Structure

```text
kotoba-logos/
├── notebooks/
│   └── logos_v0_1_alpha.ipynb
├── assets/
│   └── sample_output.txt
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── CHANGELOG.md
```

---

## Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open the notebook:
   ```bash
   jupyter notebook notebooks/logos_v0_1_alpha.ipynb
   ```
> Originally trained on Kaggle. The notebook auto-downloads the Tiny Shakespeare dataset on first run.

---

## Roadmap
- [ ] Add learning rate scheduler
- [ ] Add gradient clipping
- [ ] Save best validation checkpoint
- [ ] Improve text generation (top-k / temperature sampling)
- [ ] Train on GPU
- [ ] Compare multiple model sizes
- [ ] Explore subword / BPE tokenization

---

## Author
**Spandan Basu Chaudhuri**

---

*Built under [Kotoba](https://github.com/Spandan-139)*
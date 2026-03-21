# Logos

**Logos** is a research preview of a mini decoder-only Transformer language model built under **Kotoba**.

Built from scratch with PyTorch to understand and implement a small autoregressive language model. A character-level Transformer trained on the Tiny Shakespeare dataset for next-token prediction and text generation.

## Status
**Alpha / Research Preview** — `v0.1-alpha`

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
- Cross-entropy loss, AdamW optimizer

---

## Results

| Metric | v0.1-alpha |
|---|---|
| Train loss | 1.2218 |
| Val loss | 1.4996 |
| Train perplexity | 3.39 |
| Val perplexity | 4.48 |
| Sampling | Greedy multinomial |

Full metrics, loss curves, and sample outputs for every version live in `releases/`.

---

## Repository Structure

```text
logos/
├── logos/                  ← Python package
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── data.py
│   ├── train.py
│   └── generate.py
├── releases/
│   ├── README.md
│   └── v0.1/
│       └── alpha/
│           ├── sample_output.txt
│           ├── loss_curve.png
│           ├── metrics.md
│           └── CHANGELOG.md
├── README.md
├── CHANGELOG.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run training:
   ```python
   from logos.data import load_text, build_tokenizer, split_data
   from logos.model import MiniTransformerLM
   from logos.train import train
   from logos.generate import generate
   from logos.config import device

   text = load_text()
   encode, decode, vocab_size, chars = build_tokenizer(text)
   train_data, val_data = split_data(text, encode)

   model = MiniTransformerLM(vocab_size).to(device)
   train(model, train_data, val_data)

   print(generate(model, decode))
   ```

> Training was done on Kaggle (CPU). Use the launcher script in each release folder to reproduce a run on Kaggle.

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
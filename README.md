# Logos

**Logos** is a research preview of a mini decoder-only Transformer language model built under **Kotoba**.

Built from scratch with PyTorch to understand and document exactly how a language model is constructed and trained — not to compete with production systems, but to make the internals legible. Trained on real-world web text using BPE tokenization and GPU compute for next-token prediction and text generation.

## Status
**Alpha / Research Preview** — `v0.3-alpha`

---

## Architecture

| Component | Detail |
|---|---|
| Type | Decoder-only Transformer |
| Tokenizer | GPT-2 BPE via tiktoken (vocab size: 50,257) |
| Embedding dim (`N_EMBD`) | 256 |
| Attention heads (`N_HEAD`) | 8 |
| Layers (`N_LAYER`) | 8 |
| Context length (`BLOCK_SIZE`) | 256 |
| Batch size | 32 |
| Dropout | 0.2 |
| Total parameters | **~32M** |

### Components
- Token + positional embeddings
- Causal masked self-attention (`Head`)
- Multi-head self-attention (`MultiHeadAttention`)
- Feed-forward network with ReLU (4x expansion)
- Residual connections + Pre-LayerNorm (`Block`)
- Cross-entropy loss
- AdamW optimizer
- Cosine LR scheduler with linear warmup
- Gradient clipping
- Mixed precision training (AMP)
- Temperature + top-k + top-p sampling

---

## Results

| Version | Train loss | Val loss | Train PPL | Val PPL | Best val checkpoint | Tokenizer | Dataset | Hardware |
|---|---|---|---|---|---|---|---|---|
| v0.3-alpha | 4.8780 | 5.2579 | 131.36 | 192.08 | 5.2422 @ step 4999 | GPT-2 BPE | OpenWebText | GPU P100 |
| v0.2-alpha | 1.2607 | 1.5055 | 3.53 | 4.51 | 1.5042 @ step 4999 | Char-level | Tiny Shakespeare | CPU |
| v0.1-alpha | 1.2218 | 1.4996 | 3.39 | 4.48 | — | Char-level | Tiny Shakespeare | CPU |

> v0.3 metrics are not comparable to v0.1/v0.2 — different tokenizer, different dataset, fundamentally harder task.

---

## Repository Structure

```text
kotoba-logos/
├── logos/                  ← Python package
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── data.py
│   ├── train.py
│   └── generate.py
├── releases/
│   ├── README.md
│   ├── v0.1/
│   │   └── alpha/
│   │       ├── logos_v0.1_alpha.ipynb
│   │       ├── sample_output.txt
│   │       ├── loss_curve.png
│   │       ├── metrics.md
│   │       └── CHANGELOG.md
│   ├── v0.2/
│   │   └── alpha/
│   │       ├── logos_v0.2_alpha.ipynb
│   │       ├── sample_output.txt
│   │       ├── loss_curve.png
│   │       ├── metrics.md
│   │       └── CHANGELOG.md
│   └── v0.3/
│       └── alpha/
│           ├── logos-v0.3-alpha.ipynb
│           ├── sample_output.txt
│           ├── loss_curve.png
│           ├── metrics.md
│           └── CHANGELOG.md
├── data/                   ← cached dataset (git-ignored)
├── checkpoints/            ← saved model weights (git-ignored)
├── run.py                  ← CLI entry point
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

2. Train — downloads data, trains, saves best checkpoint automatically:
   ```bash
   python run.py train
   ```

3. Generate from the saved checkpoint:
   ```bash
   python run.py generate
   python run.py generate --tokens 200 --temperature 0.8
   python run.py generate --checkpoint ./checkpoints/logos_best.pth
   ```

Training saves the best checkpoint (by val loss) to `./checkpoints/logos_best.pth`. Generation always loads from the best checkpoint, not the final training state.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `LOGOS_DATA_CACHE` | `./data/owt_text.txt` | Path to cached OpenWebText text |
| `LOGOS_MODEL_PATH` | `./checkpoints/logos_best.pth` | Path to save/load best checkpoint |
| `LOGOS_OWT_SAMPLES` | `8000` | Number of OpenWebText samples to download |

To reproduce a Kaggle run, set these to Kaggle working paths:
```bash
LOGOS_DATA_CACHE=/kaggle/working/owt_text.txt \
LOGOS_MODEL_PATH=/kaggle/working/logos_best.pth \
python run.py train
```

> GPU strongly recommended. Training was done on Kaggle (P100). Use the notebook in each release folder to reproduce a run on Kaggle directly.

---

## Author
**Spandan Basu Chaudhuri**

---

*Built under [Kotoba](https://github.com/Spandan-139)*
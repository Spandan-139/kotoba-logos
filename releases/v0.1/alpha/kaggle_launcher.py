# ── Kaggle Launcher — Logos v0.1-alpha ───────────────────────────────────────
# Run this notebook on Kaggle. It clones the repo, runs training, and exports
# all release artifacts to /kaggle/working/ ready to download.

VERSION = "v0.1"
STAGE   = "alpha"

# ── 1. Clone repo ─────────────────────────────────────────────────────────────
!git clone https://github.com/Spandan-139/logos.git
%cd logos

# ── 2. Install dependencies ───────────────────────────────────────────────────
!pip install -r requirements.txt

# ── 3. Run training ───────────────────────────────────────────────────────────
import math
import torch
import matplotlib.pyplot as plt

from logos.config import device, GENERATE_TOKENS
from logos.data import load_text, build_tokenizer, split_data
from logos.model import MiniTransformerLM
from logos.train import train, estimate_loss
from logos.generate import generate

# Data
text                        = load_text()
encode, decode, vocab_size, chars = build_tokenizer(text)
train_data, val_data        = split_data(text, encode)

print(f"Vocab size: {vocab_size}")
print(f"Train tokens: {len(train_data)} | Val tokens: {len(val_data)}")

# Model
model = MiniTransformerLM(vocab_size).to(device)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Train
train_losses, val_losses, steps = train(model, train_data, val_data)

# ── 4. Generate ───────────────────────────────────────────────────────────────
generated_text = generate(model, decode)
print(generated_text)

# ── 5. Export release artifacts ───────────────────────────────────────────────
OUTPUT_DIR = "/kaggle/working"

# Loss curve
plt.figure(figsize=(8, 5))
plt.plot(steps, train_losses, label="Train loss")
plt.plot(steps, val_losses, label="Val loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title(f"Logos {VERSION}-{STAGE} Training Curve")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/loss_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved loss_curve.png")

# Sample output
with open(f"{OUTPUT_DIR}/sample_output.txt", "w") as f:
    f.write(f"# Sample output — Logos {VERSION}-{STAGE}\n")
    f.write(f"# Model: MiniTransformerLM (character-level decoder-only Transformer)\n")
    f.write(f"# Dataset: Tiny Shakespeare (1,115,394 chars)\n")
    f.write(f"# Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    f.write(f"# Sampling: greedy multinomial (no temperature / top-k)\n")
    f.write(f"# Generated tokens: {GENERATE_TOKENS}\n\n")
    f.write(generated_text)
print("Saved sample_output.txt")

# Metrics
final_losses = estimate_loss(model, train_data, val_data)
train_ppl    = math.exp(final_losses["train"])
val_ppl      = math.exp(final_losses["val"])

from logos.config import (BATCH_SIZE, BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER,
                           DROPOUT, LEARNING_RATE, MAX_ITERS)

md = f"""# Metrics — Logos {VERSION}-{STAGE}

## Model

| Component | Detail |
|---|---|
| Type | Decoder-only Transformer |
| Tokenizer | Character-level |
| Vocab size | {vocab_size} |
| Embedding dim | {N_EMBD} |
| Attention heads | {N_HEAD} |
| Layers | {N_LAYER} |
| Context length | {BLOCK_SIZE} |
| Batch size | {BATCH_SIZE} |
| Dropout | {DROPOUT} |
| Total parameters | {sum(p.numel() for p in model.parameters()):,} |

## Training

| Setting | Value |
|---|---|
| Dataset | Tiny Shakespeare (1,115,394 chars) |
| Train / Val split | 90% / 10% |
| Optimizer | AdamW |
| Learning rate | {LEARNING_RATE} (fixed) |
| Gradient clipping | None |
| LR scheduler | None |
| Checkpointing | None (final step saved) |
| Iterations | {MAX_ITERS} |
| Hardware | CPU (Kaggle) |

## Final Results

| Metric | Value |
|---|---|
| Train loss | **{final_losses["train"]:.4f}** |
| Val loss | **{final_losses["val"]:.4f}** |
| Train perplexity | **{train_ppl:.2f}** |
| Val perplexity | **{val_ppl:.2f}** |

## Generation

| Setting | Value |
|---|---|
| Tokens generated | {GENERATE_TOKENS} |
| Sampling | Greedy multinomial (no temperature / top-k) |

See `loss_curve.png` and `sample_output.txt`.
"""

with open(f"{OUTPUT_DIR}/metrics.md", "w") as f:
    f.write(md)
print("Saved metrics.md")

print(f"\nAll artifacts saved to {OUTPUT_DIR}/")
print(f"Download and drop into releases/{VERSION}/{STAGE}/")
import torch
import numpy as np
import random

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_URL   = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
LOCAL_TXT  = "/kaggle/working/input.txt"

# ── Model ─────────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
BLOCK_SIZE = 128        # context length
N_EMBD     = 192
N_HEAD     = 6
N_LAYER    = 6
DROPOUT    = 0.2

# ── Training ──────────────────────────────────────────────────────────────────
MAX_ITERS      = 5000
EVAL_INTERVAL  = 500
EVAL_ITERS     = 200
LEARNING_RATE  = 3e-4

# ── Generation ────────────────────────────────────────────────────────────────
GENERATE_TOKENS = 500

# ── Device ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Seed everything ───────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
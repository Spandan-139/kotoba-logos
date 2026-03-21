import torch
import numpy as np
import random

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_CACHE  = "/kaggle/working/owt_text.txt"
OWT_SAMPLES = 8000

# ── Model ─────────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
BLOCK_SIZE = 256        # context length
N_EMBD     = 256
N_HEAD     = 8
N_LAYER    = 8
DROPOUT    = 0.2

# ── Training ──────────────────────────────────────────────────────────────────
MAX_ITERS      = 5000
EVAL_INTERVAL  = 500
EVAL_ITERS     = 200
LEARNING_RATE  = 3e-4
GRAD_CLIP      = 1.0
WARMUP_ITERS   = 100
MIN_LR_RATIO   = 0.05

# ── Checkpointing ─────────────────────────────────────────────────────────────
BEST_MODEL_PATH = "/kaggle/working/logos_best_v0_3_alpha.pth"

# ── Generation ────────────────────────────────────────────────────────────────
GENERATE_TOKENS = 500
TEMPERATURE     = 0.9
TOP_K           = 40
TOP_P           = 0.9

# ── Device ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Seed everything ───────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

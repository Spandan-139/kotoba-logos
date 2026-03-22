import math
import os
import time
import torch
from .config import (MAX_ITERS, EVAL_INTERVAL, EVAL_ITERS, LEARNING_RATE,
                     GRAD_CLIP, WARMUP_ITERS, MIN_LR_RATIO, BEST_MODEL_PATH,
                     BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER, DROPOUT, device)
from .data import get_batch


def get_lr(iter_idx):
    """Cosine LR schedule with linear warmup."""
    if iter_idx < WARMUP_ITERS:
        return LEARNING_RATE * iter_idx / WARMUP_ITERS
    progress = (iter_idx - WARMUP_ITERS) / max(1, MAX_ITERS - WARMUP_ITERS)
    return LEARNING_RATE * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """Estimate mean loss over EVAL_ITERS batches for train and val splits."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y       = get_batch(split, train_data, val_data)
            _, loss    = model(X, Y)
            losses[k]  = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(model, train_data, val_data, vocab_size=None, tokenizer_type="gpt2"):
    """
    Training loop.
    Cosine LR with warmup, gradient clipping, AMP, best-checkpoint saving.
    Saves config metadata in the checkpoint so generate can reload without
    needing a separate config file.
    Returns (train_losses, val_losses, steps, best_val_loss, best_iter).
    """
    os.makedirs(os.path.dirname(os.path.abspath(BEST_MODEL_PATH)), exist_ok=True)

    optimizer    = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler       = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))
    train_losses = []
    val_losses   = []
    steps        = []
    best_val_loss = float("inf")
    best_iter     = 0
    start_time    = time.time()

    for iter_idx in range(MAX_ITERS):

        lr = get_lr(iter_idx)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_idx % EVAL_INTERVAL == 0 or iter_idx == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            steps.append(iter_idx)
            print(f"step {iter_idx:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | lr {lr:.2e}")

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                best_iter     = iter_idx
                checkpoint = {
                    "iter":             iter_idx,
                    "model_state_dict": model.state_dict(),
                    "val_loss":         best_val_loss,
                }
                if vocab_size is not None:
                    checkpoint["config"] = {
                        "vocab_size":  vocab_size,
                        "tokenizer":   tokenizer_type,
                        "BLOCK_SIZE":  BLOCK_SIZE,
                        "N_EMBD":      N_EMBD,
                        "N_HEAD":      N_HEAD,
                        "N_LAYER":     N_LAYER,
                        "DROPOUT":     DROPOUT,
                    }
                torch.save(checkpoint, BEST_MODEL_PATH)

        xb, yb = get_batch("train", train_data, val_data)

        with torch.amp.autocast('cuda', enabled=(device == "cuda")):
            _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

    elapsed = (time.time() - start_time) / 60
    print(f"\nTraining completed in {elapsed:.2f} minutes.")
    print(f"Best val loss: {best_val_loss:.4f} at step {best_iter}")
    return train_losses, val_losses, steps, best_val_loss, best_iter
import math
import time
import torch
from .config import (MAX_ITERS, EVAL_INTERVAL, EVAL_ITERS, LEARNING_RATE,
                     GRAD_CLIP, WARMUP_ITERS, MIN_LR_RATIO, BEST_MODEL_PATH, device)
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


def train(model, train_data, val_data):
    """
    Training loop for v0.2-alpha.
    Cosine LR scheduler with linear warmup, gradient clipping, best-checkpoint saving.
    Returns (train_losses, val_losses, steps, best_val_loss, best_iter).
    """
    optimizer    = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    train_losses = []
    val_losses   = []
    steps        = []
    best_val_loss = float("inf")
    best_iter     = 0
    start_time    = time.time()

    for iter_idx in range(MAX_ITERS):

        # set LR for this step
        lr = get_lr(iter_idx)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_idx % EVAL_INTERVAL == 0 or iter_idx == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            steps.append(iter_idx)
            print(f"step {iter_idx:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | lr {lr:.2e}")

            # save best checkpoint
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                best_iter     = iter_idx
                torch.save({
                    "iter":             iter_idx,
                    "model_state_dict": model.state_dict(),
                    "val_loss":         best_val_loss,
                }, BEST_MODEL_PATH)

        xb, yb = get_batch("train", train_data, val_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

    elapsed = (time.time() - start_time) / 60
    print(f"\nTraining completed in {elapsed:.2f} minutes.")
    print(f"Best val loss: {best_val_loss:.4f} at step {best_iter}")
    return train_losses, val_losses, steps, best_val_loss, best_iter
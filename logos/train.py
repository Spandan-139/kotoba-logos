import time
import torch
from config import (MAX_ITERS, EVAL_INTERVAL, EVAL_ITERS, LEARNING_RATE, device)
from data import get_batch


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
    Training loop for v0.1-alpha.
    Fixed learning rate, no grad clipping, no checkpointing.
    Returns (train_losses, val_losses, steps).
    """
    optimizer    = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    train_losses = []
    val_losses   = []
    steps        = []
    start_time   = time.time()

    for iter_idx in range(MAX_ITERS):

        if iter_idx % EVAL_INTERVAL == 0 or iter_idx == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            steps.append(iter_idx)
            print(f"step {iter_idx:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

        xb, yb = get_batch("train", train_data, val_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    elapsed = (time.time() - start_time) / 60
    print(f"\nTraining completed in {elapsed:.2f} minutes.")
    return train_losses, val_losses, steps
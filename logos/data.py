import os
import requests
import torch
from config import (DATA_URL, LOCAL_TXT, BLOCK_SIZE, BATCH_SIZE, device)


def load_text():
    """Download Tiny Shakespeare if not already present, return raw text."""
    if not os.path.exists(LOCAL_TXT):
        r = requests.get(DATA_URL, timeout=30)
        r.raise_for_status()
        with open(LOCAL_TXT, "w", encoding="utf-8") as f:
            f.write(r.text)
    with open(LOCAL_TXT, "r", encoding="utf-8") as f:
        return f.read()


def build_tokenizer(text):
    """Character-level tokenizer. Returns (encode, decode, vocab_size, chars)."""
    chars      = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi       = {ch: i for i, ch in enumerate(chars)}
    itos       = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join([itos[i] for i in ids])

    return encode, decode, vocab_size, chars


def split_data(text, encode):
    """90/10 train/val split, returns tensors."""
    data       = torch.tensor(encode(text), dtype=torch.long)
    n          = int(0.9 * len(data))
    train_data = data[:n]
    val_data   = data[n:]
    return train_data, val_data


def get_batch(split, train_data, val_data):
    """Sample a random batch from train or val data."""
    source = train_data if split == "train" else val_data
    ix     = torch.randint(len(source) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x      = torch.stack([source[i:i + BLOCK_SIZE] for i in ix])
    y      = torch.stack([source[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)
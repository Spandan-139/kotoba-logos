import os
import torch
import tiktoken
from datasets import load_dataset
from .config import DATA_CACHE, OWT_SAMPLES, BLOCK_SIZE, BATCH_SIZE, device


def load_text() -> str:
    """Download OpenWebText subset if not cached, return raw text."""
    if not os.path.exists(DATA_CACHE):
        print("Downloading OpenWebText subset...")
        dataset = load_dataset("openwebtext", split="train", streaming=True)
        samples = []
        for i, sample in enumerate(dataset):
            if i >= OWT_SAMPLES:
                break
            samples.append(sample["text"])
        text = "\n".join(samples)
        with open(DATA_CACHE, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved {len(samples)} samples.")
    else:
        with open(DATA_CACHE, "r", encoding="utf-8") as f:
            text = f.read()
    return text


def build_tokenizer(text: str):
    """
    GPT-2 BPE tokenizer via tiktoken.
    Returns (encode, decode, vocab_size).
    """
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab     # 50,257

    def encode(s: str):
        return enc.encode_ordinary(s)

    def decode(ids):
        return enc.decode(ids)

    return encode, decode, vocab_size


def split_data(text: str, encode):
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

import torch
from .config import GENERATE_TOKENS, TEMPERATURE, TOP_K, TOP_P, BEST_MODEL_PATH, device


def generate(model, decode, max_new_tokens=GENERATE_TOKENS,
             temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P):
    """
    Generate text using temperature + top-k + top-p sampling.
    Requires an already-loaded model and decode function.
    Returns the decoded string.
    """
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        ids = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )[0].tolist()
    return decode(ids)


def load_and_generate(checkpoint_path=None, max_new_tokens=GENERATE_TOKENS,
                      temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P):
    """
    Load the best saved checkpoint and generate text.
    Reconstructs the tokenizer from checkpoint metadata — no separate
    config file or live training session needed.
    Returns the decoded string.
    """
    from .model import MiniTransformerLM

    path = checkpoint_path or BEST_MODEL_PATH
    checkpoint = torch.load(path, map_location=device)

    config = checkpoint.get("config", {})
    vocab_size = config.get("vocab_size")
    if vocab_size is None:
        raise ValueError(
            f"Checkpoint at '{path}' has no config.vocab_size. "
            "Re-train with the updated train() to embed config in the checkpoint."
        )

    tokenizer_type = config.get("tokenizer", "gpt2")
    if tokenizer_type == "gpt2":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        decode = enc.decode
    else:
        chars = config.get("chars")
        if chars is None:
            raise ValueError("Checkpoint tokenizer is 'char' but no 'chars' list found in config.")
        itos = {i: ch for i, ch in enumerate(chars)}
        decode = lambda ids: "".join([itos[i] for i in ids])

    model = MiniTransformerLM(vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return generate(model, decode, max_new_tokens, temperature, top_k, top_p)
import torch
from .config import GENERATE_TOKENS, TEMPERATURE, TOP_K, device


def generate(model, decode, max_new_tokens=GENERATE_TOKENS,
             temperature=TEMPERATURE, top_k=TOP_K):
    """
    Generate text using temperature + top-k sampling.
    Returns the decoded string.
    """
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        ids = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )[0].tolist()
    return decode(ids)
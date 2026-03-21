import torch
from .config import GENERATE_TOKENS, device


def generate(model, decode, max_new_tokens=GENERATE_TOKENS):
    """
    Generate text from the model using greedy multinomial sampling.
    Returns the decoded string.
    """
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        ids = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    return decode(ids)
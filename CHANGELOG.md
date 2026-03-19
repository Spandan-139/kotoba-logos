# Changelog

## v0.1.0-alpha
- Initial research preview release
- Implemented decoder-only character-level Transformer from scratch with PyTorch
- Architecture: 6 layers, 6 heads, embedding dim 192, context length 128
- Trained 5000 steps on Tiny Shakespeare (~295 min on CPU)
- Train loss: 1.2218 | Val loss: 1.4996
- Train perplexity: 3.39 | Val perplexity: 4.48
- Total parameters: 2,715,713

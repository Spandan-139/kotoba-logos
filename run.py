#!/usr/bin/env python3
"""
Logos — run.py
Entry point for training and generation.

Usage:
  python run.py train
  python run.py train --no-generate

  python run.py generate
  python run.py generate --checkpoint ./checkpoints/logos_best.pth
  python run.py generate --tokens 200 --temperature 0.8 --top-k 50 --top-p 0.95

Environment variables:
  LOGOS_DATA_CACHE     Path to cached OpenWebText text   (default: ./data/owt_text.txt)
  LOGOS_MODEL_PATH     Path for best checkpoint          (default: ./checkpoints/logos_best.pth)
  LOGOS_OWT_SAMPLES    OpenWebText samples to download   (default: 8000)
"""

import argparse


def cmd_train(args):
    from logos.data import load_text, build_tokenizer, split_data
    from logos.model import MiniTransformerLM
    from logos.train import train
    from logos.generate import generate
    from logos.config import device

    print(f"Device: {device}")

    text = load_text()
    encode, decode, vocab_size = build_tokenizer(text)
    train_data, val_data = split_data(text, encode)

    model = MiniTransformerLM(vocab_size).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    train(model, train_data, val_data, vocab_size=vocab_size, tokenizer_type="gpt2")

    if args.generate_after:
        print("\n--- Sample output (best checkpoint) ---")
        from logos.generate import load_and_generate
        from logos.config import BEST_MODEL_PATH
        print(load_and_generate())


def cmd_generate(args):
    from logos.generate import load_and_generate

    print(load_and_generate(
        checkpoint_path=args.checkpoint,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    ))


def main():
    parser = argparse.ArgumentParser(
        description="Logos — mini decoder-only Transformer LM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── train ─────────────────────────────────────────────────────────────────
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--no-generate", dest="generate_after", action="store_false",
        help="Skip sample generation after training completes"
    )
    train_parser.set_defaults(generate_after=True)

    # ── generate ──────────────────────────────────────────────────────────────
    gen_parser = subparsers.add_parser(
        "generate", help="Generate text from a saved checkpoint"
    )
    gen_parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint (default: LOGOS_MODEL_PATH or ./checkpoints/logos_best.pth)"
    )
    gen_parser.add_argument(
        "--tokens", type=int, default=500,
        help="Number of tokens to generate (default: 500)"
    )
    gen_parser.add_argument(
        "--temperature", type=float, default=0.9,
        help="Sampling temperature (default: 0.9)"
    )
    gen_parser.add_argument(
        "--top-k", type=int, default=40,
        help="Top-k sampling (default: 40)"
    )
    gen_parser.add_argument(
        "--top-p", type=float, default=0.9,
        help="Nucleus sampling threshold (default: 0.9)"
    )

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)


if __name__ == "__main__":
    main()
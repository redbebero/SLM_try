"""Minimal checkpoint generator for the active Korean model path."""

import argparse
import sys

import torch

from core_generate import generate, load_model
from tokenizer import KoJamoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate from one Korean checkpoint.")
    parser.add_argument("checkpoint")
    parser.add_argument("prompt")
    parser.add_argument("--max-new-chars", type=int, default=50)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if "emb_cho.weight" not in state_dict:
        raise ValueError("generate.py expects a KoJamo checkpoint with emb_cho.weight")

    tokenizer = KoJamoTokenizer()
    model, device = load_model(args.checkpoint, tokenizer.get_vocab_sizes())
    print(generate(
        model,
        tokenizer,
        args.prompt,
        max_new_chars=args.max_new_chars,
        device=device,
    ))


if __name__ == "__main__":
    main()

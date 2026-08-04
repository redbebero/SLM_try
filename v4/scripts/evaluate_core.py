"""Prompt-aware exact evaluation for the active toy generation path."""

import argparse
import sys

from core_generate import generate, load_model
from tokenizer import KoJamoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--data", default="experiments/compositional_toy/valid/valid.txt")
    args = parser.parse_args()
    tokenizer = KoJamoTokenizer()
    model, device = load_model(args.checkpoint, tokenizer.get_vocab_sizes())
    hits = 0
    total = 0
    for line in open(args.data, encoding="utf-8"):
        text = line.rstrip("\n")
        if not text:
            continue
        prompt, answer = text.rsplit(" ", 1)
        predicted = generate(model, tokenizer, prompt + " ", len(answer), device)
        ok = predicted == answer
        hits += ok
        total += 1
        print(f"{'PASS' if ok else 'FAIL'} prompt={prompt + ' '!r} predicted={predicted!r} expected={answer!r}")
    print(f"exact={hits}/{total}")


if __name__ == "__main__":
    main()

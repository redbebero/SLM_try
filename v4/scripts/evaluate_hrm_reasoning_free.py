"""Measure reasoning HRM by pure autoregressive exact match, without routing."""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chat import generate, load_model
from prepare_hrm_reasoning import make_tasks, make_tasks_for_type
from tokenizer import KoJamoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--show", type=int, default=0)
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--kind", choices=("arithmetic", "jamo", "ordering"), default=None)
    args = parser.parse_args()
    tokenizer = KoJamoTokenizer()
    model, device = load_model(args.checkpoint, tokenizer.get_vocab_sizes())
    if args.segments is not None:
        model.inference_segments = args.segments
    counts = Counter()
    correct = Counter()
    tasks = (make_tasks_for_type(args.count, args.kind, seed=args.seed,
                                 compact=args.compact, canonical=args.canonical)
             if args.kind else
             make_tasks(args.count, seed=args.seed, compact=args.compact, canonical=args.canonical))
    for index, item in enumerate(tasks):
        prompt, expected = item.split("\nA: ", 1)
        kind = "arithmetic" if "[산수]" in prompt else "jamo" if "[자소]" in prompt else "ordering"
        output = generate(
            model, tokenizer, prompt + "\nA: ",
            max_new_chars=min(80, len(expected) + 12), device=device,
            stop_on_newline=True, use_reasoning_router=False,
        ).strip()
        if index < args.show:
            print(f"[{kind}] expected={expected.strip()} | output={output}")
        counts[kind] += 1
        correct[kind] += output == expected.strip()
    total = sum(counts.values())
    print({kind: f"{correct[kind]}/{counts[kind]}" for kind in sorted(counts)})
    print(f"total: {sum(correct.values())}/{total}")


if __name__ == "__main__":
    main()

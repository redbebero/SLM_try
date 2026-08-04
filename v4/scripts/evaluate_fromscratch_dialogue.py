"""Evaluate a from-scratch checkpoint on held-out public dialogue prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import torch

from chat import generate, load_model
from tokenizer import KoJamoTokenizer


def read_sft(path: Path) -> list[tuple[str, str]]:
    records = []
    for block in path.read_text(encoding="utf-8").split("\n\n"):
        if not block.startswith("Q: ") or "\nA: " not in block:
            continue
        question, answer = block[3:].split("\nA: ", 1)
        records.append((question, answer.strip()))
    return records


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    valid = read_sft(Path(args.valid))
    train = read_sft(Path(args.train))
    rng = random.Random(args.seed)
    rng.shuffle(valid)
    selected = valid[:args.count]
    train_answers = {normalize(answer) for _, answer in train}

    tokenizer = KoJamoTokenizer()
    model, device = load_model(args.checkpoint, tokenizer.get_vocab_sizes())
    results = []
    for question, expected in selected:
        prompt = question + "\nA: "
        with torch.no_grad():
            actual = generate(
                model, tokenizer, prompt, max_new_chars=120, device=device,
                stop_on_newline=True, temperature=0.0,
                repetition_penalty=1.12, use_reasoning_router=False,
            ).strip()
        results.append({
            "question": question,
            "expected": expected,
            "generated": actual,
            "empty": not bool(actual),
            "repeated_train_answer": normalize(actual) in train_answers,
            "repeat_ratio": (
                0.0 if not actual else 1.0 - len(set(actual)) / max(1, len(actual))
            ),
        })

    summary = {
        "checkpoint": args.checkpoint,
        "valid_count": len(valid),
        "evaluated": len(results),
        "empty_rate": sum(x["empty"] for x in results) / max(1, len(results)),
        "train_answer_copy_rate": sum(x["repeated_train_answer"] for x in results) / max(1, len(results)),
        "mean_repeat_ratio": sum(x["repeat_ratio"] for x in results) / max(1, len(results)),
        "prompt_hashes": [hashlib.sha1(x["question"].encode()).hexdigest() for x in results],
        "results": results,
    }
    Path(args.output).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False))
    for row in results[:8]:
        print(f"Q: {row['question']}\nA*: {row['generated']}\nA(정답): {row['expected']}\n")


if __name__ == "__main__":
    main()

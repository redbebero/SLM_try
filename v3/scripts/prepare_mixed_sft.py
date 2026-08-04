"""Combine short focused dialogue and filtered QA into one deduplicated SFT set."""

import argparse
import random
import re
from pathlib import Path


def read_pairs(path):
    text = Path(path).read_text(encoding="utf-8")
    pairs = []
    for block in re.split(r"\n\s*\n", text):
        q = re.search(r"(?m)^Q:\s*(.+)$", block)
        a = re.search(r"(?m)^A:\s*(.+)$", block)
        if q and a:
            pairs.append((q.group(1).strip(), a.group(1).strip()))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--focused", required=True)
    parser.add_argument("--qa", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--focused-count", type=int, default=1200)
    parser.add_argument("--qa-count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    focused = read_pairs(args.focused)[:args.focused_count]
    qa = read_pairs(args.qa)[:args.qa_count]
    pairs = list(dict.fromkeys(focused + qa))
    random.Random(args.seed).shuffle(pairs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mixed_sft.txt").write_text(
        "\n\n".join(f"Q: {q}\nA: {a}" for q, a in pairs) + "\n",
        encoding="utf-8",
    )
    print(f"focused={len(focused)} qa={len(qa)} unique={len(pairs)}")


if __name__ == "__main__":
    main()

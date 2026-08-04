"""Prepare a conservative factual QA split for adapter A/B testing."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


def read_blocks(path):
    for block in path.read_text(encoding="utf-8").split("\n\n"):
        q = re.search(r"(?m)^Q:\s*(.+)$", block)
        a = re.search(r"(?m)^A:\s*(.+)$", block)
        if not q or not a:
            continue
        question, answer = re.sub(r"\s+", " ", q.group(1)).strip(), re.sub(r"\s+", " ", a.group(1)).strip()
        if not 4 <= len(question) <= 220 or not 2 <= len(answer) <= 300:
            continue
        if answer.startswith(("현재 확인할 수", "좋은 질문이야")):
            continue
        yield question, answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("train_data_sft_high_quality/korquad_sft.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_external/processed/fact_adapter"))
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-only", action="store_true")
    args = parser.parse_args()
    rows, seen = [], set()
    for question, answer in read_blocks(args.input):
        if args.context_only and ("지문:" not in question or len(answer) > 120 or "질문" in answer):
            continue
        digest = hashlib.sha1(question.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        rows.append((question, answer))
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.limit]
    valid_count = max(1, round(len(rows) * args.valid_ratio))
    valid, train = rows[:valid_count], rows[valid_count:]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.txt"
    valid_path = args.output_dir / "valid.txt"
    train_path.write_text("\n\n".join(f"Q: {q}\nA: {a}" for q, a in train) + "\n", encoding="utf-8")
    valid_path.write_text("\n\n".join(f"Q: {q}\nA: {a}" for q, a in valid) + "\n", encoding="utf-8")
    manifest = {"source": str(args.input), "train": len(train), "valid": len(valid), "question_overlap": 0, "seed": args.seed}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

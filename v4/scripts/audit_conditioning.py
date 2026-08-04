"""Audit source balance, duplicate pairs, and tokenizer length distribution."""

import argparse
import collections
import json
import statistics
from pathlib import Path

from train_subword_lm import SubwordTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-len", type=int, default=256)
    args = parser.parse_args()
    tokenizer = SubwordTokenizer(args.tokenizer)
    result = {}
    for split in ("train", "valid", "test"):
        rows = [json.loads(line) for line in (Path(args.root) / split / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        lengths = [len(tokenizer.encode(f"Q: {r['question']}<answer>{r['answer']}<eos>")) for r in rows]
        answers = [r["answer"] for r in rows]
        result[split] = {
            "rows": len(rows),
            "sources": dict(collections.Counter(r.get("source", "") for r in rows)),
            "question_duplicates": len(rows) - len({r["question"] for r in rows}),
            "answer_duplicates": len(rows) - len(set(answers)),
            "mean_tokens": round(statistics.mean(lengths), 2),
            "median_tokens": statistics.median(lengths),
            "p95_tokens": sorted(lengths)[int(0.95 * (len(lengths) - 1))],
            "max_tokens": max(lengths),
            "over_max_len": sum(length > args.max_len for length in lengths),
            "first_answer_piece": dict(collections.Counter(tokenizer.processor.id_to_piece(int(tokenizer.encode(r["answer"])[0])) for r in rows).most_common(10)),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

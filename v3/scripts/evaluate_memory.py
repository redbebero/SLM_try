"""Measure positive retrieval and false-positive retrieval on JSONL queries."""

import argparse
import json
from pathlib import Path

from knowledge_memory import load_sft_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-files", nargs="+", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--threshold", type=float, default=0.20)
    args = parser.parse_args()
    memory = load_sft_memory(args.memory_files)
    cases = [json.loads(line) for line in Path(args.cases).read_text(encoding="utf-8").splitlines() if line.strip()]
    hits = 0
    false_hits = 0
    rows = []
    for case in cases:
        result = memory.retrieve(case["prompt"], args.threshold)
        hit = result is not None
        if case.get("should_retrieve", True):
            hits += hit
        else:
            false_hits += hit
        rows.append({"id": case["id"], "retrieved": hit, "result": result})
    positives = sum(case.get("should_retrieve", True) for case in cases)
    negatives = len(cases) - positives
    print(json.dumps({
        "cases": len(cases),
        "positive_recall": hits / max(1, positives),
        "negative_false_positive_rate": false_hits / max(1, negatives),
        "rows": rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

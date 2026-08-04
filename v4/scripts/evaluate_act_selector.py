"""Compare unconstrained and dialogue-act-conditioned retrieval."""

import argparse
import json
from pathlib import Path

from act_conditioned_selector import ActConditionedSelector
from conditional_selector import ConditionalSelector


def load(path):
    return [json.loads(line) for line in (Path(path) / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def score(selector, queries):
    exact = prefix = 0
    for row in queries:
        result = selector.select(row["question"], min_score=0.0)
        if result is not None:
            exact += result["answer"] == row["answer"]
            prefix += result["answer"][:10] == row["answer"][:10]
    total = max(1, len(queries))
    return {"exact": exact, "prefix10": prefix, "rows": len(queries),
            "exact_rate": exact / total, "prefix10_rate": prefix / total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--valid-dir", required=True)
    args = parser.parse_args()
    train, valid = load(args.train_dir), load(args.valid_dir)
    report = {
        "baseline": score(ConditionalSelector(train), valid),
        "act_conditioned": score(ActConditionedSelector(train), valid),
    }
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Check exact normalized pair leakage between downloaded-data splits."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def normalized(record):
    question = re.sub(r"\s+", " ", record["question"].strip().lower())
    answer = re.sub(r"\s+", " ", record["answer"].strip().lower())
    return hashlib.sha256(f"{question}\n{answer}".encode("utf-8")).hexdigest()


def load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--test", required=True)
    args = parser.parse_args()
    splits = {name: load(path) for name, path in {
        "train": args.train, "valid": args.valid, "test": args.test,
    }.items()}
    fingerprints = {name: {normalized(row) for row in rows} for name, rows in splits.items()}
    failures = {}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = fingerprints[left] & fingerprints[right]
        failures[f"{left}_{right}"] = len(overlap)
    source_counts = {
        name: dict(sorted({row["source"]: 0 for row in rows}.items()))
        for name, rows in splits.items()
    }
    for name, rows in splits.items():
        for row in rows:
            source_counts[name][row["source"]] += 1
    result = {
        "rows": {name: len(rows) for name, rows in splits.items()},
        "exact_overlap": failures,
        "source_counts": source_counts,
        "pass": all(value == 0 for value in failures.values()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

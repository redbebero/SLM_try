"""Format downloaded Q/A records for the existing syllable decoder."""

import argparse
import json
import random
from pathlib import Path


def convert(source, destination, limit=0):
    records = [json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        random.Random(42).shuffle(records)
        records = records[:limit]
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_text(
        "\n".join(f"Q: {row['question']}|{row['answer']}" for row in records) + "\n",
        encoding="utf-8",
    )
    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="experiments/downloaded_sft_short_v2")
    parser.add_argument("--output-root", default="experiments/downloaded_syllable_short")
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()
    counts = {}
    for split in ("train", "valid"):
        counts[split] = convert(
            f"{args.source_root}/{split}/records.jsonl",
            f"{args.output_root}/{split}/data.txt",
            args.limit if args.limit else 0,
        )
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

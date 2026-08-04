"""Audit token lengths and round-trip behavior for downloaded SFT splits."""

import argparse
import json
import statistics
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizer import KoJamoTokenizer


def audit(path, tokenizer, max_tokens):
    rows = []
    truncated = 0
    round_trip_failures = 0
    records_path = Path(path).with_name("records.jsonl")
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for record in records:
        sample = f"Q: {record['question']}\nA: {record['answer']}\n"
        rows.append(sample)
        encoded = tokenizer.encode(sample)
        if len(encoded) > max_tokens:
            truncated += 1
        if tokenizer.decode(encoded) != sample:
            round_trip_failures += 1
    lengths = [len(tokenizer.encode(row)) for row in rows]
    return {
        "rows": len(rows),
        "max_tokens": max_tokens,
        "truncated_rows": truncated,
        "round_trip_failures": round_trip_failures,
        "min_tokens": min(lengths) if lengths else 0,
        "median_tokens": statistics.median(lengths) if lengths else 0,
        "p95_tokens": sorted(lengths)[int(0.95 * (len(lengths) - 1))] if lengths else 0,
        "max_observed_tokens": max(lengths) if lengths else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments/downloaded_sft")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tokenizer = KoJamoTokenizer()
    result = {
        split: audit(Path(args.root) / split / "verified.txt", tokenizer, args.max_tokens)
        for split in ("train", "valid", "test")
    }
    result["round_trip_examples"] = {
        text: tokenizer.decode(tokenizer.encode(text))
        for text in ("가나다", "각힣", "ㄱㅏㄳ", "띄어 쓰기 123 ABC!?")
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Build category-specific SFT splits by filtering downloaded records only."""

import argparse
import json
from pathlib import Path


def load_rows(root, split):
    path = Path(root) / split / "records.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def keep_row(row, max_question=0, max_answer=0):
    return ((not max_question or len(row.get("question", "")) <= max_question)
            and (not max_answer or len(row.get("answer", "")) <= max_answer))


def write_split(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    path.joinpath("records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    path.joinpath("verified.txt").write_text(
        "\n\n".join(f"Q: {row['question']}\nA: {row['answer']}" for row in rows) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--max-question", type=int, default=0)
    parser.add_argument("--max-answer", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output_root)
    stats = {}
    for split in ("train", "valid", "test"):
        rows = [row for row in load_rows(args.input_root, split)
                if row.get("category") == args.category
                and keep_row(row, args.max_question, args.max_answer)]
        write_split(output / split, rows)
        stats[split] = len(rows)
    output.joinpath("MANIFEST.json").write_text(
        json.dumps({"category": args.category, "source_root": args.input_root,
                    "splits": stats}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()

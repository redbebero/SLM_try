"""Create train/valid dirs containing the same real downloaded record."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.source)
    output = Path(args.output)
    row = source.read_text(encoding="utf-8").splitlines()[0]
    record = json.loads(row)
    if not record.get("question") or not record.get("answer"):
        raise ValueError("source row must contain question and answer")
    for split in ("train", "valid"):
        target = output / split
        target.mkdir(parents=True, exist_ok=True)
        (target / "records.jsonl").write_text(row + "\n", encoding="utf-8")
        (target / "data.txt").write_text(
            f"Q: {record['question']}\nA: {record['answer']}\n",
            encoding="utf-8",
        )
    print(f"created {output} from {record['source']}:{record['source_id']}")


if __name__ == "__main__":
    main()

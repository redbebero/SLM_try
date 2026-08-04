"""Merge downloaded SFT sources by split with hash-based deduplication."""
import argparse
import json
from pathlib import Path


def load(path):
    return [json.loads(line) for line in (path / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--extra", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base, extra, output = Path(args.base), Path(args.extra), Path(args.output)
    splits = {}
    for name in ("train", "valid", "test"):
        rows = load(base / name) + load(extra / name)
        unique = {}
        for row in rows:
            unique[row["pair_hash"]] = row
        splits[name] = list(unique.values())
        target = output / name
        target.mkdir(parents=True, exist_ok=True)
        (target / "records.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in splits[name]),
            encoding="utf-8",
        )
        (target / "verified.txt").write_text(
            "\n\n".join(f"Q: {row['question']}\nA: {row['answer']}" for row in splits[name]) + "\n",
            encoding="utf-8",
        )
    split_hashes = {name: {row["pair_hash"] for row in rows} for name, rows in splits.items()}
    manifest = {
        "base": str(base), "extra": str(extra),
        "splits": {name: len(rows) for name, rows in splits.items()},
        "overlap": {f"{left}-{right}": len(split_hashes[left] & split_hashes[right])
                    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

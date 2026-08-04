"""Add a train-only nearest-example context to each SFT question."""

import argparse
import json
from pathlib import Path

from retrieval import TfidfCharNgramRetriever


def load(path):
    return [json.loads(line) for line in (Path(path) / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def augment(rows, retriever, exclude_self=False):
    output = []
    for row in rows:
        hits = retriever.search(
            row["question"], top_k=3,
            source=row.get("source"), category=row.get("category"),
        )
        if exclude_self:
            hits = [hit for hit in hits if hit.get("pair_hash") != row.get("pair_hash") and hit["question"] != row["question"]]
        context = hits[0] if hits else None
        copied = dict(row)
        if context:
            copied["question"] = (
                f"참고 질문: {context['question']}\n"
                f"참고 답변: {context['answer']}\n"
                f"현재 질문: {row['question']}"
            )
        output.append(copied)
    return output


def write_split(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    (path / "records.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    input_root, output_root = Path(args.input_root), Path(args.output_root)
    train = load(input_root / "train")
    retriever = TfidfCharNgramRetriever(train)
    write_split(output_root / "train", augment(train, retriever, exclude_self=True))
    for split in ("valid", "test"):
        write_split(output_root / split, augment(load(input_root / split), retriever))
    print(json.dumps({split: len(load(output_root / split)) for split in ("train", "valid", "test")}))


if __name__ == "__main__":
    main()

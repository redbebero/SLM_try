"""Filter Korean safe-conversation records for short, natural SFT examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


_AI_DISCLAIMER = re.compile(
    r"(?:저는|나는).{0,24}(?:인공지능|AI|챗봇|로봇).{0,80}(?:라서|이므로|때문|못|할 수 없)",
    re.IGNORECASE,
)
_REPEATED_SENTENCE = re.compile(r"([^.!?。！？\n]{4,}[.!?。！？])\s*\1")


def _normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_record(record: dict, *, max_question: int = 300, max_answer: int = 400):
    question = _normalize(record.get("instruction"))
    extra = _normalize(record.get("input"))
    answer = _normalize(record.get("output"))
    if extra:
        question = f"{question}\n{extra}" if question else extra
    if not (8 <= len(question) <= max_question and 8 <= len(answer) <= max_answer):
        return None
    if _AI_DISCLAIMER.search(answer) or _REPEATED_SENTENCE.search(answer):
        return None
    if answer.count("\n") > 2 or answer.count("목록") > 0:
        return None
    return {
        "messages": [{"role": "user", "content": question}],
        "answer": answer,
    }


def split_records(records: list[dict], valid_ratio: float = 0.05):
    train, valid = [], []
    for record in records:
        key = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
        bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
        (valid if bucket < valid_ratio else train).append(record)
    return train, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    args = parser.parse_args()

    records = []
    total = 0
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        item = clean_record(json.loads(line))
        if item is not None:
            records.append(item)
    unique = {}
    for item in records:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        unique[key] = item
    train, valid = split_records(list(unique.values()), args.valid_ratio)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("valid.jsonl", valid)):
        with (args.output_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "source": str(args.input),
        "raw_records": total,
        "clean_unique_records": len(unique),
        "train_records": len(train),
        "valid_records": len(valid),
        "filters": {"max_question": 300, "max_answer": 400, "ai_disclaimer": True},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

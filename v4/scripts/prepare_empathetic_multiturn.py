"""Prepare public Empathetic_data into context-preserving Korean SFT records."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


MARKER = re.compile(r"(?m)^(질문|답변)\s*:\s*")
BAD = re.compile(
    r"(https?://|www\.|인공지능\s*언어\s*모델|질문\s*:.*질문\s*:|답변\s*:.*답변\s*:|�)",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def parse_instruction(instruction: str, answer: str) -> dict:
    """Convert labeled turns into messages plus the final target answer."""
    instruction = str(instruction or "").strip()
    answer = _clean(answer)
    matches = list(MARKER.finditer(instruction))
    if not matches:
        question = _clean(instruction)
        return {"messages": [{"role": "user", "content": question}], "answer": answer}
    messages = []
    for index, match in enumerate(matches):
        content = _clean(instruction[match.end(): matches[index + 1].start() if index + 1 < len(matches) else len(instruction)])
        if not content:
            continue
        role = "user" if match.group(1) == "질문" else "assistant"
        messages.append({"role": role, "content": content})
    return {"messages": messages, "answer": answer}


def acceptable(row: dict) -> bool:
    messages = row.get("messages", [])
    answer = _clean(row.get("answer", ""))
    if not messages or messages[-1].get("role") != "user":
        return False
    if not (16 <= len(answer) <= 420):
        return False
    if BAD.search(answer) or any(BAD.search(str(item.get("content", ""))) for item in messages):
        return False
    if any(not (2 <= len(str(item.get("content", ""))) <= 420) for item in messages):
        return False
    if len(re.findall(r"[가-힣]", answer)) < 8:
        return False
    if re.search(r"(.{4,})\s*\1", answer):
        return False
    if len(messages) > 9:
        return False
    row["answer"] = answer
    return True


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        row = parse_instruction(item.get("instruction", ""), item.get("output", ""))
        if acceptable(row):
            rows.append(row)
    unique = {}
    for row in rows:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True)
        unique.setdefault(key, row)
    return list(unique.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data_external/raw/Empathetic_data.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = load(args.input)
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.limit]
    valid = max(1, round(len(rows) * 0.1))
    train, dev = rows[valid:], rows[:valid]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, items in (("train.jsonl", train), ("valid.jsonl", dev)):
        with (args.output_dir / name).open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest = {
        "source": str(args.input), "clean_unique": len(rows),
        "train": len(train), "valid": len(dev), "seed": args.seed,
        "split": "conversation_record_hash_after_shuffle",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

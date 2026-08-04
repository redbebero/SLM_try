"""Extract conservative Korean human-assistant pairs from ShareGPT-74k-ko."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

AI = re.compile(r"(저는|나는).{0,30}(인공지능|AI|언어 모델|챗봇).{0,60}(라서|이므로|때문|못|할 수 없)", re.I)
BAD = re.compile(r"(\.{3,}|\?{3,}|!{3,}|\bhttps?://|\bwww\.|###|\*\*|\|.*\|)")


def korean_ratio(text: str) -> float:
    letters = re.findall(r"[가-힣A-Za-z]", text)
    return sum("가" <= c <= "힣" for c in letters) / max(1, len(letters))


def accept(question: str, answer: str) -> bool:
    q, a = question.strip(), answer.strip()
    if not (8 <= len(q) <= 240 and 8 <= len(a) <= 360):
        return False
    if korean_ratio(q + " " + a) < 0.55 or AI.search(a) or BAD.search(a):
        return False
    if len(set(a.split())) < 4 or len(re.findall(r"[.!?。！？]", a)) > 8:
        return False
    return True


def split(rows):
    train, valid = [], []
    for row in rows:
        bucket = int.from_bytes(hashlib.sha256(row["answer"].encode()).digest()[:8], "big") / 2**64
        (valid if bucket < 0.05 else train).append(row)
    return train, valid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=5000)
    args = p.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows, seen = [], set()
    for item in data:
        turns = item.get("conversations", [])
        for left, right in zip(turns, turns[1:]):
            if left.get("from") != "human" or right.get("from") != "gpt":
                continue
            q, a = left.get("value", ""), right.get("value", "")
            key = (q.strip(), a.strip())
            if key in seen or not accept(*key):
                continue
            seen.add(key)
            rows.append({"messages": [{"role": "user", "content": key[0]}], "answer": key[1]})
            if len(rows) >= args.limit:
                break
        if len(rows) >= args.limit:
            break
    train, valid = split(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in (("train.jsonl", train), ("valid.jsonl", valid)):
        with (args.output_dir / name).open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {"raw_conversations": len(data), "selected": len(rows), "train": len(train), "valid": len(valid)}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

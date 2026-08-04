"""Download-independent cleaning and SFT formatting for public Korean dialogue data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from random import Random


URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPACE_RE = re.compile(r"\s+")
REPEAT_RE = re.compile(r"(.)\1{6,}")


def clean_text(value: str) -> str:
    text = CONTROL_RE.sub(" ", str(value or ""))
    if URL_RE.search(text):
        return ""
    text = SPACE_RE.sub(" ", text).strip()
    if REPEAT_RE.search(text):
        return ""
    return text


def _valid_pair(question: str, answer: str) -> bool:
    if not question or not answer or question == answer:
        return False
    if not (2 <= len(question) <= 220 and 2 <= len(answer) <= 320):
        return False
    # Reject prompt leakage and obvious non-dialogue artifacts.
    if answer.startswith(("질문:", "Q:", "사용자:")):
        return False
    return True


def load_empathetic_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            question = clean_text(item.get("instruction", ""))
            answer = clean_text(item.get("output", ""))
            if _valid_pair(question, answer):
                rows.append({"question": question, "answer": answer, "source": "empathetic"})
    return rows


def load_chatbot_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for item in csv.DictReader(handle):
            question = clean_text(item.get("Q", ""))
            answer = clean_text(item.get("A", ""))
            if _valid_pair(question, answer):
                rows.append({"question": question, "answer": answer, "source": "chatbot"})
    return rows


def _deduplicate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result = []
    seen = set()
    for row in rows:
        key = (row["question"], row["answer"])
        digest = hashlib.sha1("\t".join(key).encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        result.append(row)
    return result


def make_sft_rows(rows: list[dict[str, str]], max_items: int | None = None) -> list[str]:
    blocks = [f"Q: {row['question']}\nA: {row['answer']}" for row in _deduplicate(rows)]
    return blocks[:max_items] if max_items is not None else blocks


def _split(rows: list[dict[str, str]], validation_ratio: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = _deduplicate(rows)
    Random(seed).shuffle(rows)
    valid_count = max(1, round(len(rows) * validation_ratio))
    return rows[valid_count:], rows[:valid_count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--empathetic", type=Path, default=Path("data_external/raw/Empathetic_data.jsonl"))
    parser.add_argument("--chatbot", type=Path, default=Path("data_external/raw/ChatbotData.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_external/processed"))
    parser.add_argument("--max-train", type=int, default=8000)
    parser.add_argument("--max-valid", type=int, default=1000)
    args = parser.parse_args()

    empathetic = load_empathetic_rows(args.empathetic)
    chatbot = load_chatbot_rows(args.chatbot)
    # Keep synthetic empathy data useful but prevent it from dominating the tiny model.
    empathetic = empathetic[:6000]
    chatbot = chatbot[:3000]
    train, valid = _split(empathetic + chatbot, validation_ratio=0.1, seed=42)
    train_blocks = make_sft_rows(train, args.max_train)
    valid_blocks = make_sft_rows(valid, args.max_valid)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "external_dialogue_sft.txt").write_text("\n\n".join(train_blocks) + "\n", encoding="utf-8")
    (args.output_dir / "external_dialogue_valid.txt").write_text("\n\n".join(valid_blocks) + "\n", encoding="utf-8")
    manifest = {
        "sources": {"empathetic": len(empathetic), "chatbot": len(chatbot)},
        "train": len(train_blocks),
        "valid": len(valid_blocks),
        "seed": 42,
        "filters": {"question_chars": [2, 220], "answer_chars": [2, 320], "url": "reject", "repeated_chars": 7},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

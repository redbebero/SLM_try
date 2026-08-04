"""Convert official KorQuAD extractive QA into conservative lexical memory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_items(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    for article in payload.get("data", []):
        for paragraph in article.get("paragraphs", []):
            for qa in paragraph.get("qas", []):
                answers = qa.get("answers") or []
                if answers:
                    yield qa.get("question", "").strip(), answers[0].get("text", "").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = {}
    for path in args.input:
        for question, answer in read_items(path):
            if 4 <= len(question) <= 300 and 1 <= len(answer) <= 300:
                pairs.setdefault(question, answer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for question, answer in pairs.items():
            handle.write(f"Q: {question}\nA: {answer}\n\n")
    print(json.dumps({"pairs": len(pairs), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

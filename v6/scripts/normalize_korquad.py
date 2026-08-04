import argparse
import html
import json
import re
from pathlib import Path


def clean(text: str) -> str:
    text = re.sub(r"</?(?:p|div|br|li|tr|td|th)[^>]*>", " ", text, flags=re.IGNORECASE)
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    return re.sub(r"\s+", " ", text).strip()


def normalize_item(item):
    context = clean(item.get("context", ""))
    answer = item.get("answer", {})
    answer_text = answer.get("text", "") if isinstance(answer, dict) else str(answer)
    question = f"문서: {context}\n질문: {item['question']}"
    return {
        "id": f"korquad-{item['id']}",
        "category": "document_qa",
        "question": question,
        "solution": f"문서에서 질문과 관련된 내용을 찾아 답합니다: {answer_text}.",
        "answer": answer_text,
        "template_id": f"korquad-{item['id']}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for line in args.input.open(encoding="utf-8"):
            if args.limit and count >= args.limit:
                break
            row = normalize_item(json.loads(line))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()

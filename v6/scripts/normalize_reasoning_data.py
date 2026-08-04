import argparse
import json
from pathlib import Path


def normalize(item, source, license_name):
    answer = item.get("answer", "")
    if isinstance(answer, dict):
        answer = answer.get("text", "")
    return {
        "id": str(item["id"]),
        "question": str(item["question"]).strip(),
        "reasoning": str(item.get("reasoning", item.get("solution", ""))).strip(),
        "solution": str(item.get("solution", item.get("reasoning", ""))).strip(),
        "answer": str(answer).strip(),
        "category": str(item.get("category", "general")),
        "template_id": str(item.get("template_id", item["id"])),
        "source": source,
        "license": license_name,
        "verifier": str(item.get("verifier", "text")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--license", required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = [normalize(json.loads(line), args.source, args.license) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

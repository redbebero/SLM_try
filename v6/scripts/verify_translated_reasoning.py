import argparse
import json
import re
from pathlib import Path


def verify_row(row):
    question = str(row.get("question", "")).strip()
    solution = str(row.get("solution", row.get("reasoning", ""))).strip()
    answer = str(row.get("answer", "")).strip()
    if not question or not solution:
        return "empty question or solution"
    if not answer:
        return "empty answer"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", answer):
        if not re.search(rf"(?<!\d){re.escape(answer)}(?!\d)", solution):
            return "answer missing from solution"
    elif answer not in solution:
        return "answer missing from solution"
    return None


def verify(input_path: Path, output_path: Path):
    good, rejected = [], []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        reason = verify_row(row)
        (good if reason is None else rejected).append(row if reason is None else {"row": row, "reason": reason})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in good), encoding="utf-8")
    output_path.with_name(output_path.stem + "_rejected.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rejected), encoding="utf-8")
    print(f"accepted {len(good)} rejected {len(rejected)}")
    return good, rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.input, args.output)


if __name__ == "__main__":
    main()

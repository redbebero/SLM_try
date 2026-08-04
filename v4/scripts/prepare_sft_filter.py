"""Filter mixed Korean SFT records into short, direct QA examples."""

import argparse
import re
from pathlib import Path


def read_pairs(path):
    text = Path(path).read_text(encoding="utf-8")
    pairs = []
    for block in re.split(r"\n\s*\n", text):
        q = re.search(r"(?m)^Q:\s*(.+)$", block)
        a = re.search(r"(?m)^A:\s*(.+)$", block)
        if q and a:
            pairs.append((q.group(1).strip(), a.group(1).strip()))
    return pairs


def clean_pairs(pairs, max_question_chars=220, max_answer_chars=80):
    result = []
    seen = set()
    for question, answer in pairs:
        if not question or not answer or len(question) > max_question_chars:
            continue
        if len(answer) > max_answer_chars or "####" in answer:
            continue
        if answer.count("\"") >= 2 or answer.count("“") + answer.count("”") >= 2:
            continue
        # Reject long roleplay continuations while retaining concise dialogue.
        if any(marker in answer for marker in ("왓슨", "면접관", "(웃", "그가 ", "그녀가 ")):
            continue
        if not re.search(r"[가-힣0-9A-Za-z]", answer):
            continue
        item = (question, answer)
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=5000)
    args = parser.parse_args()
    pairs = clean_pairs(read_pairs(args.input))
    if len(pairs) < args.count:
        raise ValueError(f"filtered={len(pairs)} < requested={args.count}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "filtered_sft.txt"
    output.write_text(
        "\n\n".join(f"Q: {q}\nA: {a}" for q, a in pairs[:args.count]) + "\n",
        encoding="utf-8",
    )
    print(f"raw={len(read_pairs(args.input))} filtered={len(pairs)} written={args.count}")


if __name__ == "__main__":
    main()

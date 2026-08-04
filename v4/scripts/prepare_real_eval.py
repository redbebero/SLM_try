"""Build a deterministic real-QA holdout from filtered SFT records."""

import argparse
import json
import random
import re
from pathlib import Path


def pairs(path):
    text = Path(path).read_text(encoding="utf-8")
    result = []
    for block in re.split(r"\n\s*\n", text):
        q = re.search(r"(?m)^Q:\s*(.+)$", block)
        a = re.search(r"(?m)^A:\s*(.+)$", block)
        if q and a:
            question, answer = q.group(1).strip(), a.group(1).strip()
            if 2 <= len(answer) <= 35 and len(question) <= 220:
                result.append((question, answer))
    return list(dict.fromkeys(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args()
    source = pairs(args.input)
    rng = random.Random(args.seed)
    rng.shuffle(source)
    cases = []
    for index, (question, answer) in enumerate(source[:args.count]):
        keyword = re.sub(r"[^0-9A-Za-z가-힣]+", "", answer.split()[0])
        if len(keyword) < 2:
            continue
        cases.append({
            "id": f"real_{index:03d}",
            "prompt": question,
            "expected_keywords": [keyword],
            "max_new_chars": min(100, max(40, len(answer) * 3)),
        })
    Path(args.output).write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )
    print(f"source={len(source)} cases={len(cases)}")


if __name__ == "__main__":
    main()

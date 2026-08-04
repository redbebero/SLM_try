"""Build a held-out prefix set using the answer vocabulary of an existing split."""

import argparse
from collections import defaultdict
from pathlib import Path

from build_prefix_dataset import build


def read_rows(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--used-valid", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--min-prefix", type=int, default=16)
    parser.add_argument("--max-prefix", type=int, default=40)
    parser.add_argument("--max-answer", type=int, default=4)
    args = parser.parse_args()

    train = read_rows(args.train)
    used = set(read_rows(args.used_valid))
    answers = {row.rsplit(" ", 1)[1] for row in train if " " in row}
    groups = defaultdict(list)
    seen = set()
    candidates = build(
        args.source, 10**9, args.min_prefix, args.max_prefix,
        args.max_answer, allowed_answers=answers,
    )
    for row in candidates:
        if row in used or " " not in row:
            continue
        answer = row.rsplit(" ", 1)[1]
        if row in seen:
            continue
        seen.add(row)
        groups[answer].append(row)
    rows = []
    answer_order = sorted(groups)
    while answer_order and len(rows) < args.limit:
        next_order = []
        for answer in answer_order:
            if groups[answer]:
                rows.append(groups[answer].pop(0))
                if len(rows) >= args.limit:
                    break
            if groups[answer]:
                next_order.append(answer)
        answer_order = next_order
    root = Path(args.output)
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "valid").mkdir(parents=True, exist_ok=True)
    (root / "train/train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (root / "valid/valid.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"train={len(train)} holdout={len(rows)} answers={len(answers)}")


if __name__ == "__main__":
    main()

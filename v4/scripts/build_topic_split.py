"""Build train/holdout splits from disjoint corpus line ranges."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path


def candidates(path, start, end, min_prefix, max_prefix, max_answer):
    rows = []
    seen = set()
    lines = Path(path).read_text(encoding="utf-8").splitlines()[start:end]
    for raw in lines:
        words = raw.strip().split()
        for cut in range(4, len(words)):
            prefix = " ".join(words[:cut])
            answer = words[cut]
            row = f"{prefix} {answer}"
            if not min_prefix <= len(prefix) <= max_prefix or len(answer) > max_answer:
                continue
            if row not in seen:
                seen.add(row)
                rows.append(row)
    return rows


def balanced(rows, answers, limit):
    groups = defaultdict(list)
    for row in rows:
        if row.rsplit(" ", 1)[1] in answers:
            groups[row.rsplit(" ", 1)[1]].append(row)
    selected = []
    order = sorted(groups)
    while order and len(selected) < limit:
        next_order = []
        for answer in order:
            if groups[answer]:
                selected.append(groups[answer].pop(0))
                if len(selected) >= limit:
                    break
            if groups[answer]:
                next_order.append(answer)
        order = next_order
    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="train_data_clean/pretrain_train.txt")
    p.add_argument("--train-start", type=int, default=0)
    p.add_argument("--train-end", type=int, default=60000)
    p.add_argument("--valid-start", type=int, default=60000)
    p.add_argument("--valid-end", type=int, default=90000)
    p.add_argument("--answer-types", type=int, default=50)
    p.add_argument("--limit-train", type=int, default=2000)
    p.add_argument("--limit-valid", type=int, default=200)
    p.add_argument("--min-prefix", type=int, default=16)
    p.add_argument("--max-prefix", type=int, default=40)
    p.add_argument("--max-answer", type=int, default=4)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    train_rows = candidates(args.source, args.train_start, args.train_end, args.min_prefix, args.max_prefix, args.max_answer)
    valid_rows = candidates(args.source, args.valid_start, args.valid_end, args.min_prefix, args.max_prefix, args.max_answer)
    train_counts = Counter(row.rsplit(" ", 1)[1] for row in train_rows)
    valid_counts = Counter(row.rsplit(" ", 1)[1] for row in valid_rows)
    common = set(train_counts) & set(valid_counts)
    answers = {a for a, _ in valid_counts.most_common() if a in common}
    answers = set(sorted(answers, key=lambda a: (-valid_counts[a], a))[:args.answer_types])
    train = balanced(train_rows, answers, args.limit_train)
    valid = balanced(valid_rows, answers, args.limit_valid)
    root = Path(args.output)
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "valid").mkdir(parents=True, exist_ok=True)
    (root / "train/train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (root / "valid/valid.txt").write_text("\n".join(valid) + "\n", encoding="utf-8")
    print(f"train={len(train)} valid={len(valid)} answers={len(answers)}")


if __name__ == "__main__":
    main()

"""Build prefix-to-next-word samples from the clean downloaded corpus."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path


def build(source, limit, min_prefix, max_prefix, max_answer, allowed_answers=None):
    rows = []
    seen = set()
    for raw in Path(source).read_text(encoding="utf-8").splitlines():
        words = raw.strip().split()
        for cut in range(4, len(words)):
            prefix = " ".join(words[:cut])
            answer = words[cut]
            if not min_prefix <= len(prefix) <= max_prefix:
                continue
            if not answer or len(answer) > max_answer:
                continue
            if allowed_answers is not None and answer not in allowed_answers:
                continue
            row = f"{prefix} {answer}"
            if row not in seen:
                seen.add(row)
                rows.append(row)
    return rows[:limit]


def balance_by_answer(rows, limit):
    """Select rows round-robin by answer while preserving source order per answer."""
    groups = defaultdict(list)
    for row in rows:
        groups[row.rsplit(" ", 1)[1]].append(row)
    selected = []
    answers = sorted(groups)
    while answers and len(selected) < limit:
        next_answers = []
        for answer in answers:
            group = groups[answer]
            if group:
                selected.append(group.pop(0))
                if len(selected) >= limit:
                    break
            if group:
                next_answers.append(answer)
        answers = next_answers
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-train", type=int, default=2000)
    parser.add_argument("--limit-valid", type=int, default=200)
    parser.add_argument("--min-prefix", type=int, default=40)
    parser.add_argument("--max-prefix", type=int, default=120)
    parser.add_argument("--max-answer", type=int, default=8)
    parser.add_argument("--balance-answer", action="store_true")
    parser.add_argument("--overlap-valid", action="store_true")
    parser.add_argument("--answer-types", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.output)
    valid_candidates = build(
        "val_data_clean/pretrain_val.txt", 10**9, args.min_prefix,
        args.max_prefix, args.max_answer,
    )
    valid_answers = {row.rsplit(" ", 1)[1] for row in valid_candidates}
    if args.answer_types > 0:
        valid_counts = Counter(row.rsplit(" ", 1)[1] for row in valid_candidates)
        valid_answers = {
            answer for answer, _ in valid_counts.most_common(args.answer_types)
        }
    train = build(
        "train_data_clean/pretrain_train.txt", 10**9, args.min_prefix,
        args.max_prefix, args.max_answer,
        allowed_answers=valid_answers if args.overlap_valid else None,
    )
    if args.balance_answer:
        train = balance_by_answer(train, args.limit_train)
    else:
        train = train[:args.limit_train]
    train_answers = {row.rsplit(" ", 1)[1] for row in train}
    valid = [row for row in valid_candidates if row.rsplit(" ", 1)[1] in train_answers]
    if args.balance_answer:
        valid = balance_by_answer(valid, args.limit_valid)
    else:
        valid = valid[:args.limit_valid]
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "valid").mkdir(parents=True, exist_ok=True)
    (root / "train/train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (root / "valid/valid.txt").write_text("\n".join(valid) + "\n", encoding="utf-8")
    print(f"train={len(train)} valid={len(valid)}")


if __name__ == "__main__":
    main()

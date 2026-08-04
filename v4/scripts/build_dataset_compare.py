"""Build a deterministic small subset from the downloaded clean corpus."""

from pathlib import Path
import argparse


def select(source, limit, min_chars=12, max_chars=None):
    rows = []
    for raw in Path(source).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if len(line) >= min_chars and " " in line and (max_chars is None or len(line) <= max_chars) and line not in rows:
            rows.append(line)
        if len(rows) >= limit:
            break
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=12)
    parser.add_argument("--output", default="experiments/dataset_compare")
    args = parser.parse_args()
    root = Path(args.output)
    train = select("train_data_clean/pretrain_train.txt", 2000, args.min_chars, args.max_chars)
    valid = select("val_data_clean/pretrain_val.txt", 200, args.min_chars, args.max_chars)
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "valid").mkdir(parents=True, exist_ok=True)
    (root / "train/train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (root / "valid/valid.txt").write_text("\n".join(valid) + "\n", encoding="utf-8")
    print(f"train={len(train)} valid={len(valid)}")


if __name__ == "__main__":
    main()

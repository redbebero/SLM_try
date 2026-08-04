"""Create a deterministic randomized split of normalized QA."""

import random
from pathlib import Path


def main():
    rows = [line for line in Path("experiments/normalized_qa/train/train.tsv").read_text(encoding="utf-8").splitlines() if "\t" in line]
    rows += [line for line in Path("experiments/normalized_qa/valid/valid.tsv").read_text(encoding="utf-8").splitlines() if "\t" in line]
    rows = list(dict.fromkeys(rows))
    random.Random(42).shuffle(rows)
    split = int(len(rows) * 0.8)
    root = Path("experiments/normalized_qa_random")
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "valid").mkdir(parents=True, exist_ok=True)
    (root / "train/train.tsv").write_text("\n".join(rows[:split]) + "\n", encoding="utf-8")
    (root / "valid/valid.tsv").write_text("\n".join(rows[split:]) + "\n", encoding="utf-8")
    print(f"train={split} valid={len(rows)-split}")


if __name__ == "__main__":
    main()

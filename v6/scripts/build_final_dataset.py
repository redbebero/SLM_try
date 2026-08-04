import argparse
import json
from pathlib import Path


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    owned = Path("data/owned")
    train = read(owned / "train.jsonl")
    translated = read("data/translated/gsm8k_ko.jsonl")
    korquad = read("data/normalized/korquad_train.jsonl")
    instruction = read("data/instruction/koqa_18.jsonl")
    assert len(train) == 180 and len(translated) == 108 and len(korquad) >= 54 and len(instruction) == 18
    write(args.output_dir / "train.jsonl", train + translated + korquad[:54] + instruction)
    for split in ("dev", "test", "ood"):
        write(args.output_dir / f"{split}.jsonl", read(owned / f"{split}.jsonl"))
    print("train", 360, "dev", 60, "test", 30, "ood", 30)


if __name__ == "__main__":
    main()

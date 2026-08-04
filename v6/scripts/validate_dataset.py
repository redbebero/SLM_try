import argparse
import json
from pathlib import Path


REQUIRED = {"id", "category", "question", "solution", "answer", "template_id"}
SPLITS = ("train", "dev", "test", "ood")


def validate(data_dir: Path):
    ids = set()
    templates = {}
    counts = {}
    for split in SPLITS:
        path = data_dir / f"{split}.jsonl"
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            item = json.loads(line)
            missing = REQUIRED - item.keys()
            if missing:
                raise ValueError(f"{path}:{line_number}: missing {sorted(missing)}")
            if item["id"] in ids:
                raise ValueError(f"duplicate id: {item['id']}")
            if not str(item["answer"]).strip():
                raise ValueError(f"empty answer: {item['id']}")
            ids.add(item["id"])
            rows.append(item)
        counts[split] = len(rows)
        for item in rows:
            previous = templates.setdefault(item["template_id"], split)
            if previous != split:
                raise ValueError(f"template overlaps splits: {item['template_id']}")
    print(f"valid: {sum(counts.values())} rows; {len(templates)} templates; {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    validate(args.data_dir)

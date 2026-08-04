import argparse
import json
from collections import defaultdict
from pathlib import Path


def split_name(template_id: str) -> str:
    local = int(template_id.rsplit("-", 1)[1])
    return {0: "train", 1: "train", 2: "train", 3: "train", 4: "train", 5: "train", 6: "dev", 7: "dev", 8: "test", 9: "ood"}[local]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    groups = defaultdict(list)
    for line in args.input.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        groups[split_name(item["template_id"])].append(item)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "dev", "test", "ood"):
        path = args.output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for item in groups[name]:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(name, len(groups[name]), path)


if __name__ == "__main__":
    main()

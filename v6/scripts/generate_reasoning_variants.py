import argparse
import json
from pathlib import Path

try:
    from scripts.build_reasoning_suite import _make
except ModuleNotFoundError:
    from build_reasoning_suite import _make


CATEGORIES = ["arithmetic", "multi_step", "comparison", "state_change", "temporal_logic", "reading_inference"]


def build_variants(count=8000):
    rows = []
    for index in range(count):
        category = CATEGORIES[index % len(CATEGORIES)]
        local_index = index // len(CATEGORIES)
        row = _make(category, "train", local_index)
        row["id"] = f"owned-reasoning-{index:05d}"
        row["template_id"] = f"owned-reasoning-{category}-{local_index:04d}"
        row["source"] = "deterministic-generator"
        row["license"] = "project-owned"
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8000)
    args = parser.parse_args()
    rows = build_variants(args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

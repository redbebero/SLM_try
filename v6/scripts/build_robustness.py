import argparse
import json
from pathlib import Path

try:
    from scripts.build_reasoning_suite import _make
except ModuleNotFoundError:
    from build_reasoning_suite import _make


CATEGORIES = ["arithmetic", "multi_step", "comparison", "state_change", "temporal_logic", "reading_inference"]


def build(count_per_category=20):
    rows = []
    for category in CATEGORIES:
        for index in range(count_per_category):
            row = _make(category, "robustness", index + 2000)
            row["id"] = f"kr-robust-{category}-{index:03d}"
            row["template_id"] = f"kr-robust-{category}-{index:03d}"
            row["source"] = "project-owned-robustness"
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()

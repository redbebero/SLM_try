"""Create deterministic synthetic wording variants without changing labels."""

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .data import load_records

POSITIVE_SUFFIXES = (" 시전해", " 발동해", " 지금", " 즉시", " 힘을 모아", " 주문으로", " 마법으로")
NEGATIVE_SUFFIXES = (" 라고 말해", " 문장이다", " 적어 둔다", " 지금", " 다시 말해")


def expand_records(
    records: list[dict[str, Any]],
    *,
    positive_variations: int = len(POSITIVE_SUFFIXES),
    negative_variations: int = len(NEGATIVE_SUFFIXES),
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = list(records)
    for record in records:
        is_unknown = record["target"]["status"] == "UNKNOWN"
        suffixes = NEGATIVE_SUFFIXES if is_unknown else POSITIVE_SUFFIXES
        count = negative_variations if is_unknown else positive_variations
        for index, suffix in enumerate(suffixes[:count], start=1):
            variant = copy.deepcopy(record)
            variant["id"] = f"{record['id']}_v{index:02d}"
            variant["input"]["incantation"] = f"{record['input']['incantation']}{suffix}"
            variant["provenance"] = {
                "type": "SYNTHETIC",
                "source_id": record["provenance"]["source_id"],
                "verified": False,
            }
            if "source_url" in record["provenance"]:
                variant["provenance"]["source_url"] = record["provenance"]["source_url"]
            expanded.append(variant)
    return expanded


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/training-spells.jsonl"))
    parser.add_argument("--additions", type=Path, default=Path("data/seed-spells.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/training-spells.expanded.jsonl"))
    args = parser.parse_args()

    records = load_records(args.input)
    if args.additions.exists():
        records.extend(load_records(args.additions))
    expanded = expand_records(records)
    write_jsonl(expanded, args.output)
    print({"base_records": len(records), "expanded_records": len(expanded), "output": str(args.output)})


if __name__ == "__main__":
    main()
